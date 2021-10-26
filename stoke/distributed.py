# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles distributed related classes -- mixin style"""

import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import Enum
from typing import List, Optional, Tuple, Union

import deepspeed as ds
import horovod.torch as hvd
import torch
from deepspeed.utils.distributed import mpi_discovery
from fairscale.optim.oss import OSS

from stoke.configs import ClipGradConfig, ClipGradNormConfig
from stoke.extensions import (
    DistributedHandlerEnum,
    FairscaleFSDPExtension,
    FairscaleSDDPExtension,
)
from stoke.utils import unrolled_print


class BaseDistributed(ABC):
    """Base class for distributed backends

    This class handles common functionality for all of the different distributed backends including setup, loss sync,
    gradient accumulation context, step context and various properties/attributes related to distributed frameworks

    Attributes
    ----------
    device_id
    initialized
    rank
    world_size
    _batch_size_per_device: int
        batch size per device or for non-distributed the overall batch size
    _device_id: int, default: None
        Current device id
    _info_rank: Union[int, List[int]]
        Which device(s) to print information
    _name: str
        name of current backend
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self,
        device_id: Optional[Union[int, str]],
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        name: str,
        verbose: bool = True,
    ):
        """Init for BaseDistributed class

        Parameters
        ----------
        device_id: int, default: None
            Current device id
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        name: str
            name of current backend
        verbose: bool, default: True
            flag for Stoke print verbosity

        """
        self._batch_size_per_device = batch_size_per_device
        self._device_id = device_id
        self._info_rank = info_rank
        self._name = name
        self._verbose = verbose

    def _print_info(self):
        """Basic print of backend initialization status

        Returns
        -------
        None

        """
        self._print_device(f"{self._name} Initialized: {self.initialized}")

    def setup_distributed(self):
        """Base setup distributed

        Does nothing as nothing needs to be wrapped

        Returns
        -------
        None

        """
        pass

    def wrap_distributed(
        self,
        model: torch.nn.Module,
        grad_accum: Optional[int],
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Base wrapper for distributed backends

        Does nothing but print as nothing needs to be wrapped

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]], default: None
            current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps

        Returns
        -------
        model: torch.nn.Module
            same as input model
        optimizer: Union[torch.optim.Optimizer, OSS]]
            same as input optimizer

        """
        # Print info if verbose
        if self._verbose:
            self._print_info()
        return model, optimizer

    def detach_and_sync_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device=None,
    ):
        """Takes loss(es) and detaches from the compute graph and syncs across devices if needed (via an all-reduce)

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        Union[float, List[float], Tuple[float]]
            loss(es) that has(have) been detached from the graph

        """
        if isinstance(loss, (list, tuple)):
            return type(loss)(val.item() for val in loss)
        else:
            return loss.item()

    def grad_accum_context(self, model: torch.nn.Module):
        """Returns base context for gradient accumulation

        By default no context is used

        Parameters
        ----------
        model: torch.nn.Module
            current model object

        Returns
        -------
        nullcontext()

        """
        return nullcontext()

    def step_context(self, optimizer: Union[torch.optim.Optimizer, OSS]):
        """Returns base context for the step call

        By default no context is used

        Parameters
        ----------
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        nullcontext()

        """
        return nullcontext()

    def clean(self):
        """Base clean call

        Nothing to do here...

        Returns
        -------
        None

        """
        pass

    def _call_init(self):
        """Base init call

        Nothing to do here...

        Returns
        -------
        None

        """
        pass

    def _print_device(self, msg: Union[str, List[str]]):
        """Prints a str of list of strs on the currently set _info_rank

        Internal version of public print_device that always points to the set _info_rank

        Parameters
        ----------
        msg: Union[str, List[str]]
            message(s) to print

        Returns
        -------
        None

        """
        self.print_device(msg=msg, rank=self._info_rank)

    def print_device(
        self,
        msg: Union[str, List[str]],
        rank: Optional[Union[int, List[int]]] = 0,
        single_line: bool = False,
    ):
        """Public facing method to print on specific device ranks

        Parameters
        ----------
        msg: Union[str, List[str]]
            message(s) to print
        rank: Optional[Union[int, List[int]]], default: 0
            device rank to print to (prevents printing on multiple devices in distributed mode)
        single_line: bool, default: False
            if iterable print all on one line space and comma separated

        Returns
        -------
        None

        """
        # Ignore the rank check if the current rank is a non-distributed version
        if self.rank == "cpu" or self.rank == "gpu":
            unrolled_print(msg, single_line=single_line)
        # if it's a list then check the rank against the list
        elif isinstance(rank, list) and self.rank in rank:
            unrolled_print(msg, single_line=single_line)
        # If its an int then check the equality
        elif isinstance(rank, int) and rank == self.rank:
            unrolled_print(msg, single_line=single_line)
        # the else is essentially skip print
        else:
            pass

    def barrier(self):
        """Calls the underlying distributed barrier if available"""
        pass

    @property
    def device_id(self):
        """Returns the current device id"""
        return self._device_id

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        pass

    @property
    @abstractmethod
    def initialized(self):
        pass


class DistributedNullCPU(BaseDistributed):
    def __init__(
        self,
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        verbose: bool = True,
        **kwargs,
    ):
        """Init for DistributedNullCPU

        Parameters
        ----------
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        Notes
        -----
        Device ID set to None as it is not needed for non distributed CPU

        """
        super(DistributedNullCPU, self).__init__(
            device_id="cpu",
            batch_size_per_device=batch_size_per_device,
            info_rank=info_rank,
            name="PyTorch CPU",
            verbose=verbose,
        )

    @property
    def rank(self):
        """Returns current distributed rank

        No rank so return string of cpu
        """
        return "cpu"

    @property
    def world_size(self):
        """Returns current world size"""
        return 1

    @property
    def initialized(self):
        """Returns if distributed backend is initialized correctly"""
        return True


class DistributedNullGPU(BaseDistributed):
    def __init__(
        self,
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        verbose: bool = True,
        **kwargs,
    ):
        """Init for DistributedNullCPU

        Parameters
        ----------
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        Notes
        -----
        Device ID set to the current CUDA device as there is only a single GPU being used

        """
        super(DistributedNullGPU, self).__init__(
            device_id=torch.cuda.current_device(),
            batch_size_per_device=batch_size_per_device,
            info_rank=info_rank,
            name="PyTorch GPU",
            verbose=verbose,
        )

    @property
    def rank(self):
        """Returns current distributed rank

        No rank so return string of gpu
        """
        return "gpu"

    @property
    def world_size(self):
        """Returns current world size"""
        return 1

    @property
    def initialized(self):
        """Returns if distributed backend is initialized correctly"""
        return True


class DistributedDDP(BaseDistributed):
    """Class for using DDP as the distributed backend

    This class handles common functionality for the DDP backend including setup, loss sync,
    gradient accumulation context, step context and various properties/attributes

    Attributes
    ----------
    device_id
    initialized
    rank
    world_size
    _batch_size_per_device: int
        batch size per device or for non-distributed the overall batch size
    _ddp_config: DDPConfig
        Configuration object for DDP backend
    _ddp_handler
        wrapper method that will modify the DDP instance to use SDDP if flagged
    _device_id: int, default: None
        Current device id
    _info_rank: Union[int, List[int]]
        Which device(s) to print information
    _name: str
        name of current backend
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self,
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        verbose: bool = True,
        **kwargs,
    ):
        """Init call for DistributedDDP

        Parameters
        ----------
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here ddp_config, sharded_config, or fully_sharded_config might be passed in

        """
        self._ddp_config = kwargs["ddp_config"]
        super(DistributedDDP, self).__init__(
            device_id=self._ddp_config.local_rank,
            batch_size_per_device=batch_size_per_device,
            info_rank=info_rank,
            name="PyTorch DDP",
            verbose=verbose,
        )
        # This creates the wrapper method depending on DDP or SDDP
        self._ddp_handler = self._create_ddp_handler(kwargs)(
            verbose=self._verbose,
            sddp_config=kwargs["sharded_config"],
            fsdp_config=kwargs["fully_sharded_config"],
            ddp_config=self._ddp_config,
        )

    @staticmethod
    def _create_ddp_handler(kwargs: dict):
        """Determines which DDP related class to use based on the kwarg config passed through

        Parameters
        ----------
        kwargs: dict
            Extra arguments from the __init__ call

        Returns
        -------
        FairscaleSDDPExtension or BaseDDP

        """
        if kwargs["sharded_config"] is not None:
            return DistributedHandlerEnum.sddp.value
        elif kwargs["fully_sharded_config"] is not None:
            return DistributedHandlerEnum.fsdp.value
        else:
            return DistributedHandlerEnum.base.value

    def _call_init(self):
        """Does any backend initialization work related to DDP setup

        Borrows code from DeepSpeed to setup DDP via openMPI
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/distributed.py

        Returns
        -------
        None

        """
        # Borrowing a bit of code from deepspeed
        required_env = [
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_RANK",
        ]
        if self._ddp_config.auto_mpi_discovery and not all(
            map(lambda v: v in os.environ, required_env)
        ):
            try:
                from mpi4py import MPI

                mpi_discovery(verbose=True)
            except ImportError as e:
                print(
                    e,
                    ": mpi4py cannot be imported -- please install Stoke with the MPI option (pip install stoke[mpi])",
                )
        # Initialize call for DDP
        torch.distributed.init_process_group(
            backend=self._ddp_config.backend, init_method=self._ddp_config.init_method
        )

    def setup_distributed(self):
        """Handles any underlying DDP setup post init

        Returns
        -------
        None

        """
        # Set the device rank
        torch.cuda.set_device(self._device_id)
        # Call the init fnc here after device id is set
        self._call_init()

    def wrap_distributed(
        self,
        model: torch.nn.Module,
        grad_accum: Optional[int],
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Overrides base implementation for wrapping with either DDP, Fairscale SDDP or Fairscale FSDP

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]], default: None
            current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Wrapped optimizer object

        """
        self._print_device(f"{self._name} Class: {type(self._ddp_handler).__name__}")
        # Print info if verbose
        if self._verbose:
            self._print_info()
            self._print_device(
                [
                    f"{self._name} -- Device ID: {torch.cuda.current_device()}",
                    f"{self._name} -- Rank: {self.rank}",
                ]
            )
        if self._ddp_config.convert_to_sync_batch_norm:
            self.print_device(
                f"Converting all BatchNorm*D layers to torch.nn.SyncBatchNorm layers..."
            )
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
        if self._verbose and isinstance(
            self._ddp_handler, (FairscaleSDDPExtension, FairscaleFSDPExtension)
        ):
            self._print_device(
                f"Wrapped PyTorch DDP with {type(self._ddp_handler).__name__}"
            )
        # Pass through to the handler for DDP wrappers
        model, optimizer = self._ddp_handler.handle_ddp(
            model=model, optimizer=optimizer, grad_accum=grad_accum, rank=self.rank
        )
        return model, optimizer

    def detach_and_sync_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device=None,
    ):
        """Takes loss(es) and detaches from the compute graph and syncs across devices if needed (via an all-reduce)

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        Union[float, List[float], Tuple[float]]
            loss(es) that has(have) been synced across multiple devices and detached from the graph

        """
        if isinstance(loss, (list, tuple)):
            return type(loss)(
                self._single_detach_and_sync_loss(val, device) for val in loss
            )
        else:
            return self._single_detach_and_sync_loss(loss, device)

    def _single_detach_and_sync_loss(self, loss: torch.Tensor, device=None):
        """Take a single loss and detach it from the compute graph and sync across devices if needed

        Parameters
        ----------
        loss: torch.Tensor
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        float
            detached, synced, and mean calculated across devices

        """
        # map to the same device the loss is on pre detach if not set
        if device is None:
            device = loss.device
        detached_loss = loss.item()
        with torch.no_grad():
            loss_tensor = torch.tensor(detached_loss, device=device, dtype=loss.dtype)
            # Make sure everyone is synced before calling all reduce
            torch.distributed.barrier()
            # Loss tensor is worker specific so all_reduce (and SUM)
            torch.distributed.all_reduce(loss_tensor)
            # Detach and divide by the world size to get the mean on each device
            return loss_tensor.item() / self.world_size

    def grad_accum_context(self, model: torch.nn.Module):
        """Return the context to wrap the gradient accumulation steps

        DDP: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html (Skip unnecessary all-reduce(s))
        SDDP: https://fairscale.readthedocs.io/en/latest/api/nn/sharded_ddp.html
        FSDP: https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html

        Parameters
        ----------
        model: torch.nn.Module
            current model object

        Returns
        -------
        no_sync() context if no_sync flag in config to prevent un-needed communication overhead when using gradient
        accumulation else nullcontext

        """
        if self._verbose and self._ddp_config.no_sync:
            self._print_device("DDP Using no sync context")
        context = model.no_sync() if self._ddp_config.no_sync else nullcontext()
        return context

    def barrier(self):
        """Calls the underlying distributed barrier if available"""
        torch.distributed.barrier()

    @property
    def rank(self):
        """Returns current distributed rank"""
        return torch.distributed.get_rank()

    @property
    def world_size(self):
        """Returns current world size"""
        return torch.distributed.get_world_size()

    @property
    def initialized(self):
        """Returns if distributed backend is initialized correctly"""
        return torch.distributed.is_initialized()

    def clean(self):
        """Cleans up at the end of a DDP run"""
        torch.distributed.destroy_process_group()


class DistributedDeepspeed(BaseDistributed):
    """Class for using Deepspeed as the distributed backend

    This class handles common functionality for the deepspeed backend including setup, loss sync,
    gradient accumulation context, step context and various properties/attributes

    Attributes
    ----------
    device_id
    initialized
    rank
    world_size
    _batch_size_per_device: int
        batch size per device or for non-distributed the overall batch size
    _deepspeed_config: DeepspeedConfig
        Configuration object for Deepspeed backend
    _device_id: int, default: None
        Current device id
    _info_rank: Union[int, List[int]]
        Which device(s) to print information
    _name: str
        name of current backend
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self,
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        verbose: bool = True,
        **kwargs,
    ):
        """Init call for DistributedDeepspeed

        Parameters
        ----------
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here deepspeed_config, grad_accum_steps or grad_clip
            might be passed in

        """
        self._deepspeed_config = kwargs["deepspeed_config"]
        # Call init first to pass local rank to super
        self._call_init()
        # Forward device to super -- should be set from MPI lookup that is called
        super(DistributedDeepspeed, self).__init__(
            device_id=int(os.environ["LOCAL_RANK"]),
            batch_size_per_device=batch_size_per_device,
            info_rank=info_rank,
            name="Deepspeed",
            verbose=verbose,
        )
        self._deepspeed_init_config = self._handle_deepspeed_configs(
            grad_accum_steps=kwargs["grad_accum_steps"], grad_clip=kwargs["grad_clip"]
        )

    def _call_init(self):
        """Does any backend initialization work related to deepspeed setup

        Returns
        -------
        None

        """
        ds.init_distributed(
            dist_backend=self._deepspeed_config.dist_backend,
            auto_mpi_discovery=self._deepspeed_config.auto_mpi_discovery,
            distributed_port=self._deepspeed_config.distributed_port,
            verbose=self._deepspeed_config.verbose,
            init_method=self._deepspeed_config.init_method,
        )

    def setup_distributed(self):
        """Handles any underlying deepspeed setup post init

        Returns
        -------
        None

        """
        # Set the device rank
        torch.cuda.set_device(self._device_id)

    def wrap_distributed(
        self,
        model: torch.nn.Module,
        grad_accum: Optional[int],
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Overrides base implementation for wrapping with Deepspeed

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]], default: None
            current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Wrapped optimizer object

        """
        # Print info if verbose
        if self._verbose:
            self._print_info()
            self._print_device(
                f"{self._name} -- Device ID: {torch.cuda.current_device()}"
            )
            self._print_device(f"{self._name} -- Rank: {self.rank}")

        model, optimizer, _, _ = ds.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
            config_params=self._deepspeed_init_config,
        )
        return model, optimizer

    def _handle_deepspeed_configs(
        self,
        grad_accum_steps: int,
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]],
    ):
        """Handles building the dictionary of configs that the deepspeed initialize call expects

        https://www.deepspeed.ai/docs/config-json/

        Parameters
        ----------
        grad_accum_steps: int
            number of gradient accumulation steps
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]], default: None
            gradient clipping config objects

        Returns
        -------
        dict
            All deepspeed parameters merged together from individual pieces

        """
        # empty dict to start
        ds_config = {}
        # Map batch size stuff -- need to define 2/3
        ds_config.update(self._map_ds_batch_configs(grad_accum_steps=grad_accum_steps))
        # Skip optimizer & skip scheduler
        # Map communication
        ds_config.update(self._map_ds_communication_configs())
        # Map FP16 and add enabled flag if selected
        ds_config.update(self._map_ds_fp16_configs())
        # Map grad clipping
        ds_config.update(self._map_ds_grad_clip_configs(grad_clip=grad_clip))
        # Map zero -- internally map param offloading and optimizer offloading to zero
        ds_config.update(self._map_ds_zero_configs())
        # Map aio
        ds_config.update(self._map_ds_aio_configs())
        # Map logging
        ds_config.update(self._map_ds_logging_configs())
        # Map flops -- enabled
        ds_config.update(self._map_ds_flops_configs())
        # Map activation checkpointing
        ds_config.update(self._map_ds_activation_checkpointing_configs())
        # Map tensorboard
        ds_config.update(self._map_ds_tensorboard_config())
        # Map PLD
        ds_config.update(self._map_ds_pld_config())
        return ds_config

    def _map_ds_pld_config(self):
        """Maps progressive layer drop parameters

        https://www.deepspeed.ai/tutorials/progressive_layer_dropping/
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/constants.py#L293

        Returns
        -------
        dict
            pld parameters or enabled false dict

        """
        if self._deepspeed_config.progressive_layer_drop is not None:
            map_dict = {
                v.name: getattr(self._deepspeed_config.progressive_layer_drop, v.name)
                for v in self._deepspeed_config.progressive_layer_drop.__attrs_attrs__
            }
            map_dict.update({"enabled": True})
            return {"progressive_layer_drop": map_dict}
        else:
            return {"progressive_layer_drop": {"enabled": False}}

    def _map_ds_tensorboard_config(self):
        """Maps tensorboard related parameters

        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/constants.py#L268

        Returns
        -------
        dict
            tensorboard parameters or enabled false dict

        """
        if self._deepspeed_config.tensorboard is not None:
            map_dict = {
                v.name: getattr(self._deepspeed_config.tensorboard, v.name)
                for v in self._deepspeed_config.tensorboard.__attrs_attrs__
            }
            map_dict.update({"enabled": True})
            return {"tensorboard": map_dict}
        else:
            return {"tensorboard": {"enabled": False}}

    def _map_ds_grad_clip_configs(
        self, grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]]
    ):
        """Maps grad clipping related parameters

        https://www.deepspeed.ai/docs/config-json/#gradient-clipping

        Parameters
        ----------
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]], default: None
            gradient clipping config objects

        Returns
        -------
        dict
            gradient clipping parameters or empty dict

        """
        if grad_clip is not None:
            if isinstance(grad_clip, ClipGradNormConfig):
                return {"gradient_clipping": grad_clip.max_norm}
            else:
                raise ValueError(
                    f"Deepspeed does not currently support "
                    f'{type(grad_clip).__name__.replace("Config", "")}'
                )
        else:
            return {}

    def _map_ds_logging_configs(self):
        """Maps logging related parameters

        https://www.deepspeed.ai/docs/config-json/#logging

        Returns
        -------
        dict
            logging parameters or empty dict

        """
        return {
            "steps_per_print": self._deepspeed_config.steps_per_print,
            "dump_state": self._deepspeed_config.dump_state,
            "wall_clock_breakdown": self._deepspeed_config.wall_clock_breakdown,
        }

    def _map_ds_activation_checkpointing_configs(self):
        """Maps activation checkpointing related parameters

        https://www.deepspeed.ai/docs/config-json/#activation-checkpointing

        Returns
        -------
        dict
            activation checkpointing parameters or empty dict

        """
        if self._deepspeed_config.activation_checkpointing is not None:
            map_dict = {
                v.name: getattr(self._deepspeed_config.activation_checkpointing, v.name)
                for v in self._deepspeed_config.activation_checkpointing.__attrs_attrs__
            }
            return {"activation_checkpointing": map_dict}
        else:
            return {}

    def _map_ds_flops_configs(self):
        """Maps flops related parameters

        https://www.deepspeed.ai/docs/config-json/#flops-profiler

        Returns
        -------
        dict
            flops parameters or enabled false dict

        """
        if self._deepspeed_config.flops_profiler is not None:
            map_dict = {
                v.name: getattr(self._deepspeed_config.flops_profiler, v.name)
                for v in self._deepspeed_config.flops_profiler.__attrs_attrs__
            }
            map_dict.update({"enabled": True})
            return {"flops_profiler": map_dict}
        else:
            return {"flops_profiler": {"enabled": False}}

    def _map_ds_aio_configs(self):
        """Maps async i/o related parameters

        https://www.deepspeed.ai/docs/config-json/#asynchronous-io

        Returns
        -------
        dict
            async i/o parameters or empty dict

        """
        if self._deepspeed_config.aio is not None:
            map_dict = {
                v.name: getattr(self._deepspeed_config.aio, v.name)
                for v in self._deepspeed_config.aio.__attrs_attrs__
            }
            return {"aio": map_dict}
        else:
            return {}

    def _map_ds_zero_configs(self):
        """Maps ZeRO related parameters

        https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training

        Returns
        -------
        dict
            ZeRO related parameters

        """
        map_dict = {}
        for v in self._deepspeed_config.zero_optimization.__attrs_attrs__:
            if v.name == "offload_optimizer":
                map_dict.update(self._map_ds_offload_optimizer_configs())
            elif v.name == "offload_param":
                map_dict.update(self._map_ds_offload_param_configs())
            # Just map the rest since the name:value is correct
            else:
                map_dict.update(
                    {v.name: getattr(self._deepspeed_config.zero_optimization, v.name)}
                )
        # Default overlap com to True for ZeRO stage 3
        map_dict["overlap_comm"] = (
            True if map_dict["stage"] == 3 else map_dict["overlap_comm"]
        )
        return {"zero_optimization": map_dict}

    def _map_ds_offload_param_configs(self):
        """Maps ZeRO parameter offload parameters

        https://www.deepspeed.ai/docs/config-json/#parameter-offloading

        Returns
        -------
        dict
            ZeRO offload parameter parameters

        """
        # Use a bit of introspection to pull out the attrs stuff systematically as the name mapping is correct
        if self._deepspeed_config.zero_optimization.offload_param is not None:
            map_dict = {
                v.name: getattr(
                    self._deepspeed_config.zero_optimization.offload_param, v.name
                )
                for v in self._deepspeed_config.zero_optimization.offload_param.__attrs_attrs__
            }
            return {"offload_param": map_dict}
        else:
            return {"offload_param": None}

    def _map_ds_offload_optimizer_configs(self):
        """Maps ZeRO optimizer offload parameters

        https://www.deepspeed.ai/docs/config-json/#optimizer-offloading

        Returns
        -------
        dict
            ZeRO offload optimizer parameters

        """
        # Use a bit of introspection to pull out the attrs stuff systematically as the name mapping is correct
        if self._deepspeed_config.zero_optimization.offload_optimizer is not None:
            map_dict = {
                v.name: getattr(
                    self._deepspeed_config.zero_optimization.offload_optimizer, v.name
                )
                for v in self._deepspeed_config.zero_optimization.offload_optimizer.__attrs_attrs__
            }
            # Set some post init values
            map_dict["pipeline"] = (
                map_dict["pipeline_read"] or map_dict["pipeline_write"]
            )
            return {"offload_optimizer": map_dict}
        else:
            return {"offload_optimizer": None}

    def _map_ds_fp16_configs(self):
        """Maps FP16 related parameters

        https://www.deepspeed.ai/docs/config-json/#fp16-training-options

        Returns
        -------
        dict
            fp16 related parameters or enabled false dict

        """
        if self._deepspeed_config.fp16 is not None:
            # Use a bit of introspection to pull out the attrs stuff systematically as the name mapping is correct
            map_dict = {
                v.name: getattr(self._deepspeed_config.fp16, v.name)
                for v in self._deepspeed_config.fp16.__attrs_attrs__
            }
            # Add the enabled flag
            map_dict.update({"enabled": True})
            return {"fp16": map_dict}
        else:
            return {"fp16": {"enabled": False}}

    def _map_ds_batch_configs(self, grad_accum_steps: int):
        """Maps batch size related parameters

        https://www.deepspeed.ai/docs/config-json/#batch-size-related-parameters

        Parameters
        ----------
        grad_accum_steps: int
            number of gradient accumulation steps

        Returns
        -------
        dict
            batch size related parameters

        """
        # Need to define 2/3
        return {
            "train_micro_batch_size_per_gpu": self._batch_size_per_device,
            "gradient_accumulation_steps": grad_accum_steps,
        }

    def _map_ds_communication_configs(self):
        """Maps communication related parameters

        https://www.deepspeed.ai/docs/config-json/#communication-options

        Returns
        -------
        dict
            communication related parameters

        """
        return {
            "fp32_allreduce": self._deepspeed_config.fp32_allreduce,
            "gradient_predivide_factor": self._deepspeed_config.gradient_predivide_factor,
            "prescale_gradients:": self._deepspeed_config.prescale_gradients,
            "sparse_gradients": self._deepspeed_config.sparse_gradients,
        }

    def detach_and_sync_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device=None,
    ):
        """Takes loss(es) and detaches from the compute graph and syncs across devices if needed (via an all-reduce)

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        Union[float, List[float], Tuple[float]]
            loss(es) that has(have) been synced across multiple devices and detached from the graph

        """
        if isinstance(loss, (list, tuple)):
            return type(loss)(
                self._single_detach_and_sync_loss(val, device) for val in loss
            )
        else:
            return self._single_detach_and_sync_loss(loss, device)

    def _single_detach_and_sync_loss(self, loss: torch.Tensor, device=None):
        """Take a single loss and detach it from the compute graph and sync across devices if needed

        Parameters
        ----------
        loss: torch.Tensor
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        float
            detached, synced, and mean calculated across devices

        """
        # map to the same device the loss is on pre detach if not set
        if device is None:
            device = loss.device
        detached_loss = loss.item()
        with torch.no_grad():
            loss_tensor = torch.tensor(detached_loss, device=device, dtype=loss.dtype)
            # Loss tensor is worker specific so all_reduce (and SUM)
            torch.distributed.all_reduce(loss_tensor)
            # Detach and divide by the world size to get the mean on each device
            return loss_tensor.item() / self.world_size

    def barrier(self):
        """Calls the underlying distributed barrier if available"""
        torch.distributed.barrier()

    @property
    def rank(self):
        """Returns current distributed rank"""
        return torch.distributed.get_rank()

    @property
    def world_size(self):
        """Returns current world size"""
        return torch.distributed.get_world_size()

    @property
    def initialized(self):
        """Returns if distributed backend is initialized correctly"""
        return torch.distributed.is_initialized()

    def clean(self):
        """Cleans up at the end of a DDP run"""
        torch.distributed.destroy_process_group()


class DistributedHorovod(BaseDistributed):
    """Class for using Horovod as the distributed backend

    This class handles common functionality for the horovod backend including setup, loss sync,
    gradient accumulation context, step context and various properties/attributes

    Attributes
    ----------
    device_id
    initialized
    rank
    world_size
    _batch_size_per_device: int
        batch size per device or for non-distributed the overall batch size
    _device_id: int, default: None
        Current device id
    _horovod_config: HorovodConfig
        Configuration object for Horovod backend
    _info_rank: Union[int, List[int]]
        Which device(s) to print information
    _name: str
        name of current backend
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self,
        batch_size_per_device: int,
        info_rank: Union[int, List[int]],
        verbose: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        batch_size_per_device: int
            batch size per device or for non-distributed the overall batch size
        info_rank: Union[int, List[int]]
            Which device(s) to print information
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here horovod_config might be passed in

        """
        # Grab the config
        self._horovod_config = kwargs["horovod_config"]
        # Initialize first so the local rank call cal be forwarded to super
        self._call_init()
        super(DistributedHorovod, self).__init__(
            device_id=hvd.local_rank(),
            batch_size_per_device=batch_size_per_device,
            info_rank=info_rank,
            name="Horovod",
            verbose=verbose,
        )
        self._multi_loss = (
            len(kwargs["loss"]) if isinstance(kwargs["loss"], (list, tuple)) else 1
        )

    def _call_init(self):
        """Does any backend initialization work related to horovod setup

        Returns
        -------
        None

        """
        hvd.init()

    def _hvd_convert_to_sync_batch_norm(
        self, module: torch.nn.Module, process_group=None
    ):
        """Replaces all BatchNorm*D layers with horovod.torch.SyncBatchNorm layers

        https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm.convert_sync_batchnorm
        https://nvidia.github.io/apex/_modules/apex/parallel.html#convert_syncbn_model

        Parameters
        ----------
        module: torch.nn.Module
            current model object
        process_group: default: None
            process group to scope synchronization, default is the whole world

        Returns
        -------
        module_output: torch.nn.Module
            modified version of model with all BatchNorm*D layers replaced with horovod.torch.SyncBatchNorm layers

        Notes
        -----
        Borrows heavily from the current torch convert_sync_batchnorm and apex convert_syncbn_model implementations
        only changing the underlying layer type to use the hvd implementation

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = hvd.SyncBatchNorm(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )
            # Handle the copy of affine vars if affine
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            # Handle the swap of running stats
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
        # Iterate recursively and replace
        for name, child in module.named_children():
            module_output.add_module(
                name=name,
                module=self._hvd_convert_to_sync_batch_norm(
                    module=child, process_group=process_group
                ),
            )
        # delete and return
        del module
        return module_output

    def setup_distributed(self):
        """Handles any underlying horovod setup post init

        Returns
        -------
        None

        """
        # Set the device rank
        torch.cuda.set_device(self._device_id)

    def wrap_distributed(
        self,
        model: torch.nn.Module,
        grad_accum: Optional[int],
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Overrides base implementation for wrapping with Horovod

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]], default: None
            current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Wrapped optimizer object

        """
        # Print info if verbose
        if self._verbose:
            self._print_info()
            self._print_device(
                f"{self._name} -- Device ID: {torch.cuda.current_device()}"
            )
            self._print_device(f"{self._name} -- Rank: {self.rank}")
        op_dict = {"Average": hvd.Average, "Sum": hvd.Sum, "Adasum": hvd.Adasum}
        optimizer = hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=model.named_parameters(),
            backward_passes_per_step=grad_accum * self._multi_loss
            if grad_accum is not None
            else self._multi_loss,
            compression=hvd.Compression.fp16
            if self._horovod_config.compression
            else hvd.Compression.none,
            gradient_predivide_factor=self._horovod_config.gradient_predivide_factor,
            op=op_dict.get(self._horovod_config.op),
        )
        # Broadcast the initial variable states from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        return model, optimizer

    def detach_and_sync_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device=None,
    ):
        """Takes loss(es) and detaches from the compute graph and syncs across devices if needed (via an all-reduce)

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        Union[float, List[float], Tuple[float]]
            loss(es) that has(have) been synced across multiple devices and detached from the graph

        """
        if isinstance(loss, (list, tuple)):
            return type(loss)(
                self._single_detach_and_sync_loss(val, device) for val in loss
            )
        else:
            return self._single_detach_and_sync_loss(loss, device)

    def _single_detach_and_sync_loss(self, loss: torch.Tensor, device=None):
        """Take a single loss and detach it from the compute graph and sync across devices if needed

        Parameters
        ----------
        loss: torch.Tensor
            current loss(es) on the device
        device: default: None
            output device of the sync call

        Returns
        -------
        float
            detached, synced, and mean calculated across devices

        """
        # map to the same device the loss is on pre detach if not set
        if device is None:
            device = loss.device
        detached_loss = loss.item()
        with torch.no_grad():
            loss_tensor = torch.tensor(detached_loss, device=device, dtype=loss.dtype)
            # Make sure everyone is synced before the all-reduce
            # Horovod doesn't have a native barrier so lean on join to take care of it
            # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
            hvd.join()
            # Loss tensor is worker specific so allreduce -- force SUM from Horovod
            sum_tensor = hvd.allreduce(loss_tensor, op=hvd.Sum)
            # Detach and divide by the world size to get the mean on each device
            return sum_tensor.item() / self.world_size

    def step_context(self, optimizer: Union[torch.optim.Optimizer, OSS]):
        """Return the context to wrap the step call

        Parameters
        ----------
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        skip_synchronize() context to prevent un-needed communication overhead when using gradient accumulation

        """
        # Hidden here -- Horovod docs are terrible
        # https://horovod.readthedocs.io/en/latest/api.html#horovod.torch.DistributedOptimizer
        if self._verbose:
            self._print_device(
                "Horovod skipping synchronize as it was triggered pre grad-clip"
            )
        return optimizer.skip_synchronize()

    def barrier(self):
        """Calls the underlying distributed barrier if available"""
        # Horovod doesn't have a native barrier so lean on join to take care of it
        # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
        hvd.join()

    @property
    def rank(self):
        """Returns current distributed rank"""
        return hvd.rank()

    @property
    def world_size(self):
        """Returns current world size"""
        return hvd.size()

    @property
    def initialized(self):
        """Returns if distributed backend is initialized correctly"""
        return hvd.is_initialized()


class RunnerDistEnum(Enum):
    """Enum for building the runtime object with distributed functionality"""

    cpu = DistributedNullCPU
    gpu = DistributedNullGPU
    ddp = DistributedDDP
    horovod = DistributedHorovod
    deepspeed = DistributedDeepspeed
