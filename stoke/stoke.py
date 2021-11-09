# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""API interface to Stoke that handles any necessary config, context, setup etc."""

from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from uuid import uuid4

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import ShardedDataParallel as SDDP
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler, Sampler

from stoke.configs import (
    AMPConfig,
    ApexConfig,
    ClipGradConfig,
    ClipGradNormConfig,
    DDPConfig,
    DeepspeedConfig,
    FairscaleFSDPConfig,
    FairscaleOSSConfig,
    FairscaleSDDPConfig,
    HorovodConfig,
    StokeOptimizer,
)
from stoke.data import StokeDataLoader
from stoke.distributed import RunnerDistEnum
from stoke.extensions import RunnerOptimizerEnum
from stoke.fp16 import RunnerFP16Enum
from stoke.io_ops import RunnerIOEnum
from stoke.status import DistributedOptions, FP16Options, StokeStatus
from stoke.utils import (
    ParamNormalize,
    T_co,
    _collate_fn_t,
    _worker_init_fn_t,
    zero_optimizer_grads,
)


class Stoke:
    """High level stoke object that manages all necessary configs and provides a unified interface to ops

    This is the main class within Stoke. Functionally it manages all interfaces to the necessary wrapped ops (model,
    loss, backward, step), provides helper functions, and dynamically constructs the runtime that handles the
    combinatorics problem of underlying frameworks (DDP, Horovod, Deepspeed, Fairscale),
    mixed-precision (AMP or APEX) and devices (CPU or GPU)

    Attributes
    ----------
    amp_config
    apex_config
    batch_size
    cuda
    ddp_config
    deepspeed_config
    distributed
    effective_batch_size
    ema_loss
    fp16
    fsdp_config
    fully_sharded
    gpu
    grad_accum
    grad_clip
    horovod_config
    is_amp
    is_apex
    is_ddp
    is_deepspeed
    is_horovod
    loss_access
    model_access
    nccl
    num_model_parameters
    optimizer
    oss
    oss_config
    rank
    scaler
    sddp_config
    sharded
    status
    world_size
    _agg_loss: Union[float, List[float], Tuple[float]]
        aggregated loss for grad accumulation (single or multiple losses)
    _backward_steps: int
        Number of times gradients have been calculated on a batch of samples (calls to backward)
    _grad_accum_counter: int
        counter for grad accumulation steps
    _loss: Union[Callable, List[Callable], Tuple[Callable]]
        callable function that calculates a loss from the model outputs
    _last_step_loss: list, tuple, or float
        last loss step calculation aggregated over device(s)
    _model: torch.nn.Module
        instance of torch.nn.Module for Stoke to handle
    _optimizer: StokeOptimizer
        StokeOptimizer config object that describes the torch.optim.Optimizer and it's kwargs
    _optimizer_steps: int
        Number of times step has been called on the optimizer
    _runner: StokeRunner
        the dynamically created runtime object that handles all ops
    _status: StokeStatus
        StokeStatus object that sets and maintains the current configuration
    _verbose: bool
        print verbosity
    _rolling_loss_steps: int
        number of steps that have been called for the rolling loss
    _rolling_mean_loss: list, tuple, or float
        current ema loss
    _ema_weight: float
        weight used for any ema calculation on metrics

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: StokeOptimizer,
        loss: Union[Callable, List[Callable], Tuple[Callable]],
        batch_size_per_device: int,
        grad_accum_steps: Optional[int] = 1,
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]] = None,
        gpu: bool = False,
        fp16: Optional[FP16Options] = None,
        distributed: Optional[DistributedOptions] = None,
        fairscale_oss: bool = False,
        fairscale_sddp: bool = False,
        fairscale_fsdp: bool = False,
        configs: Optional[
            List[
                Union[
                    AMPConfig,
                    ApexConfig,
                    DDPConfig,
                    DeepspeedConfig,
                    FairscaleOSSConfig,
                    FairscaleSDDPConfig,
                    FairscaleFSDPConfig,
                    HorovodConfig,
                ]
            ]
        ] = None,
        info_rank: Optional[Union[int, List[int]]] = 0,
        verbose: bool = True,
        ema_weight: float = 0.1,
    ):
        """Init for Stoke class object

        Parameters
        ----------
        model: torch.nn.Module
            PyTorch model
        optimizer: StokeOptimizer
            Optimizer configuration
        loss: Union[Callable, List[Callable], Tuple[Callable]]
            Callable loss function or functions
        batch_size_per_device: int
            Batch size at the single device level
        grad_accum_steps: Optional[int], default: 1
            Number of gradient accumulation steps
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]], default: None
            Gradient clipping configuration
        gpu: bool, default: False
            flag to use GPU device(s)
        fp16: Optional[FP16Options], default: None
            Choice of mixed-precision backend
        distributed: Optional[DistributedOptions], default: None
            Choice of distributed backend
        fairscale_oss: bool, default: False
            Flag to activate optimizer state sharding using Fairscale
        fairscale_sddp: bool, default: False
            Flag to activate sharded DDP using Fairscale
        fairscale_fsdp: bool, default: False
            Flag to activate fully sharded DDP using Fairscale
        configs: Optional[List[Union[AMPConfig, ApexConfig, DDPConfig, DeepspeedConfig, FairscaleOSSConfig, FairscaleSDDPConfig, FairscaleFSDPConfig, HorovodConfig]], default: None
            Configuration objects for runtimes
        info_rank: Optional[Union[int, List[int]]], default = 0
            Constrain prints to specific devices
        verbose: bool, default: True
            Flag for verbosity
        ema_weight: float, default: 0.5
            weight used for any ema calculation on metrics

        """
        # Verbosity
        self._verbose = verbose
        # Info rank
        self._info_rank = info_rank
        # EMA
        self._ema_weight = ema_weight
        # Setup the StokeState
        self._status = StokeStatus(
            batch_size_per_device=batch_size_per_device,
            grad_accum=grad_accum_steps,
            grad_clip=grad_clip,
            gpu=gpu,
            fp16=fp16,
            distributed=distributed,
            fairscale_oss=fairscale_oss,
            fairscale_sddp=fairscale_sddp,
            fairscale_fsdp=fairscale_fsdp,
            configs=configs,
        )
        # Run some checks
        self._model = self._check_model(model)
        self._optimizer = self._check_optimizer(optimizer)
        self._loss = self._check_loss(loss)
        # Dynamically construct the StokeRunner from the StokeStatus
        self._runner, class_info = self._build_runner()
        # Setup distributed backend
        self._runner.setup_distributed()
        # Post here the runner will have the print_device function that is mapped to the self.print here
        # as it needs rank to be accessible before working
        if self._verbose:
            dev_id = (
                self.rank
                if (self.rank == "cpu" or self.rank == "gpu")
                else self._info_rank
            )
            self.print(f"Printing verbose information on rank(s): {dev_id}")
            # Print the runner class info from the mixins
            self.print(class_info)
        # Possibly place model on GPU depending on StokeStatus -- before wrap calls
        self._place_model_on_gpu()
        # Handle the wrap ops in the correct order
        self._handle_ordered_wrap_ops(optimizer=optimizer)
        # Create some tracking vars
        self._grad_accum_counter = 0
        self._optimizer_steps = 0
        self._backward_steps = 0
        self._last_step_loss = self._set_loss_to_zero()
        self._agg_loss = self._set_loss_to_zero()
        self._rolling_mean_loss = self._set_loss_to_zero()
        self._rolling_loss_steps = 0
        # Set post-init status variables
        self._status.set_post_init_values(world_size=self.world_size)
        # Print the final configuration
        if self._verbose:
            self.print(msg=self._status)

    def _wrap_optimizer_then_model(self, optimizer: StokeOptimizer):
        """Handles wrapping of optimizer then the model

        This holds only for SDDP, Horovod, and APEX as these need to use an instantiated optimizer before wrapped
        methods are called

        Parameters
        ----------
        optimizer: StokeOptimizer
            Optimizer configuration

        Returns
        -------
        None

        """
        # Build the optimizer
        self._optimizer = self._runner.build_optimizer(
            optimizer=optimizer["optimizer"],
            optimizer_kwargs=optimizer["optimizer_kwargs"],
            model=self._model,
        )
        # Setup/Initialize FP16 backend -- in this case the optimizer is passed through
        self._runner.wrap_fp16(model=self._model, optimizer=self._optimizer)
        # Wrap with distributed backend -- in this case the optimizer is passed through
        self._model, self._optimizer = self._runner.wrap_distributed(
            model=self._model, grad_accum=self.grad_accum, optimizer=self._optimizer
        )

    def _wrap_model_then_optimizer(self, optimizer: StokeOptimizer):
        """Handles wrapping of model then optimizer

        Parameters
        ----------
        optimizer: StokeOptimizer
            Optimizer configuration

        Returns
        -------
        None

        """
        # Wrap with distributed backend -- in this case the optimizer is passed as None since it doesn't exist yet
        # don't use the return for the optimizer in this case
        self._model, _ = self._runner.wrap_distributed(
            model=self._model, grad_accum=self.grad_accum, optimizer=None
        )
        # Setup/Initialize FP16 backend -- in this case the optimizer is passed as None since it doesn't exist yet
        self._runner.wrap_fp16(model=self._model, optimizer=None)
        # Build the optimizer
        self._optimizer = self._runner.build_optimizer(
            optimizer=optimizer["optimizer"],
            optimizer_kwargs=optimizer["optimizer_kwargs"],
            model=self._model,
        )

    def _handle_ordered_wrap_ops(self, optimizer: StokeOptimizer):
        """Handles wrapping model, using FP16, and wrapping optimizer in the correct order depending on Stoke Status

        Parameters
        ----------
        optimizer: StokeOptimizer
            Optimizer configuration

        Returns
        -------
        None

        """
        # if SDDP + OSS, Horovod, and APEX then we need to make sure that the optimizer gets wrapped before the model
        # gets wrapped, all other models follow standard DDP paradigm (or their own DeepSpeed)
        if (self.sharded and self.oss) or self.is_apex or self.is_horovod:
            self._wrap_optimizer_then_model(optimizer=optimizer)
        else:
            self._wrap_model_then_optimizer(optimizer=optimizer)

    def _check_accum(self):
        """Checks if the current step is the last accumulation step

        Returns
        -------
        bool

        """
        return (self._grad_accum_counter + 1) % (self.grad_accum + 1) == 0

    def _check_pre_accum(self):
        """Checks if we are at the pre-accumulate step

        Returns
        -------
        bool

        """
        return (self._grad_accum_counter + 1) % (self.grad_accum + 1) == self.grad_accum

    def _set_loss_to_zero(self):
        """Used to set a loss tracker to zero depending on the type

        Returns
        -------
        float or list or tuple of reset loss

        """
        return (
            type(self._loss)([0.0] * len(self._loss))
            if isinstance(self._loss, (list, tuple))
            else 0.0
        )

    def reset_ema(self):
        """Used to reset the current state of the rolling mean loss

        Returns
        -------
        None

        """
        self._rolling_mean_loss = self._set_loss_to_zero()
        self._rolling_loss_steps = 0

    def print_ema_loss(
        self, prepend_msg: str = "Current EMA Loss", single_line: bool = False
    ):
        """Prints the current ema loss synced across all devices

        Handles single or multiple losses. Prints only on devices specified by self._info_rank

        Parameters
        ----------
        prepend_msg: str, default: "Current EMA Loss"
            message prepend to print
        single_line: bool, default: False
            if iterable print all on one line space and comma separated

        Returns
        -------
        None

        """
        if isinstance(self._rolling_mean_loss, (list, tuple)):
            print_vals = [
                f"{prepend_msg} {idx}: {val:.3f}"
                for idx, val in enumerate(self._rolling_mean_loss)
            ]
            self.print(print_vals, single_line=single_line)
        else:
            self.print(f"{prepend_msg}: {self._rolling_mean_loss:.3f}")

    def print_mean_accumulated_synced_loss(
        self,
        prepend_msg: str = "Mean Accumulated & Synced Loss",
        pre_backwards: bool = True,
        single_line: bool = False,
    ):
        """Prints the mean accumulated and device synced loss only after the grad accumulation step

        Handles single or multiple losses. Prints only on devices specified by self._info_rank

        Parameters
        ----------
        prepend_msg: str, default: "Mean Accumulated & Synced Loss"
            message prepend to print
        pre_backwards: bool, default: True
            if being called pre backward step
        single_line: bool, default: False
            if iterable print all on one line space and comma separated

        Returns
        -------
        None

        """
        check_fn = self._check_pre_accum if pre_backwards else self._check_accum
        if check_fn():
            if isinstance(self._agg_loss, (list, tuple)):
                print_vals = self._scale_agg_loss()
                self.print(print_vals, single_line=single_line)
            else:
                self.print(f"{prepend_msg}: {self._scale_agg_loss():.3f}")

    def _scale_agg_loss(self):
        """Scales the mean aggregated loss by  grad accum

        Returns
        -------
        scale_vals: list or float of mean aggregated loss

        """
        if isinstance(self._agg_loss, (list, tuple)):
            scale_vals = [
                val / self.grad_accum for idx, val in enumerate(self._agg_loss)
            ]
        else:
            scale_vals = self._agg_loss / self.grad_accum
        return scale_vals

    def print_synced_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        prepend_msg: str = "Step Synced Loss",
        device=None,
        single_line: bool = False,
    ):
        """Prints a device synced loss at a single step

        Handles single or multiple losses. Prints only on devices specified by self._info_rank

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es) on the device
        prepend_msg: str, default: "Step Synced Loss"
            message prepend to print
        device: default: None
            specify the device to place the synced loss on (defaults to same device)
        single_line: bool, default: False
            if iterable print all on one line space and comma separated

        Returns
        -------
        None

        """
        printable_loss = self.detach_and_sync_loss(loss, device)
        if isinstance(printable_loss, (list, tuple)):
            print_vals = [
                f"{prepend_msg} {idx}: {val * self.grad_accum:.3f}"
                for idx, val in enumerate(printable_loss)
            ]
            self.print(print_vals, single_line=single_line)
        else:
            self.print(msg=f"{prepend_msg}: {printable_loss * self.grad_accum:.3f}")

    def print_on_devices(
        self, msg: Union[str, List[str]], rank: Optional[Union[int, List[int]]] = 0
    ):
        """Wraps runner print interface for shorter semantics

        Parameters
        ----------
        msg: str
            message to print
        rank: Union[int, List[int]], default: 0
            which ranks to print on

        Returns
        -------
        None

        """
        self._runner.print_device(msg=msg, rank=rank)

    def print(self, msg: Union[str, List[str]], single_line: bool = False):
        """Wraps the runners print device and forces print on the _info_rank attribute(s)

        Parameters
        ----------
        msg: str
            message to print
        single_line: bool, default: False
            if iterable print all on one line space and comma separated

        Returns
        -------
        None

        """
        self._runner.print_device(
            msg=msg, rank=self._info_rank, single_line=single_line
        )

    @staticmethod
    def _check_model(model: torch.nn.Module):
        """Verifies the type of the model

        Parameters
        ----------
        model: torch.nn.Module
            current torch model

        Returns
        -------
        None

        """
        # Check if the model is an nn.Module such that it has a forward method
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Stoke -- Model is not of type torch.nn.Module, currently {type(model)}"
            )
        return model

    @staticmethod
    def _check_optimizer(optimizer: StokeOptimizer):
        """Verifies the type of the optimizer

        Parameters
        ----------
        optimizer: StokeOptimizer
            Current optimizer configuration TypedDict (aka dict)

        Returns
        -------
        None

        """
        if not isinstance(optimizer, dict):
            raise TypeError(
                f"Stoke -- Optimizer is not of type torch.optim.Optimizer, currently {type(optimizer)}"
            )
        return optimizer

    def _check_loss(self, loss: Union[Callable, List[Callable], Tuple[Callable]]):
        """Checks to make sure the loss function(s) is/are callable

        Parameters
        ----------
        loss: Union[Callable, List[Callable], Tuple[Callable]]
            Current callable loss(es)

        Returns
        -------
        None

        """
        if isinstance(loss, (list, tuple)):
            loss = [self._check_loss(val) for val in loss]
            return loss
        elif isinstance(loss, Callable):
            return loss
        else:
            raise TypeError(
                f"Stoke -- Loss is not of type Callable, currently {type(loss)}"
            )

    def _place_model_on_gpu(self):
        """Automatically moves the model to GPU device(s)

        Returns
        -------
        None

        """
        if self.gpu and not self.is_deepspeed:
            if self._verbose:
                self.print(f"Automatically handling moving model to GPU(s)...")
            self._model.cuda()

    def _build_runner(self):
        """Builds the runtime object from the mixin style classes

        Mixes the distributed class, fp16 class, and optimizer class into a single object such that all can be called
        from the same interface. Prevents verbose calls to multiple objects and unifies all functionality under a
        a single interface. Might prevent some IDE type-hinting as it's dynamic

        Returns
        -------
        StokeRunner
            runtime runner object

        """
        # Get the classes
        dist_class = self._get_distributed_mixin()
        fp16_class = self._get_fp16_mixin()
        optimizer_class = self._get_optimizer_mixin()
        io_class = self._get_io_mixin()

        # Python MRO hack to make sure the inits of all the Mixin classes get called
        def __multiple_mixin_init__(*args, **kwargs):
            dist_class.__init__(*args, **kwargs)
            fp16_class.__init__(*args, **kwargs)
            optimizer_class.__init__(*args, **kwargs)
            io_class.__init__(*args, **kwargs)

        # Configs pass through
        kwargs_dict = {
            "amp_config": self.amp_config,
            "apex_config": self.apex_config,
            "ddp_config": self.ddp_config,
            "deepspeed_config": self.deepspeed_config,
            "horovod_config": self.horovod_config,
            "oss_config": self.oss_config,
            "sharded_config": self.sddp_config,
            "fully_sharded_config": self.fsdp_config,
        }
        # Generate the runner class from the mixins based on the StokeStatus
        runner_class = type(
            "StokeRunner",
            (dist_class, fp16_class, optimizer_class, io_class),
            {"__init__": __multiple_mixin_init__},
        )(
            verbose=self._verbose,
            batch_size_per_device=self.batch_size,
            grad_accum_steps=self.grad_accum,
            grad_clip=self.grad_clip,
            info_rank=self._info_rank,
            loss=self._loss,
            **kwargs_dict,
        )
        # Make a list of class info for print later
        class_info = [
            f"Distributed Mixin: {dist_class.__name__}",
            f"Optimizer Mixin: {dist_class.__name__}",
            f"FP16 Mixin: {fp16_class.__name__}",
            f"IO Mixin: {io_class.__name__}",
        ]
        return runner_class, class_info

    def _get_io_mixin(self):
        """Determines which IO class to use

        Embedded logic based on the enum class

        Returns
        -------
        ABCMeta
            un-instantiated ioclass

        """
        if self.is_deepspeed:
            return_class = RunnerIOEnum.deepspeed.value
        elif self.is_horovod:
            return_class = RunnerIOEnum.horovod.value
        elif self.is_ddp:
            return_class = RunnerIOEnum.ddp.value
        else:
            return_class = RunnerIOEnum.base.value
        return return_class

    def _get_optimizer_mixin(self):
        """Determines which optimizer class to use

        Embedded logic based on the enum class

        Returns
        -------
        ABCMeta
            un-instantiated optimizer class

        """
        if self.oss:
            return_class = RunnerOptimizerEnum.oss.value
        else:
            return_class = RunnerOptimizerEnum.base.value
        return return_class

    def _get_distributed_mixin(self):
        """Determines which distributed class to use

        Embedded logic based on the enum class

        Returns
        -------
        ABCMeta
            un-instantiated distributed class

        """
        # if not gpu then fall to cpu single
        if not self.gpu:
            return_class = RunnerDistEnum.cpu.value
        # if gpu but no distributed then fall to single gpu
        elif self.gpu and (self.distributed is None):
            return_class = RunnerDistEnum.gpu.value
        elif self.gpu and (self.distributed is not None):
            return_class = RunnerDistEnum[self.distributed].value
        else:
            raise ValueError("Stoke -- Cannot map to a valid distributed class")
        return return_class

    def _get_fp16_mixin(self):
        """Determines which fp16 class to use

        Embedded logic based on the enum class

        Returns
        -------
        ABCMeta
            un-instantiated fp16 class

        """
        if self.fp16 is not None:
            return_class = RunnerFP16Enum[self.fp16].value
        else:
            return_class = RunnerFP16Enum.full.value
        return return_class

    def DataLoader(
        self,
        dataset: Dataset[T_co],
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """Provides a shim interface to torch.utils.data.DataLoader with mapped kwargs.

        Shim is necessary for two reasons... to inject some horovod runtime configs (make sure forkserver is called)
        and to automatically handle device placement since the gpu/fp16 flags can't be determined until the StokeStatus
        object is available which is post init. This could be disconnected from this class but it would require the
        user to forward on device or fp16 configs which breaks the paradigm that the flags only need to be set and
        never handled

        Parameters
        ----------
        dataset: Dataset
            dataset from which to load the data.
        shuffle: bool, default: False
            set to ``True`` to have the data reshuffled at every epoch.
        sampler: Sampler or Iterable, default: None
            defines the strategy to draw samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler: Sampler or Iterable, default: None:
            like :attr:`sampler`, but returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers: int, default: 0
            how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
        collate_fn: callable, optional:
            merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory: bool, default: False:
            If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them. If your
            data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last: bool, default: False
            set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller.
        timeout: numeric, default: 0
            if positive, the timeout value for collecting a batch from workers. Should always be non-negative.
        worker_init_fn: callable, default: None
            If not ``None``, this will be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading.
        generator: torch.Generator: None
            If not ``None``, this RNG will be used by RandomSampler to generate random indexes and multiprocessing
            to generate `base_seed` for workers.
        prefetch_factor: int, default: 2
            Number of samples loaded in advance by each worker. ``2`` means there will be a total of 2 * num_workers
            samples prefetched across all workers.
        persistent_workers: bool, default: False
            If ``True``, the data loader will not shutdown the worker processes after a dataset has been
            consumed once. This allows to maintain the workers `Dataset` instances alive.

        Returns
        -------
        StokeDataLoader
            wrapped torch.utils.data.DataLoader object

        """
        # Check if forkserver is available for horovod and use
        if (
            num_workers > 0
            and "forkserver" in torch.multiprocessing.get_all_start_methods()
            and self.is_horovod
            and self.horovod_config.use_fork_server
        ):
            torch.multiprocessing.set_start_method("forkserver")
            if self._verbose:
                print(
                    f"Stoke -- Attempting to use forkserver as multiprocessing_context"
                )

        if self.distributed is not None and not isinstance(sampler, DistributedSampler):
            raise TypeError(
                f"Stoke -- Using a distributed backend requires passing an instance of a "
                f"DistributedSampler to the sampler argument"
            )
        if self._verbose and self.gpu:
            self.print(
                f"Stoke -- Automatically handling moving model input data to GPU(s)..."
            )
        # Assemble a kwargs dict as the super call with direct named args can cause un-traceable behavior (#23)
        # Insight from PyTorch Lightning that has to handle their DataLoader shims with all kwargs
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/6609b2e46f5eb2cde6c42aedf5b843d050a4bb8d/pytorch_lightning/trainer/data_loading.py#L215
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "generator": generator,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }
        # Forward the already known options from the Stoke status
        return StokeDataLoader(dataset, gpu=self.gpu, fp16=self.fp16, **kwargs)

    def model(self, *args, **kwargs):
        """Wrapped model forward call

        Parameters
        ----------
        *args: list or tuple
            Additional arguments should be passed as keyword arguments
        **kwargs: dict, optional
            Extra arguments passed to the model forward call

        Returns
        -------
        model forward output

        """
        with self._runner.model_context:
            return self._model(*args, **kwargs)
            # return self.model_access(*args, **kwargs)

    def loss(self, *args, **kwargs):
        """Wrapped callable loss function call

        Handles internal logic of aggregating up the losses for single and multiple losses

        Parameters
        ----------
        *args: list or tuple
            Additional arguments should be passed as keyword arguments
        **kwargs: dict, optional
            Extra arguments passed to the loss function call(s)

        Returns
        -------
        outputs of callable loss function(s)

        """
        # TODO: WIP Handle multiple losses. Should support list/tuple of losses. Check non base PyTorch
        with self._runner.loss_context:
            if isinstance(self._loss, (list, tuple)):
                loss = type(self._loss)(val(*args, **kwargs) for val in self._loss)
                sync_loss = [self.detach_and_sync_loss(val) for val in loss]
                self._last_step_loss = type(self._loss)(
                    val for idx, val in enumerate(sync_loss)
                )
                self._agg_loss = type(self._loss)(
                    self._agg_loss[idx] + val for idx, val in enumerate(sync_loss)
                )
                self._handle_ema_loss(loss=sync_loss)
                if self.grad_accum > 1 and self.model_access.training:
                    loss = type(loss)(val / self.grad_accum for val in loss)
            else:
                loss = self._loss(*args, **kwargs)
                sync_loss = self.detach_and_sync_loss(loss)
                self._last_step_loss = sync_loss
                self._agg_loss += sync_loss
                self._handle_ema_loss(loss=sync_loss)
                # Handle grad accumulation by dividing by the accumulation steps
                if self.grad_accum > 1 and self.model_access.training:
                    loss = loss / self.grad_accum
            return loss

    def _handle_ema_loss(self, loss: Union[float, List[float], Tuple[float]]):
        """Handles calculating the ema loss

        Parameters
        ----------
        loss: Union[float, List[float], Tuple[float]]
            current calculated loss list, tuple or float

        Returns
        -------
        None

        """
        self._rolling_loss_steps += 1
        if isinstance(loss, (list, tuple)):
            self._rolling_mean_loss = type(self._rolling_mean_loss)(
                self._ema_loss(value=val, current_mean=self._rolling_mean_loss[idx])
                for idx, val in enumerate(loss)
            )
        else:
            self._rolling_mean_loss = self._ema_loss(
                value=loss, current_mean=self._rolling_mean_loss
            )

    def _ema_loss(self, value: float, current_mean: float):
        """Calculate the ema of the loss

        Parameters
        ----------
        value: float
            current loss value
        current_mean: float
            current mean value

        Returns
        -------
        current ema value: float

        """
        if self._rolling_loss_steps == 1:
            return value
        else:
            return (self._ema_weight * value) + (
                (1.0 - self._ema_weight) * current_mean
            )

    def backward(
        self, loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ):
        """Wrapped backwards call

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            Callable loss function(s)

        Returns
        -------
        None

        """
        # Increment the grad counter
        self._grad_accum_counter += 1
        # Set the context based on the counter
        dist_cm = (
            nullcontext()
            if self._check_accum()
            else self._runner.grad_accum_context(self._model)
        )
        with dist_cm:
            self._runner.backward_call(
                loss=loss, model=self.model_access, optimizer=self._optimizer
            )
        # Increment the number of total calls to backward (each backward to a loss is only considered 1)
        self._backward_steps += 1

    def step(self):
        """Wrapped step call

        Handles grad clipping internally

        Returns
        -------
        None

        """
        # Step the optimizer only if the modulo is zero
        if self._check_accum():
            if self._verbose and self.grad_accum > 0:
                self.print(f"Gradient Accumulation Steps: {self.grad_accum}")
            # Clip if needed
            if self.grad_clip is not None:
                self._runner.clip_grad(
                    self.grad_clip,
                    self._model if self.fully_sharded else self.model_access,
                    self._optimizer,
                    oss=self.oss,
                    horovod=self.is_horovod,
                    deepspeed=self.is_deepspeed,
                    fsdp=self.fully_sharded,
                )
            # Handle the optimizer step
            step_cm = (
                self._runner.step_context(self._optimizer)
                if self.grad_clip is not None
                else nullcontext()
            )
            with step_cm:
                self._runner.step_call(
                    model=self.model_access, optimizer=self._optimizer
                )
            # Reset for the accumulated step
            self._reset()
            # Increment the number of step calls to the optimizer
            self._optimizer_steps += 1
        # if deepspeed we need to step everytime as it handles the grad accumulation internally
        elif self.is_deepspeed:
            # Handle the optimizer step
            step_cm = (
                self._runner.step_context(self._optimizer)
                if self.grad_clip is not None
                else nullcontext()
            )
            with step_cm:
                self._runner.step_call(
                    model=self.model_access, optimizer=self._optimizer
                )

    def _reset(self):
        """Resets the state post optimizer step call

        Returns
        -------
        None

        """
        if self._verbose:
            self.print("Resetting all grad/variables for next optimizer step")
        # Zero the grads if not deepspeed
        if not self.is_deepspeed:
            self.zero_grads()
        # Reset counter
        self._grad_accum_counter = 0
        # Reset agg loss -- single or mutiple losses
        self._agg_loss = self._set_loss_to_zero()

    def save(
        self,
        path: str,
        name: str = uuid4(),
        extension: str = "pt",
        create_directory: bool = True,
        extras: Optional[dict] = None,
    ):
        """Saves a model checkpoint using the correct backend interface

        Parameters
        ----------
        path: str
            path to directory to save the model checkpoint (prefer absolute paths over relative paths)
        name: str, default: uuid4()
            name used to save checkpoint file
        extension: str, default: '.pt'
            extension used to save PyTorch model checkpoint
        create_directory: bool, default: True
            flag to create the directory path if it doesn't exist
        extras: dict, default: None
            a dictionary of any extra things to save

        Returns
        -------
        path: str
            path to directory that the model checkpoint was saved
        tag: str
            full tag name the model checkpoint was saved as

        """
        out_path, tag = self._runner.save(
            model=self._model if self.fully_sharded else self.model_access,
            optimizer=self.optimizer,
            path=path,
            backward_step=self._backward_steps,
            grad_accum_step=self._grad_accum_counter,
            optimizer_step=self._optimizer_steps,
            name=name,
            scaler_dict=self.fp16_state_dict,
            extension=extension,
            create_directory=create_directory,
            extras=extras,
            status=self.status.status,
        )
        self.print(f"Successfully saved model checkpoint to {out_path}/{tag}")
        return out_path, tag

    def load(self, path: str, tag: str, strict: bool = True):
        """Loads a model checkpoint using the correct backend interface

        Parameters
        ----------
        path: str
            path to directory that the model checkpoint was saved (prefer absolute paths over relative paths)
        tag: str
            full tag name the model checkpoint was saved as
        strict: bool
            ignore non-matching keys

        Returns
        -------
        extras: dict, default: None
            a dictionary of any custom fields the user passed to the save function

        """
        # TODO: How to deal with mapping between backends? e.g. FP16 model back to FP32? Or multi-gpu to CPU?
        backward_step, grad_accum_step, optimizer_step, extras = self._runner.load(
            model=self._model if self.fully_sharded else self.model_access,
            optimizer=self.optimizer,
            gpu=self.gpu,
            path=path,
            tag=tag,
            scaler_dict_fn=self._load_fp16_state_dict_fn(),
            strict=strict,
        )
        # Reset values based on what was in the load dict
        self._backward_steps = backward_step
        self._grad_accum_counter = grad_accum_step
        self._optimizer_steps = optimizer_step
        self.print(f"Successfully loaded model checkpoint from {path}/{tag}")
        # Return the extras dict
        return extras

    def print_num_model_parameters(
        self, normalize: ParamNormalize = ParamNormalize.MILLION
    ):
        """

        Parameters
        ----------
        normalize: ParamNormalize, default: ParamNormalize.MILLION
            ParamNormalize choice for pretty print normalizing

        Returns
        -------
        None

        """
        self.print(
            f"Total Trainable Model Parameters: "
            f"{(self.num_model_parameters / normalize.value):.3f} {normalize.name}"
        )

    def detach_and_sync_loss(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device=None,
    ):
        """Shorthand method to detach and sync loss

        Maps to the runner function of the same name

        Parameters
        ----------
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            current loss(es)
        device: default: None
            device to sync across

        Returns
        -------
        loss that is synced across devices and all_reduced w/ SUM

        """
        return self._runner.detach_and_sync_loss(loss=loss, device=device)

    def zero_grads(self):
        """Zeros the optimizer grads depending on the optimizer type

        Returns
        -------
        None

        """
        zero_optimizer_grads(
            optimizer=self._optimizer, apex=self.is_apex, horovod=self.is_horovod
        )

    def reset(self):
        """Public method for resetting the underlying stoke state

        Returns
        -------
        None

        """
        self._reset()

    def reset_tracking(self):
        """Public method for resetting all underlying stoke tracked variables

        Returns
        -------
        None

        """
        # Create some tracking vars
        self._grad_accum_counter = 0
        self._optimizer_steps = 0
        self._backward_steps = 0
        self._last_step_loss = self._set_loss_to_zero()
        self._agg_loss = self._set_loss_to_zero()
        self._rolling_mean_loss = self._set_loss_to_zero()
        self._rolling_loss_steps = 0

    def dump_model_parameter_info(self):
        """Dumps all parameter information for named parameters (shape, device, dtype)

        Returns
        -------
        None

        """
        self.print("Dumping all model parameter information to stdout....")
        for name, param in self.model_access.named_parameters():
            if param.requires_grad:
                self.print(
                    f"Name: {name}, Shape: {param.shape}, "
                    f"Device: {param.device}, dtype: {param.dtype}"
                )

    def _load_fp16_state_dict_fn(self):
        """Returns the function to load the sacler state dict

        Returns
        -------
        mp_state_dict_fn: Callable, default: None
            callable function to load the scaler state dict

        """
        mp_state_dict_fn = None
        if self.scaler is not None:
            if self.is_apex:
                try:
                    from apex import amp

                    mp_state_dict_fn = amp.load_state_dict
                except ImportError as e:
                    print(
                        e,
                        ": Stoke -- apex cannot be imported -- please install (https://github.com/NVIDIA/apex)",
                    )
            else:
                mp_state_dict_fn = self.scaler.load_state_dict
        return mp_state_dict_fn

    def barrier(self):
        """Calls the underlying distributed barrier if available"""
        self._runner.barrier()

    @property
    def step_loss(self):
        """Gets the last step loss synced across device(s) (unscaled)"""
        return self._last_step_loss

    @property
    def model_access(self):
        """Interface for model access due to the different types between the DP, DDP, and SDDP implementations"""
        if isinstance(self._model, (DDP, DP, SDDP, FSDP)):
            return self._model.module
        else:
            return self._model

    @property
    def loss_access(self):
        """Gets loss tensor(s)"""
        return self._loss

    @property
    def optimizer(self):
        """Gets the optimizer"""
        return self._optimizer

    @property
    def scaler(self):
        """Gets the current scaler object"""
        return self._runner.scaler

    @property
    def fp16_state_dict(self):
        """Gets the fp16 state dict from various methods"""
        mp_state_dict = None
        if self.scaler is not None:
            if self.is_apex:
                try:
                    from apex import amp

                    mp_state_dict = amp.state_dict()
                except ImportError as e:
                    print(
                        e,
                        ": Stoke -- apex cannot be imported -- please install (https://github.com/NVIDIA/apex)",
                    )
            elif self.is_amp:
                mp_state_dict = self.scaler.state_dict()
        return mp_state_dict

    @property
    def status(self):
        """Gets the StokeStatus object"""
        return self._status

    @property
    def batch_size(self):
        """Shortcut to batch size"""
        return self._status.batch_size

    @property
    def effective_batch_size(self):
        """Shortcut to effective batch size"""
        return self._status.effective_batch_size

    @property
    def grad_clip(self):
        """Shortcut to get grad clip"""
        return self._status.grad_clip

    @property
    def grad_accum(self):
        """Shortcut to get grad accumulation"""
        return self._status.grad_accum

    @property
    def gpu(self):
        """Shortcut to get GPU status"""
        return self._status.gpu

    @property
    def cuda(self):
        """Shortcut to get cuda status"""
        return self._status.cuda

    @property
    def nccl(self):
        """Shortcut to get nccl status"""
        return self._status.nccl

    @property
    def fp16(self):
        """Shortcut to get FP16 status"""
        return self._status.fp16

    @property
    def is_apex(self):
        """Returns if APEX is activated"""
        return self._status.is_fp16_apex

    @property
    def is_amp(self):
        """Returns if AMP is activated"""
        return self._status.is_fp16_amp

    @property
    def distributed(self):
        """Shortcut to distributed status"""
        return self._status.distributed

    @property
    def is_ddp(self):
        """Returns if DDP is activated"""
        return self._status.is_distributed_ddp

    @property
    def is_horovod(self):
        """Returns if Horovod is activated"""
        return self._status.is_distributed_horovod

    @property
    def is_deepspeed(self):
        """Returns if Deepspeed is acticated"""
        return self._status.is_distributed_deepspeed

    @property
    def oss(self):
        """Returns if Fairscale optimizer state sharding status"""
        return self._status.oss

    @property
    def sharded(self):
        """Returns if Fairscale sharded DDP status"""
        return self._status.sharded

    @property
    def fully_sharded(self):
        """Returns if Fairscale fully sharded DDP status"""
        return self._status.fully_sharded

    @property
    def world_size(self):
        """Shortcut to get world size"""
        return self._runner.world_size

    @property
    def rank(self):
        """Shortcut to get rank"""
        return self._runner.rank

    @property
    def amp_config(self):
        """Returns amp config or None based on amp state"""
        return self._status.amp_config if self.is_amp else None

    @property
    def apex_config(self):
        """Returns apex config or None based on apex state"""
        return self._status.apex_config if self.is_apex else None

    @property
    def ddp_config(self):
        """Returns ddp config or None based on ddp state"""
        return self._status.ddp_config if self.is_ddp else None

    @property
    def deepspeed_config(self):
        """Returns deepspeed config or None based on deepspeed state"""
        return self._status.deepspeed_config if self.is_deepspeed else None

    @property
    def oss_config(self):
        """Returns oss config or None based on ossstate"""
        return self._status.oss_config if self.oss else None

    @property
    def sddp_config(self):
        """Returns sddp config or None based on sddp state"""
        return self._status.sddp_config if self.sharded else None

    @property
    def fsdp_config(self):
        """Returns fsdp config or None based on fsdp state"""
        return self._status.fsdp_config if self.fully_sharded else None

    @property
    def horovod_config(self):
        """Returns horovod config or None based on horovod state"""
        return self._status.horovod_config if self.is_horovod else None

    @property
    def num_model_parameters(self):
        """Returns number of parameters that require gradients"""
        return sum(p.numel() for p in self.model_access.parameters() if p.requires_grad)

    @property
    def ema_loss(self):
        """Returns the current rolling mean loss"""
        return self._rolling_mean_loss
