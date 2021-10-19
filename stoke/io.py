# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles i/o related functions -- mixin style"""

from abc import ABC
from enum import Enum
from typing import Callable, Dict, Optional, Union

import horovod.torch as hvd
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel
from fairscale.optim.oss import OSS

from stoke.utils import make_folder


class BaseStokeIO(ABC):
    """Base class for handling IO for different backends

    Attributes
    ----------
    _save_rank: int, default: 0
        device to restrict calls to if necessary (e.g. horovod, ddp)
    _prefix: str
        prefix to append to all checkpoints
    _verbose: bool, default: True
        Flag for verbosity

    """

    def __init__(self, save_rank: int = 0, verbose: bool = True, **kwargs):
        """Init for BaseStokeIO class

        Parameters
        ----------
        save_rank: int, default: 0
            device to restrict calls to if necessary (e.g. horovod, ddp)
        verbose: bool, default: True
            Flag for verbosity

        """
        self._save_rank = save_rank
        self._prefix = "stoke"
        self._verbose = verbose

    def _make_tag(self, name: str, backward_step: int):
        """Constructs the save tag

        Parameters
        ----------
        name: str
            name used to save checkpoint file
        backward_step: int
            current number of backward calls (for saving unique name/tag)

        Returns
        -------
        str

        """
        return f"{self._prefix}-{name}-backward-step-{backward_step}"

    def _make_full_save_path(
        self, path: str, name: str, backward_step: int, extension: str
    ):
        """Constructs the full string path from each piece and appends a stoke prefix

        Parameters
        ----------
        path: str
            path to directory to save the model checkpoint (prefer absolute paths over relative paths)
        name: str
            name used to save checkpoint file
        backward_step: int
            current number of backward calls (for saving unique name/tag)
        extension: str
            extension used to save PyTorch model checkpoint

        Returns
        -------
        str

        """
        return f"{path}/{self._make_tag(name=name, backward_step=backward_step)}.{extension}"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        path: str,
        backward_step: int,
        grad_accum_step: int,
        optimizer_step: int,
        name: str,
        status: dict,
        scaler_dict: Optional[dict] = None,
        extension: str = "pt",
        create_directory: bool = True,
        extras: Optional[dict] = None,
    ):
        """Implementation(s) for saving a PyTorch model checkpoint

        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        path: str
            path to directory to save the model checkpoint (prefer absolute paths over relative paths)
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation (for resuming training correctly)
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        name: str
            name used to save checkpoint file
        status: dict
            current stoke status dictionary
        scaler_dict: dict, default: None
            state_dict from native PyTorch AMP, Fairscale, or APEX
        extension: str, default: '.pt'
            extension used to save PyTorch model checkpoint
        create_directory: bool, default: True
            flag to create the directory path if it doesn't exist
        extras: dict, default: None
            a dictionary of any extra things to save

        Returns
        -------
        out_path: str
            path to directory that the model checkpoint was saved
        tag: str
            full tag name the model checkpoint was saved as

        """
        # Call private as no logic is needed for the base save call
        out_path, tag = self._save(
            model_dict=model.state_dict(),
            optimizer_dict=optimizer.state_dict(),
            path=path,
            backward_step=backward_step,
            optimizer_step=optimizer_step,
            name=name,
            scaler_dict=scaler_dict,
            extension=extension,
            create_directory=create_directory,
            extras=extras,
            grad_accum_step=grad_accum_step,
            status=status,
        )
        return out_path, tag

    def _save(
        self,
        model_dict: Dict,
        optimizer_dict: Dict,
        path: str,
        backward_step: int,
        grad_accum_step: int,
        optimizer_step: int,
        name: str,
        status: Dict,
        scaler_dict: Optional[Dict],
        extension: str,
        create_directory: bool,
        extras: Optional[Dict],
    ):
        """Private base implementation for saving a PyTorch model checkpoint

        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Parameters
        ----------
        model: Dict
            current model object dictionary
        optimizer: Dict
            current optimizer object dictionary
        scaler_dict: Optional[Dict]
            state_dict from native PyTorch AMP, Fairscale, or APEX
        path: str
            path to directory to save the model checkpoint (prefer absolute paths over relative paths)
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation (for resuming training correctly)
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        name: str
            name used to save checkpoint file
        status: Dict
            current stoke status dictionary
        extension: str
            extension used to save PyTorch model checkpoint
        create_directory: bool
            flag to create the directory path if it doesn't exist
        extras: Dict
            a dictionary of any extra things to save

        Returns
        -------
        path: str
            path to directory that the model checkpoint was saved
        tag: str
            full tag name the model checkpoint was saved as

        """
        # Construct the path
        save_path = self._make_full_save_path(
            path=path, name=name, backward_step=backward_step, extension=extension
        )
        if self._verbose:
            self._print_device(f"Attempting to save model checkpoint to {save_path}")
        # Save the model with the constructed path
        try:
            if create_directory:
                make_folder(path)
            torch.save(
                {
                    "backward_step": backward_step,
                    "grad_accum_step": grad_accum_step,
                    "optimizer_step": optimizer_step,
                    "stoke_status": status,
                    "model_state_dict": model_dict,
                    "optimizer_state_dict": optimizer_dict,
                    "scaler_state_dict": scaler_dict,
                    "extras": extras,
                },
                save_path,
            )
        except OSError as e:
            self._print_device(f"Unable to save model to given path: {save_path}")
            raise e
        return (
            path,
            f"{self._make_tag(name=name, backward_step=backward_step)}.{extension}",
        )

    def _load(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        map_loc: str,
        path: str,
        tag: str,
        scaler_dict_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        """Private base implementation for loading a PyTorch model checkpoint

        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        map_loc: str
            device map
        gpu: bool
            if using gpu device or not
        path: str
            path to directory that the model checkpoint was saved (prefer absolute paths over relative paths)
        tag: str
            full tag name the model checkpoint was saved as
        scaler_dict_fn: Callable, default: None
            callable function to load the scaler state dict
        strict: bool
            ignore non-matching keys

        Returns
        -------
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation (for resuming training correctly)
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        extras: dict
            a dictionary of any extra things that were saved

        """
        # Load the dictionary
        try:
            load_dict = torch.load(f"{path}/{tag}", map_location=map_loc)
            # Load the model state dict
            model.load_state_dict(
                state_dict=load_dict["model_state_dict"], strict=strict
            )
            # Handle the fully sharded data parallel case where the shard needs to be pulled from the full state dict
            if isinstance(model, FullyShardedDataParallel):
                self._print_device(
                    "Handling loading of correct optimizer sharded state for Fairscale FSDP"
                )
                optimizer.load_state_dict(
                    state_dict=model.get_shard_from_optim_state_dict(
                        load_dict["optimizer_state_dict"]
                    )
                )
            # Fallback to the default load form the fully state dict
            else:
                # Load the optimizer state dict
                optimizer.load_state_dict(state_dict=load_dict["optimizer_state_dict"])
            # Load the scaler state if needed
            if scaler_dict_fn is not None:
                scaler_dict_fn(load_dict["scaler_state_dict"])
        except OSError as e:
            self._print_device(f"Unable to load model from given path: {path}/{tag}")
            raise e
        return (
            load_dict["backward_step"],
            load_dict["grad_accum_step"],
            load_dict["optimizer_step"],
            load_dict["extras"],
        )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        gpu: bool,
        path: str,
        tag: str,
        scaler_dict_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        """Implementation for loading a PyTorch model checkpoint

        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        gpu: bool
            if using gpu device or not
        path: str
            path to directory that the model checkpoint was saved (prefer absolute paths over relative paths)
        tag: str
            full tag name the model checkpoint was saved as
        scaler_dict_fn: Callable, default: None
            callable function to load the scaler state dict
        strict: bool
            ignore non-matching keys

        Returns
        -------
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation (for resuming training correctly)
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        extras: dict
            a dictionary of any extra things that were saved

        """
        # Load the dictionary
        # map to cuda:device_id or cpu no matter what (covers CPU->GPU and GPU->GPU)
        # this should be functional for cuda:0 since this will catch the single GPU case only
        map_loc = f"cuda:{self.device_id}" if gpu else self.device_id
        self._print_device(f"Load is mapping to {map_loc}")
        # Call the private load interface
        backward_step, grad_accum_step, optimizer_step, extras = self._load(
            model=model,
            optimizer=optimizer,
            map_loc=map_loc,
            path=path,
            tag=tag,
            scaler_dict_fn=scaler_dict_fn,
            strict=strict,
        )
        return backward_step, grad_accum_step, optimizer_step, extras


class DeepspeedIO(BaseStokeIO):
    def __init__(self, save_rank: int = 0, **kwargs):
        super(DeepspeedIO, self).__init__(save_rank=save_rank, **kwargs)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        path: str,
        backward_step: int,
        grad_accum_step: int,
        optimizer_step: int,
        name: str,
        status: dict,
        scaler_dict: Optional[dict] = None,
        extension: str = "pt",
        create_directory: bool = True,
        extras: Optional[dict] = None,
    ):
        """Deepspeed override implementation for saving a PyTorch model checkpoint

        Deepspeed maintains it's own wrapper for saving so it needs to be called here. It looks like it will save
        multiple pieces depending on sharding but I'm not sure

        https://www.deepspeed.ai/getting-started/#model-checkpointing
        https://github.com/microsoft/DeepSpeed/blob/ed3de0c21b1fea330de9c1a78a23ca33f340ef20/deepspeed/runtime/engine.py#L1822

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        path: str
            path to directory to save the model checkpoint (prefer absolute paths over relative paths)
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        name: str
            name used to save checkpoint file
        status: dict
            current stoke status dictionary
        scaler_dict: Callable
            state_dict from native PyTorch AMP, Fairscale, or APEX
        extension: str, default: '.pt'
            extension used to save PyTorch model checkpoint (Note: Deepspeed will ignore this due to it's internal
            implementation)
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

        Notes
        -----
        From deepspeed save_checkpoint doc_string:
        all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.

        """
        # Construct the tag for deepspeed
        tag = self._make_tag(name=name, backward_step=backward_step)
        # Construct the path
        save_path = self._make_full_save_path(
            path=path, name=name, backward_step=backward_step, extension=extension
        )
        if self._verbose:
            self._print_device(f"Attempting to save model checkpoint to {save_path}")
        # Use a barrier to make sure the save is done only when all devices are finished with prior calls
        torch.distributed.barrier()
        # Save the model with the constructed path
        try:
            client_sd = {
                "backward_step": backward_step,
                "grad_accum_step": grad_accum_step,
                "optimizer_step": optimizer_step,
                "stoke_status": status,
                "extras": extras,
            }
            _ = model.save_checkpoint(
                path, tag, client_state=client_sd, save_latest=False
            )
        except OSError as e:
            self._print_device(f"Unable to save model to given path: {path}")
            raise e
        # Use a barrier to make sure no one exits until the save is complete
        torch.distributed.barrier()
        return path, tag

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        gpu: bool,
        path: str,
        tag: str,
        scaler_dict_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        """Deepspeed override implementation for loading a PyTorch model checkpoint

        https://www.deepspeed.ai/getting-started/#model-checkpointing

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        gpu: bool
            if using gpu device or not
        path: str
            path to directory that the model checkpoint was saved (prefer absolute paths over relative paths)
        tag: str
            full tag name the model checkpoint was saved as
        scaler_dict_fn: Callable, default: None
            callable function to load the scaler state dict
        strict: bool
            ignore non-matching keys

        Returns
        -------
        backward_step: int
            current number of backward calls (for resuming training correctly)
        grad_accum_step: int,
            current step of gradient accumulation (for resuming training correctly)
        optimizer_step: int
            current number of optimizer calls (for resuming training correctly)
        extras: dict
            a dictionary of any extra things that were saved

        """
        # Load the dictionary
        # map to cuda:device_id (as this will prevent the save on device 0 from clashing with the current device id)
        map_loc = f"cuda:{self.device_id}"
        self._print_device(f"Load is mapping to {map_loc}")
        try:
            _, client_sd = model.load_checkpoint(
                path, tag, load_module_strict=strict, load_optimizer_states=True
            )
        except OSError as e:
            self._print_device(f"Unable to load model from given path: {path}/{tag}")
            raise e
        return (
            client_sd["backward_step"],
            client_sd["grad_accum_step"],
            client_sd["optimizer_step"],
            client_sd["extras"],
        )


class DDPIO(BaseStokeIO):
    def __init__(self, save_rank: int = 0, **kwargs):
        super(DDPIO, self).__init__(save_rank=save_rank, **kwargs)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        path: str,
        backward_step: int,
        grad_accum_step: int,
        optimizer_step: int,
        name: str,
        status: dict,
        scaler_dict: Optional[dict] = None,
        extension: str = "pt",
        create_directory: bool = True,
        extras: Optional[dict] = None,
    ):
        # Use a barrier to make sure the save is done only when all devices are finished with prior calls
        torch.distributed.barrier()
        # FSDP needs different syntax for saving
        if isinstance(model, FullyShardedDataParallel):
            self._print_device(
                "Handling consolidation of optimizer sharded states for Fairscale FSDP"
            )
            # Need to be called on all ranks
            model_state = model.state_dict()
            optimizer_state = model.gather_full_optim_state_dict(optimizer)
            # Use a logical barrier to only save on the 0 idx device
            if self.rank == self._save_rank:
                # Dispatch to private save method if logic is met
                path, tag = self._save(
                    model_dict=model_state,
                    optimizer_dict=optimizer_state,
                    path=path,
                    backward_step=backward_step,
                    optimizer_step=optimizer_step,
                    name=name,
                    scaler_dict=scaler_dict,
                    extension=extension,
                    create_directory=create_directory,
                    extras=extras,
                    grad_accum_step=grad_accum_step,
                    status=status,
                )
        else:
            # If OSS then make sure it's consolidated before saving as norm PyTorch checkpoint
            # This needs to be called on all ranks but can be given a recipient_rank
            if isinstance(optimizer, OSS):
                self._print_device(
                    f"Consolidating optimizer sharded states onto device {self._save_rank}"
                )
                optimizer.consolidate_state_dict(recipient_rank=self._save_rank)
            # Use a logical barrier to only save on the 0 idx device
            if self.rank == self._save_rank:
                # Dispatch to private save method if logic is met
                path, tag = self._save(
                    model_dict=model.state_dict(),
                    optimizer_dict=optimizer.state_dict(),
                    path=path,
                    backward_step=backward_step,
                    optimizer_step=optimizer_step,
                    name=name,
                    scaler_dict=scaler_dict,
                    extension=extension,
                    create_directory=create_directory,
                    extras=extras,
                    grad_accum_step=grad_accum_step,
                    status=status,
                )
        # Use a barrier to make sure no one exits until the save is complete
        torch.distributed.barrier()
        return (
            path,
            f"{self._make_tag(name=name, backward_step=backward_step)}.{extension}",
        )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        gpu: bool,
        path: str,
        tag: str,
        scaler_dict_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        # Use a barrier to make sure the load is done only when all devices are finished with prior calls
        torch.distributed.barrier()
        # Load the dictionary
        # map to cuda:device_id (as this will prevent the save on device 0 from clashing with the current device id)
        map_loc = f"cuda:{self.device_id}"
        self._print_device(f"Load is mapping to {map_loc}")
        # Call the private load interface
        backward_step, grad_accum_step, optimizer_step, extras = self._load(
            model=model,
            optimizer=optimizer,
            map_loc=map_loc,
            path=path,
            tag=tag,
            scaler_dict_fn=scaler_dict_fn,
            strict=strict,
        )
        # Use a barrier to make sure no one exits until the load is complete across all devices
        torch.distributed.barrier()
        return backward_step, grad_accum_step, optimizer_step, extras


class HorovodIO(BaseStokeIO):
    def __init__(self, save_rank: int = 0, **kwargs):
        super(HorovodIO, self).__init__(save_rank=save_rank, **kwargs)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        path: str,
        backward_step: int,
        grad_accum_step: int,
        optimizer_step: int,
        name: str,
        status: dict,
        scaler_dict: Optional[dict] = None,
        extension: str = "pt",
        create_directory: bool = True,
        extras: Optional[dict] = None,
    ):
        # Use a barrier to make sure the save is done only when all devices are finished with prior calls
        # Horovod doesn't have a native barrier so lean on join to take care of it
        # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
        hvd.join()
        # Use a logical barrier to only save on the 0 idx device
        if self.rank == self._save_rank:
            # Dispatch to private save method if logic is met
            path, tag = self._save(
                model_dict=model.state_dict(),
                optimizer_dict=optimizer.state_dict(),
                path=path,
                backward_step=backward_step,
                optimizer_step=optimizer_step,
                name=name,
                scaler_dict=scaler_dict,
                extension=extension,
                create_directory=create_directory,
                extras=extras,
                grad_accum_step=grad_accum_step,
                status=status,
            )
        # Use a barrier to make sure no one exits until the save is complete
        # Horovod doesn't have a native barrier so lean on join to take care of it
        # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
        hvd.join()
        return (
            path,
            f"{self._make_tag(name=name, backward_step=backward_step)}.{extension}",
        )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        gpu: bool,
        path: str,
        tag: str,
        scaler_dict_fn: Optional[Callable] = None,
        strict: bool = True,
    ):
        # Use a barrier to make sure the load is done only when all devices are finished with prior calls
        # Horovod doesn't have a native barrier so lean on join to take care of it
        # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
        hvd.join()
        # map to cuda:device_id -- horovod will only load on cuda:0 and then broadcast instead of loading on multiple
        # devices? TODO: Check if this is necessary or could we just load like DDP and skip the broadcast?
        # Terrible Horovod docs strike again -- load on dev 0 and sync -- but this doesn't deal with amp/apex
        # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
        # I think we can just ignore this and load on all devices
        map_loc = f"cuda:{self.device_id}"
        self._print_device(f"Load is mapping to {map_loc}")
        backward_step, grad_accum_step, optimizer_step, extras = self._load(
            model=model,
            optimizer=optimizer,
            map_loc=map_loc,
            path=path,
            tag=tag,
            scaler_dict_fn=scaler_dict_fn,
            strict=strict,
        )
        # Use a barrier to make sure no one exits until the load is complete across all devices
        # Horovod doesn't have a native barrier so lean on join to take care of it
        # https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.join
        hvd.join()
        return backward_step, grad_accum_step, optimizer_step, extras


class RunnerIOEnum(Enum):
    base = BaseStokeIO
    deepspeed = DeepspeedIO
    ddp = DDPIO
    horovod = HorovodIO
