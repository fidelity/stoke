# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles FP16/mixed-precision related classes -- mixin style"""

from abc import ABC
from contextlib import nullcontext
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.optim.oss import OSS

from stoke.configs import ClipGradConfig, ClipGradNormConfig


class BaseFP16(ABC):
    """Base class for mixed precision and FP16 functionality

    This class handles base and common functionality for all of the different mixed-precision backends. Contains
    functionality related to gradient clipping, backward call, step call, and context wrappers for the model forward
    and loss calls

    Attributes
    ----------
    loss_context
    model_context
    scaler
    _scaler: default: None
        scaler object for backends that require one
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, scaler=None, verbose: bool = True):
        """Init for BaseFP16 class

        Parameters
        ----------
        scaler: default: None
            scaler object for backends that require one
        verbose: bool, default: True
            flag for verbosity

        """
        self._scaler = scaler
        self._verbose = verbose

    def _scaler_info(self):
        if self._verbose and self._scaler is not None:
            self._print_device(
                f"FP16 Mixin: Initialized scaler of type {type(self._scaler).__name__}"
            )

    def wrap_fp16(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps model and optimizer with specific mixed-precision related backend wrappers

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        model: torch.nn.Module
            modified version of model object for mixed-precision backends
        optimizer: Union[torch.optim.Optimizer, OSS]]
            modified version of optimizer object for mixed-precision backends

        """
        self._scaler_info()
        return model, optimizer

    def clip_grad(
        self,
        grad_clip: Union[ClipGradConfig, ClipGradNormConfig],
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        oss: bool,
        horovod: bool,
        deepspeed: bool,
        fsdp: bool,
    ):
        """Base handle clipping the current gradients

        Determines which method to use based on the gradient clipping config and the current runtime state

        Parameters
        ----------
        grad_clip: Union[ClipGradConfig, ClipGradNormConfig]
            gradient clipping config that will determine which method to use
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        oss: bool
            optimizer state sharding flag
        horovod: bool
            horovod flag
        deepspeed: bool
            deepspeed flag
        fsdp: bool
            fully sharded data parallel flag for Fairscale

        Returns
        -------
        None

        """
        if deepspeed:
            if self._verbose:
                self._print_device(
                    "Letting deepspeed internally handle clipping calculated/accumulated "
                    "gradients..."
                )
        else:
            if self._verbose:
                self._print_device(
                    f'{type(grad_clip).__name__.replace("Config", "")} '
                    f"is automatically clipping calculated/accumulated gradients..."
                )
            if horovod:
                # Hidden here -- Horovod docs are terrible
                # https://horovod.readthedocs.io/en/latest/api.html#horovod.torch.DistributedOptimizer
                if self._verbose:
                    self._print_device(
                        f"Calling Horovod optimizer.synchronize() pre grad-clip"
                    )
                optimizer.synchronize()
            if isinstance(grad_clip, ClipGradConfig):
                self.clip_grad_value(
                    model=model, optimizer=optimizer, clip_value=grad_clip.clip_value
                )
            elif isinstance(grad_clip, ClipGradNormConfig):
                self.clip_grad_norm(
                    model=model,
                    optimizer=optimizer,
                    max_norm=grad_clip.max_norm,
                    norm_type=grad_clip.norm_type,
                    oss=oss,
                    fsdp=fsdp,
                )
            else:
                raise ValueError(
                    f"Stoke -- clip_grad received an incorrect instance type of {type(grad_clip)}"
                )

    def clip_grad_value(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        clip_value: float,
    ):
        """Base handle clip gradients by value

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        clip_value: float
            absolute value to clip grads

        Returns
        -------
        None

        """
        if self.scaler is not None:
            if self._verbose:
                self._print_device(f"Automatically unscaling gradients...")
            self._scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        oss: bool = False,
        fsdp: bool = False,
    ):
        """Base handle clip gradients by the norm

        Depending on some extension flags switch between the correct clip_grad_norm calls

        OSS: https://fairscale.readthedocs.io/en/latest/api/optim/oss.html
        FSDP: https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        max_norm: Union[float, int]
            max norm of the gradients
        norm_type: Union[float, int]
            type of the used p-norm
        oss: bool, default: False
            optimizer state sharding flag
        fsdp: bool, default: False
            fully sharded data parallel flag for Fairscale

        Returns
        -------
        None

        """
        if self.scaler is not None:
            if self._verbose:
                self._print_device(f"Automatically unscaling gradients...")
            self._scaler.unscale_(optimizer)
        # need to fallback to the OSS Fairscale implementation for norm as the shards need to sync for the norm
        if oss:
            optimizer.clip_grad_norm(max_norm=max_norm, norm_type=norm_type)
        # need to fallback to the Fairscale FSDP implementation for norm as the shards need to sync for the norm
        elif fsdp:
            model.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm, norm_type=norm_type
            )

    @property
    def scaler(self):
        """Returns grad scaler"""
        return self._scaler

    @property
    def loss_context(self):
        """Returns the base context wrapper for the loss call"""
        return nullcontext()

    @property
    def model_context(self):
        """Returns the base context wrapper for the model call"""
        return nullcontext()

    def backward_call(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
    ):
        """Base wrapped backward call

        Parameters
        ----------
        loss: loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            loss tensor(s)
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        if isinstance(loss, (list, tuple)):
            for idx, val in enumerate(loss):
                val.backward(retain_graph=(idx == 0))
        else:
            loss.backward()

    def step_call(
        self, model: torch.nn.Module, optimizer: Union[torch.optim.Optimizer, OSS]
    ):
        """Base wrapped step of the optimizer

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        # Step the optimizer
        optimizer.step()


class NullFP16(BaseFP16):
    def __init__(self, verbose: bool = True, **kwargs):
        """Init for NullFP16 class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        Notes
        -----
        Scaler set to None as it is not needed

        """
        super(NullFP16, self).__init__(scaler=None, verbose=verbose)


class DeepspeedFP16(NullFP16):
    def __init__(self, verbose: bool = True, **kwargs):
        super(DeepspeedFP16, self).__init__(verbose=verbose)

    def backward_call(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
    ):
        """Override of deepspeed wrapped backward call

        Deepspeed calls backward via the model engine instead of the loss

        Parameters
        ----------
        loss: loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            loss tensor(s)
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        if isinstance(loss, (list, tuple)):
            for idx, val in enumerate(loss):
                model.backward(val, retain_graph=(idx == 0))
        else:
            model.backward(loss)

    def step_call(
        self, model: torch.nn.Module, optimizer: Union[torch.optim.Optimizer, OSS]
    ):
        """Override of deepspeed wrapped backward call

        Deepspeed calls step via the model engine instead of the optimizer

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        model.step()


class ApexBaseFP16(BaseFP16):
    """Base class for Apex FP16 methods

    This class handles base and common functionality for O1 and O2 Apex mixed-precision backends. Contains
    functionality related to gradient clipping, backward call, step call, and context wrappers for the model forward
    and loss calls

    Attributes
    ----------
    loss_context
    model_context
    scaler
    _apex_config: ApexConfig
        Configuration object for Apex
    _multi_loss: int, default: 1
        Holds the number of losses to use (apex can use multiple scalers per loss)
    _scaler: default: None
        scaler object for backends that require one
    _verbose bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, verbose: bool = True, **kwargs):
        """Init for ApexBaseFP16 class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here loss or apex_config might be passed in

        Notes
        -----
        Scaler set to None as it is not needed

        """
        super(ApexBaseFP16, self).__init__(scaler=None, verbose=verbose)
        self._conditional_import()
        self._apex_config = kwargs["apex_config"]
        self._multi_loss = (
            len(kwargs["loss"]) if isinstance(kwargs["loss"], (list, tuple)) else 1
        )

    @staticmethod
    def _conditional_import():
        """Attempts to conditionally import apex if the functionality is required

        Raises
        ------
        ImportError
            If apex cannot be imported

        Returns
        -------
        None

        """
        try:
            global amp
            from apex import amp
        except ImportError as e:
            print(
                e,
                ": apex cannot be imported -- please install (https://github.com/NVIDIA/apex)",
            )

    def _apex_convert_to_sync_batch_norm(self, model: torch.nn.Module):
        """Replaces all BatchNorm*D layers with apex.parallel.SyncBatchNorm layers

        Parameters
        ----------
        model: torch.nn.Module
            current model object

        Returns
        -------
        model: torch.nn.Module
            modified version of model with all BatchNorm*D layers replaced with apex.parallel.SyncBatchNorm layers

        """
        self.print_device(
            f"Converting all BatchNorm*D layers to apex.parallel.SyncBatchNorm layers..."
        )
        try:
            from apex.parallel import convert_syncbn_model

            model = convert_syncbn_model(module=model)
        except ImportError as e:
            print(
                e,
                ": apex cannot be imported -- please install (https://github.com/NVIDIA/apex)",
            )
        return model

    def clip_grad_value(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        clip_value: float,
    ):
        """Override handle clip gradients by value for APEX

        Need to call master_params within APEX to clip correctly

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        clip_value: float
            absolute value to clip grads

        Returns
        -------
        None

        """
        if self._verbose:
            self._print_device(
                f"Automatically clipping calculated/accumulated gradients..."
            )
        torch.nn.utils.clip_grad_value_(
            amp.master_params(optimizer), clip_value=clip_value
        )

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        oss: bool = False,
        fsdp: bool = False,
    ):
        """Override handle clip gradients by the norm for APEX

        Need to call master_params within APEX to clip correctly

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object
        max_norm: Union[float, int]
            max norm of the gradients
        norm_type: Union[float, int]
            type of the used p-norm
        oss: bool, default: False
            optimizer state sharding flag
        fsdp: bool, default: False
            fully sharded data parallel flag for Fairscale

        Returns
        -------
        None

        """
        if self._verbose:
            self._print_device(
                f"Automatically clipping calculated/accumulated gradients..."
            )
        torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer), max_norm=max_norm, norm_type=norm_type
        )

    def backward_call(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
    ):
        """Override wrapped backward call for APEX

        Need to use APEX scale_loss context with backward call

        Parameters
        ----------
        loss: loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            loss tensor(s)
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        if isinstance(loss, (list, tuple)):
            for idx, val in enumerate(loss):
                with amp.scale_loss(
                    val,
                    optimizer,
                    loss_id=idx if self._apex_config.scaler_per_loss else 0,
                ) as scaled_loss:
                    scaled_loss.backward(retain_graph=(idx == 0))
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()


class ApexO2AmpFP16(ApexBaseFP16):
    def __init__(self, verbose: bool = True, **kwargs):
        """Init for ApexO2AmpFP16 class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here loss or apex_config might be passed in

        Notes
        -----
        Scaler set to None as it is not needed

        """
        super(ApexO2AmpFP16, self).__init__(verbose=verbose, **kwargs)

    def wrap_fp16(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps model and optimizer with Apex O2 mixed-precision related backend wrappers

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        model: torch.nn.Module
            modified version of model object for mixed-precision backends
        optimizer: Union[torch.optim.Optimizer, OSS]]
            modified version of optimizer object for mixed-precision backends

        """
        self._scaler_info()
        if self._apex_config.convert_to_sync_batch_norm:
            model = self._apex_convert_to_sync_batch_norm(model=model)
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level="O2",
            cast_model_outputs=self._apex_config.cast_model_outputs,
            max_loss_scale=self._apex_config.max_loss_scale,
            min_loss_scale=self._apex_config.min_loss_scale,
            verbosity=self._apex_config.verbosity,
            num_losses=self._multi_loss if self._apex_config.scaler_per_loss else 1,
        )
        return model, optimizer


class ApexO1AmpFP16(ApexBaseFP16):
    def __init__(self, verbose: bool = True, **kwargs):
        """Init for ApexO1AmpFP16 class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here loss or apex_config might be passed in

        Notes
        -----
        Scaler set to None as it is not needed

        """
        super(ApexO1AmpFP16, self).__init__(verbose=verbose, **kwargs)

    def wrap_fp16(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, OSS]] = None,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps model and optimizer with Apex O1 mixed-precision related backend wrappers

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        model: torch.nn.Module
            modified version of model object for mixed-precision backends
        optimizer: Union[torch.optim.Optimizer, OSS]]
            modified version of optimizer object for mixed-precision backends

        """
        self._scaler_info()
        if self._apex_config.convert_to_sync_batch_norm:
            model = self._apex_convert_to_sync_batch_norm(model=model)
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level="O1",
            cast_model_outputs=self._apex_config.cast_model_outputs,
            max_loss_scale=self._apex_config.max_loss_scale,
            min_loss_scale=self._apex_config.min_loss_scale,
            verbosity=self._apex_config.verbosity,
            num_losses=self._multi_loss if self._apex_config.scaler_per_loss else 1,
        )
        return model, optimizer


class NativeAmpFP16(BaseFP16):
    """Base class for PyTorch Native AMP FP16 methods

    This class handles base and common functionality for native PyTorch AMP mixed-precision backends. Contains
    functionality related to gradient clipping, backward call, step call, and context wrappers for the model forward
    and loss calls

    Attributes
    ----------
    loss_context
    model_context
    scaler
    _amp_config: AMPConfig
        Configuration object for Apex
    _scaler: default: torch.cuda.amp.GradScaler
        scaler object for loss
    _verbose bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, verbose: bool = True, **kwargs):
        """Init for NativeAmpFP16 class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call -- here amp_config or sharded_config might be passed in

        Notes
        -----
        Scaler set between torch.cuda.amp.GradScaler and ShardedGradScaler depending on if a sharded config is passed
        via kwargs

        """
        self._amp_config = kwargs["amp_config"]
        # Switch the scaler obj ref depending on fairscale sharding
        scaler = (
            ShardedGradScaler
            if (kwargs["sharded_config"] is not None)
            or (kwargs["fully_sharded_config"] is not None)
            else torch.cuda.amp.GradScaler
        )
        super(NativeAmpFP16, self).__init__(
            scaler=scaler(
                backoff_factor=self._amp_config.backoff_factor,
                enabled=True,
                growth_factor=self._amp_config.growth_factor,
                growth_interval=self._amp_config.growth_interval,
                init_scale=self._amp_config.init_scale,
            ),
            verbose=verbose,
        )

    @property
    def loss_context(self):
        """Overrides base and returns the native AMP autocast context"""
        return torch.cuda.amp.autocast(enabled=True)

    @property
    def model_context(self):
        """Overrides base and returns the native AMP autocast context"""
        return torch.cuda.amp.autocast(enabled=True)

    def backward_call(
        self,
        loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
    ):
        """Overrides base wrapped backward call for AMP scaled backward call

        Parameters
        ----------
        loss: loss: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            loss tensor(s)
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        if isinstance(loss, (list, tuple)):
            for idx, val in enumerate(loss):
                self._scaler.scale(val).backward(retain_graph=(idx == 0))
        else:
            self._scaler.scale(loss).backward()

    def step_call(
        self, model: torch.nn.Module, optimizer: Union[torch.optim.Optimizer, OSS]
    ):
        """Overrides base wrapped step of the optimizer with the AMP scaler version

        Parameters
        ----------
        model: torch.nn.Module
            current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        Returns
        -------
        None

        """
        self.scaler.step(optimizer)
        self.scaler.update()


class RunnerFP16Enum(Enum):
    """Enum for building the runtime object with mixed-precision functionality"""

    full = NullFP16
    apex_O1 = ApexO1AmpFP16
    apex_O2 = ApexO2AmpFP16
    amp = NativeAmpFP16
    deepspeed = DeepspeedFP16
