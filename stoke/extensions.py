# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles extension wrapper related classes -- mixin style"""

from abc import ABC
from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

import attr
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel, ShardedDataParallel
from fairscale.optim.oss import OSS

from stoke.configs import (
    DDPConfig,
    FairscaleFSDPConfig,
    FairscaleOSSConfig,
    FairscaleSDDPConfig,
)


@attr.s(auto_attribs=True)
class _FairscaleFSDPConfig(FairscaleFSDPConfig):
    mixed_precision: bool = False


class BaseOptimizer(ABC):
    """Base class for creating an optimizer

    Attributes
    ----------
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, verbose: bool = True, **kwargs):
        """Init for BaseOptimizer class

        Parameters
        ----------
        verbose: bool, default: True
            flag for verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        """
        self._verbose = verbose

    def build_optimizer(
        self,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """Instantiates a torch optimizer object from the type and optimizer kwargs

        Parameters
        ----------
        optimizer: Type[torch.optim.Optimizer]
            type of torch optimizer
        optimizer_kwargs: Dict
            dictionary of all kwargs to pass to the optimizer
        model: torch.nn.Module
            model object

        Returns
        -------
        torch.optim.Optimizer
            instantiated torch optimizer object

        """
        if self._verbose:
            self._print_device(f"Creating basic torch optimizer: {optimizer.__name__}")
        return optimizer(params=model.parameters(), **optimizer_kwargs)


class FairscaleOSSExtension(BaseOptimizer):
    """Inherits from BaseOptimizer for OSS class creation

    Attributes
    ----------
    _oss_config: FairscaleOSSConfig,
        Configuration object for Fairscale OSS
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, oss_config: FairscaleOSSConfig, verbose: bool = True, **kwargs):
        """Init for FairscaleOSSExtension class

        Parameters
        ----------
        oss_config: FairscaleOSSConfig
            Configuration object for Fairscale OSS
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        """
        super(FairscaleOSSExtension, self).__init__(verbose=verbose)
        self._oss_config = oss_config

    def build_optimizer(
        self,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict,
        model: torch.nn.Module,
    ) -> OSS:
        """Instantiates a Fairscale OSS optimizer object from the type and optimizer kwargs

        Parameters
        ----------
        optimizer: Type[torch.optim.Optimizer]
            type of torch optimizer
        optimizer_kwargs: Dict
            dictionary of all kwargs to pass to the optimizer
        model: torch.nn.Module
            model object

        Returns
        -------
        OSS
            instantiated Fairscale OSS optimizer object

        """
        if self._verbose:
            self._print_device(
                f"Creating Fairscale OSS wrapped PyTorch optimizer: {optimizer.__name__}"
            )
        return OSS(
            params=model.parameters(),
            optim=optimizer,
            broadcast_fp16=self._oss_config.broadcast_fp16,
            **optimizer_kwargs,
        )


class RunnerOptimizerEnum(Enum):
    """Enum for optimizer creation"""

    oss = FairscaleOSSExtension
    base = BaseOptimizer


class BaseDDP:
    """Base class for using the DDP backend

    Attributes
    ----------
    _ddp_config: DDPConfig
        Base DDP configuration object
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(self, ddp_config: DDPConfig, verbose: bool = True, **kwargs):
        """Init for BaseDDP

        Parameters
        ----------
        ddp_config: DDPConfig
            Base DDP configuration object
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        """
        self._verbose = verbose
        self._ddp_config = ddp_config

    def handle_ddp(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        grad_accum: Optional[int],
        rank: int,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps the model in the base DDP call

        Parameters
        ----------
        model: torch.nn.Module
            Current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps
        rank: int
            Current CUDA device rank in the distributed setup

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        """
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[rank],
            output_device=rank,
            bucket_cap_mb=self._ddp_config.bucket_cap_mb,
            broadcast_buffers=self._ddp_config.broadcast_buffers,
            find_unused_parameters=self._ddp_config.find_unused_parameters,
            gradient_as_bucket_view=self._ddp_config.gradient_as_bucket_view,
        )
        return model, optimizer


class FairscaleSDDPExtension:
    """Class for using the Fairscale SDDP backend

    Attributes
    ----------
    _sddp_config: FairscaleSDDPConfig
        Base Fairscale ShardedDataParallel configuration object
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self, sddp_config: FairscaleSDDPConfig, verbose: bool = True, **kwargs
    ):
        """Init for FairscaleSDDPExtension

        Parameters
        ----------
        sddp_config: FairscaleSDDPConfig
            Base Fairscale ShardedDataParallel configuration objet
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        """
        self._verbose = verbose
        self._sddp_config = sddp_config

    def handle_ddp(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        grad_accum: Optional[int],
        rank: int,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps the model in the ShardedDataParallel call

        Parameters
        ----------
        model: torch.nn.Module
            Current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps
        rank: int
            Current CUDA device rank in the distributed setup

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        """
        model = ShardedDataParallel(
            module=model,
            sharded_optimizer=optimizer,
            broadcast_buffers=self._sddp_config.broadcast_buffers,
            sync_models_at_startup=self._sddp_config.sync_models_at_startup,
            reduce_buffer_size=self._sddp_config.reduce_buffer_size,
            auto_refresh_trainable=self._sddp_config.auto_refresh_trainable,
            reduce_fp16=self._sddp_config.reduce_fp16,
        )
        return model, optimizer


class FairscaleFSDPExtension:
    """Class for using the Fairscale FSDP backend

    Attributes
    ----------
    _fsdp_config: _FairscaleFSDPConfig
        Base Fairscale Fully Sharded Data Parallel configuration object
    _verbose: bool, default: True
        flag for Stoke print verbosity

    """

    def __init__(
        self, fsdp_config: _FairscaleFSDPConfig, verbose: bool = True, **kwargs
    ):
        """Init for FairscaleSDDPExtension

        Parameters
        ----------
        _fsdp_config: _FairscaleFSDPConfig
            Base Fairscale Fully Sharded Data Parallel configuration object
        verbose: bool, default: True
            flag for Stoke print verbosity
        **kwargs: dict, optional
            Extra arguments passed to the __init__ call

        """
        self._verbose = verbose
        self._fsdpp_config = fsdp_config

    def handle_ddp(
        self,
        model: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, OSS],
        grad_accum: Optional[int],
        rank: int,
    ) -> Tuple[torch.nn.Module, Union[torch.optim.Optimizer, OSS]]:
        """Wraps the model in the FullyShardedDataParallel call

        Also sets grad divide factors
        https://fairscale.readthedocs.io/en/latest/_modules/fairscale/nn/data_parallel/fully_sharded_data_parallel.html#FullyShardedDataParallel.set_gradient_divide_factors

        Parameters
        ----------
        model: torch.nn.Module
            Current model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            Current optimizer object
        grad_accum: int, default: None
            Number of gradient accumulation steps
        rank: int
            Current CUDA device rank in the distributed setup

        Returns
        -------
        model: torch.nn.Module
            Wrapped model object
        optimizer: Union[torch.optim.Optimizer, OSS]
            current optimizer object

        """
        model = FullyShardedDataParallel(
            module=model,
            reshard_after_forward=self._fsdpp_config.reshard_after_forward,
            mixed_precision=self._fsdpp_config.mixed_precision,
            fp32_reduce_scatter=self._fsdpp_config.fp32_reduce_scatter,
            flatten_parameters=self._fsdpp_config.flatten_parameters,
            move_params_to_cpu=self._fsdpp_config.move_params_to_cpu,
            compute_dtype=self._fsdpp_config.compute_dtype,
            buffer_dtype=self._fsdpp_config.buffer_dtype,
            move_grads_to_cpu=self._fsdpp_config.move_grads_to_cpu,
            bucket_cap_mb=self._fsdpp_config.bucket_cap_mb,
            no_broadcast_optim_state=self._fsdpp_config.no_broadcast_optim_state,
            clear_autocast_cache=self._fsdpp_config.clear_autocast_cache,
            force_input_to_fp32=self._fsdpp_config.force_input_to_fp32,
            verbose=self._fsdpp_config.verbose,
        )
        # Trigger the set of pre-divide or post-divide factors if set in the config
        model.set_gradient_divide_factors(
            pre=self._fsdpp_config.gradient_predivide_factor
            if self._fsdpp_config.gradient_predivide_factor is not None
            else model.gradient_predivide_factor,
            post=self._fsdpp_config.gradient_postdivide_factor
            if self._fsdpp_config.gradient_postdivide_factor is not None
            else model.gradient_postdivide_factor,
            recursive=True,
        )
        return model, optimizer


class DistributedHandlerEnum(Enum):
    """Enum for DDP use"""

    sddp = FairscaleSDDPExtension
    fsdp = FairscaleFSDPExtension
    base = BaseDDP
