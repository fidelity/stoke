# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles extension wrapper related classes -- mixin style"""

from abc import ABC
from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

import torch
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim.oss import OSS

from stoke.configs import DDPConfig, FairscaleOSSConfig, FairscaleSDDPConfig


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
    """Base class for using the DDP backend

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
            Base Fairscale ShardedDataParallel configuration obje
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


class DistributedHandlerEnum(Enum):
    """Enum for DDP use"""

    sddp = FairscaleSDDPExtension
    base = BaseDDP
