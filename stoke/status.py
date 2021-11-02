# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles setting the status/state of Stoke"""

import os
from enum import Enum
from typing import List, Optional, Union

import attr
import torch

from stoke.configs import (
    AMPConfig,
    ApexConfig,
    ClipGradConfig,
    ClipGradNormConfig,
    DDPConfig,
    DeepspeedConfig,
    DeepspeedFP16Config,
    FairscaleFSDPConfig,
    FairscaleOSSConfig,
    FairscaleSDDPConfig,
    HorovodConfig,
)
from stoke.extensions import _FairscaleFSDPConfig


class DistributedOptions(Enum):
    """Enum that defines the options for Distributed backends"""

    horovod = "horovod"
    ddp = "ddp"
    deepspeed = "deepspeed"


class FP16Options(Enum):
    """Enum that defines the options for FP16 backends"""

    apex_O1 = "apex_O1"
    apex_O2 = "apex_O2"
    amp = "amp"
    deepspeed = "deepspeed"


class _MissingLocalRankException(Exception):
    """Custom exception for when local rank cannot be found"""

    pass


class StokeStatus:
    """Low level stoke object that manages and sets the status of the overall run time configuration

    Based on the set flags this object checks for valid combinations (as there are a lot that will not work together)
    and builds a status object whose attributes are forwarded on via property decorators. Handles managing init of
    backend config objects and any post init modifications.

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
    fp16
    fsdp_config
    fully_sharded
    gpu
    grad_accum
    grad_clip
    horovod_config
    is_distributed_ddp
    is_distributed_deepspeed
    is_distributed_horovod
    is_fairscale
    is_fp16_apex
    is_fp16_deepspeed
    nccl
    oss
    oss_config
    sddp_config
    sharded
    status
    zero

    _configs: dict
        dictionary of config objects
    _key_list: list
        valid config objects to check against
    _status: dict
        dictionary that is the current requested state of Stoke

    """

    def __init__(
        self,
        batch_size_per_device: int,
        grad_accum: Optional[int],
        grad_clip: Optional[Union[ClipGradConfig, ClipGradNormConfig]],
        gpu: bool,
        fp16: Optional[FP16Options],
        distributed: Optional[DistributedOptions],
        fairscale_oss: bool,
        fairscale_sddp: bool,
        fairscale_fsdp: bool,
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
        ],
    ):
        """Init for StokeStatus class object

        Parameters
        ----------
        batch_size_per_device: int
            Batch size at the single device level
        grad_accum: Optional[int], default: 1
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
        configs: Optional[List[Union[AMPConfig, ApexConfig, DDPConfig, DeepspeedConfig, FairscaleOSSConfig, FairscaleSDDPConfig, HorovodConfig]], default: None
            Configuration objects for runtimes
        """
        # Allowed keys for configs
        self._key_list = [
            "AMPConfig",
            "ApexConfig",
            "DDPConfig",
            "DeepspeedConfig",
            "FairscaleOSSConfig",
            "FairscaleSDDPConfig",
            "FairscaleFSDPConfig" "HorovodConfig",
        ]
        # Set the configs first which allows for checking of some config vars later
        self._configs = self._set_configs(configs=configs)
        if (grad_clip is not None) and not isinstance(
            grad_clip, (ClipGradConfig, ClipGradNormConfig)
        ):
            raise TypeError(
                f"Stoke -- grad_clip argument must be of type ClipGradConfig or ClipGradNormConfig"
            )
        # Set simple state vars -- post combo check so validity is fine to set
        self._status = {
            "cuda": torch.cuda.is_available(),
            "nccl": torch.distributed.is_nccl_available(),
            "batch_size": batch_size_per_device,
            "grad_accum": grad_accum if grad_accum is not None else 1,
            "grad_clip": grad_clip,
            "gpu": gpu,
            "distributed": distributed,
            "zero": self._configs.get("DeepspeedConfig").zero_optimization.stage
            if self._configs.get("DeepspeedConfig")
            else None,
            "oss": fairscale_oss,
            "sharded": fairscale_sddp,
            "fully_sharded": fairscale_fsdp,
            "world_size": -1,
        }
        # Check fp16 since it might need APEX imports and update state dict
        self._status.update({"fp16": self._set_fp16(fp16=fp16)})
        # Check all the invalid combinations
        self._check_all_raised_combinations()

    def _check_all_raised_combinations(self):
        """Checks all acceptable/restricted combinations and raises exceptions for any invalid combinations

        README.md should have a table of acceptable combinations

        Returns
        -------
        None

        """
        # No gpu if no CUDA
        if self.gpu and not self.cuda:
            raise ValueError("Stoke -- GPU(s) cannot be used as CUDA is not available")
        # No fairscale and deepspeed
        if self.is_fairscale and (
            self.is_distributed_deepspeed or self.is_fp16_deepspeed
        ):
            raise ValueError(
                f"Stoke -- Cannot use both fairscale extensions "
                f"(currently: oss: {self.oss}, sddp: {self.sharded}) "
                f"and deepspeed (currently: distributed: {self.is_distributed_deepspeed}, "
                f"fp16: {self.is_fp16_deepspeed})"
            )
        # No Distributed without gpu, cuda, and nccl
        if (
            not self.cuda or not self.gpu or not self.nccl
        ) and self.distributed is not None:
            raise ValueError(
                f"Stoke -- Distributed requires CUDA (currently: {self.cuda}), GPU (currently: {self.gpu}), "
                f"and NCCL (currently: {self.nccl})"
            )
        # No FP16 without CUDA
        if not self.cuda and (self.fp16 is not None):
            raise ValueError(f"Stoke -- FP16 training requires CUDA availability")
        # No fairscale without gpu, cuda, and nccl and DDP (will catch Horovod)
        if (
            not self.cuda
            or not self.gpu
            or not self.nccl
            or not self.is_distributed_ddp
        ) and self.is_fairscale:
            raise ValueError(
                f"Stoke -- Fairscale extensions (currently: oss: {self.oss}, sddp: {self.sharded}) "
                f"requires CUDA (currently: {self.cuda}), "
                f"GPU (currently: {self.gpu}), "
                f"DDP (currently: {self.is_distributed_ddp}) and NCCL (currently: {self.nccl})"
            )
        # No SDDP w/o OSS
        if self.sharded and not self.oss:
            raise ValueError(
                f"Stoke -- Fairscale SDDP requires OSS (currently: oss: {self.oss}, sddp: {self.sharded})"
            )
        # FSDP stands alone
        if (self.sharded or self.oss) and self.fully_sharded:
            raise ValueError(
                f"Stoke -- Fairscale FSDP does not require SDDP or OSS as it manages OSS itself"
                f"(currently: oss: {self.oss}, sddp: {self.sharded}. fsdp: {self.fully_sharded})"
            )
        # No fairscale with APEX
        if self.is_fairscale and self.is_fp16_apex:
            raise ValueError(
                f"Stoke -- Fairscale does not currently support APEX (currently: {self.is_fp16_apex}) "
                f"for mixed precision"
            )
        # No fairscale oss with grad clip by value
        if (self.oss or self.fully_sharded) and isinstance(
            self.grad_clip, ClipGradConfig
        ):
            raise ValueError(
                f"Stoke -- Fairscale OSS and FSDP do not currently support torch.nn.utils.clip_grad_value_ "
                f"(currently: {type(self.grad_clip).__name__})"
            )
        # No deepspeed FP16 without deepspeed distributed
        if self.is_fp16_deepspeed and not self.is_distributed_deepspeed:
            raise ValueError(
                f"Stoke -- Deepspeed FP16 (currently: {self.is_fp16_deepspeed}) requires the use of "
                f"Deepspeed distributed (currently: {self.is_distributed_deepspeed})"
            )
        # No other FP16 with deepspeed distributed
        if (
            self.is_distributed_deepspeed
            and self.fp16 is not None
            and not self.is_fp16_deepspeed
        ):
            raise ValueError(
                f"Stoke -- Deepspeed distributed (currently: {self.is_distributed_deepspeed}) only "
                f"supports its own internal FP16 implementation (currently: {self.fp16})"
            )
        # No zero > 0 without deepspeed FP16
        if (
            self.is_distributed_deepspeed
            and self.zero > 0
            and not self.is_fp16_deepspeed
        ):
            raise ValueError(
                f"Stoke -- Deepspeed ZeRO extension (currently: Stage-{self.zero}) requires Deepspeed"
                f"FP16 extension (currently: {self.is_fp16_deepspeed})"
            )

    def _set_fp16(self, fp16: Optional[FP16Options]):
        """Sets the state of the FP16 backend

        Seeing as the APEX install is not packaged currently with Stoke (or if it is requires building some things from
        source it's liable to fail). Handling it this way allows Stoke not to break if APEX isn't installed correctly

        Parameters
        ----------
        fp16: FP16Options, optional
            Enum that defines the options for FP16 backends

        Returns
        -------
        FP16Options or None

        """
        if self._status.get("cuda") and (fp16 is not None):
            if fp16 == "apex_O1" or fp16 == "apex_O2":
                # Try/Except the apex import to see if it's available
                try:
                    from apex import amp
                except ImportError as e:
                    print(
                        e,
                        ": Stoke -- apex cannot be imported -- please install (https://github.com/NVIDIA/apex)",
                    )
            return fp16
        else:
            return None

    def _set_configs(self, configs):
        """Determines which configs were set from user input and sets all others to None

        Parameters
        ----------
        configs: list
            List of any user specified run time configs

        Returns
        -------
        config_dict: dict or None
            dictionary of config objects or None

        """
        # Set those that are specified within a dict
        if configs is not None:
            config_dict = {type(val).__name__: val for val in configs}
        else:
            config_dict = {}
        # Set those missing within the existing config dict to None so property accessors work correctly
        none_dict = {val: None for val in self._key_list if val not in config_dict}
        config_dict.update(none_dict)
        return config_dict

    def set_post_init_values(self, world_size: int):
        """Sets post-init values that cannot be set prior to run-time instantiation

        Some values cannot be accessed until after run-time instantiation as the property accessors are not setup yet

        Parameters
        ----------
        world_size: int
            current distributed world size

        Returns
        -------
        None

        """
        self._status.update({"world_size": world_size})

    @property
    def status(self):
        """Shortcut to status dict"""
        return self._status

    @property
    def batch_size(self):
        """Shortcut to batch size"""
        return self._status.get("batch_size")

    @property
    def effective_batch_size(self):
        """Shortcut to effective batch size"""
        return self.batch_size * self.grad_accum * self._status.get("world_size")

    @property
    def grad_clip(self):
        """Shortcut to get grad clip"""
        return self._status.get("grad_clip")

    @property
    def grad_accum(self):
        """Shortcut to get grad accumulation"""
        return self._status.get("grad_accum")

    @property
    def gpu(self):
        """Shortcut to get GPU status"""
        return self._status.get("gpu")

    @property
    def cuda(self):
        """Shortcut to get cuda status"""
        return self._status.get("cuda")

    @property
    def nccl(self):
        """Shortcut to get nccl status"""
        return self._status.get("nccl")

    @property
    def fp16(self):
        """Shortcut to get FP16 status"""
        return self._status.get("fp16")

    @property
    def is_fp16_apex(self):
        """Returns if APEX is activated"""
        return self.fp16 == "apex_O1" or self.fp16 == "apex_O2"

    @property
    def is_fp16_amp(self):
        """Returns if AMP is activated"""
        return self.fp16 == "amp"

    @property
    def is_fp16_deepspeed(self):
        """Returns if Deepspeed FP16 is activated"""
        return self.fp16 == "deepspeed"

    @property
    def oss(self):
        """Returns if Fairscale optimizer state sharding status"""
        return self._status.get("oss")

    @property
    def sharded(self):
        """Returns if Fairscale sharded DDP status"""
        return self._status.get("sharded")

    @property
    def fully_sharded(self):
        """Returns if Fairscale fully sharded DDP status"""
        return self._status.get("fully_sharded")

    @property
    def world_size(self):
        """Returns the current world size"""
        return self._status.get("world_size")

    @property
    def zero(self):
        """Returns what stage of ZeRO Deepspeed is using"""
        return self._status.get("zero")

    @property
    def is_fairscale(self):
        """Returns if any part of Fairscale is activated"""
        return self.oss or self.sharded or self.fully_sharded

    @property
    def distributed(self):
        """Shortcut to distributed setting"""
        return self._status.get("distributed")

    @property
    def is_distributed_deepspeed(self):
        """Returns if Deepspeed is activated"""
        return self.distributed == "deepspeed"

    @property
    def is_distributed_ddp(self):
        """Returns if DDP is activated"""
        return self.distributed == "ddp"

    @property
    def is_distributed_horovod(self):
        """Returns if Horovod is activated"""
        return self.distributed == "horovod"

    @property
    def apex_config(self):
        """Checks for user defined ApexConfig and/or sets a default config object

        Returns
        -------
        ApexConfig
            User set ApexConfig or the defaulted version

        """
        config = self._configs.get("ApexConfig")
        return config if config is not None else ApexConfig()

    @property
    def amp_config(self):
        """Checks for user defined AMPConfig and/or sets a default config object

        Returns
        -------
        AMPConfig
            User set AMPConfig or the defaulted version

        """
        config = self._configs.get("AMPConfig")
        return config if config is not None else AMPConfig()

    @property
    def ddp_config(self):
        """Checks for user defined DDPConfig and/or sets a default config object

        Handles some post init logic looking for LOCAL_RANK and raises if it cannot find it
        https://pytorch.org/docs/stable/distributed.html#launch-utility

        Returns
        -------
        DDPConfig
            User set DDPConfig or the defaulted version

        """
        config = self._configs.get("DDPConfig")
        # Here need to check if the config passed through defined the local rank or not...
        # Assuming that it's being caught from the arg parser... if not try and grab it from
        # the env (set from the launcher)
        if config is not None and config.local_rank is None:
            try:
                local_rank = int(os.environ["LOCAL_RANK"])
            except _MissingLocalRankException:
                raise _MissingLocalRankException(
                    f"Stoke -- Device local rank must be defined within the DDPConfig "
                    f" (handled by parsing --local_arg from the torch.distributed.launch "
                    f"command) or defined as env variable LOCAL_RANK (handled by calling "
                    f"torch.distributed.launch with the --use_env flag)"
                )
            # Evolve the config if grabbing from the env variable
            config = attr.evolve(config, local_rank=local_rank)
        elif config is None:
            try:
                local_rank = int(os.environ["LOCAL_RANK"])
            except _MissingLocalRankException:
                raise _MissingLocalRankException(
                    f"Stoke -- Device local rank must be defined within the DDPConfig "
                    f" (handled by parsing --local_arg from the torch.distributed.launch "
                    f"command) or defined as env variable LOCAL_RANK (handled by calling "
                    f"torch.distributed.launch with the --use_env flag)"
                )
            # Set a default config with the local rank from the env
            config = DDPConfig(local_rank=local_rank)
        return config

    @property
    def deepspeed_config(self):
        """Checks for user defined DeepspeedConfig and/or sets a default config object

        Handles the internal logic of Deepspeed FP16 as it is a status flag in the config and not a class object
        like AMP or APEX

        Returns
        -------
        DeepspeedConfig
            User set DeepspeedConfig or the defaulted version

        """
        config = self._configs.get("DeepspeedConfig")
        # Deepspeed only has a single config so FP16 needs to be handled here based on the status flag if no config
        # is passed through
        # Fall back to basics of both if no config
        if self.fp16 == "deepspeed" and config is None:
            config = DeepspeedConfig(fp16=DeepspeedFP16Config())
        # Fall back to defaults if a config is passed but the FP16 Config wasn't set
        elif self.fp16 == "deepspeed" and config is not None and config.fp16 is None:
            config = attr.evolve(config, fp16=DeepspeedFP16Config())
        # Fall back to hard defaults if just using distributed
        elif config is None:
            config = DeepspeedConfig()
        else:
            config = config
        return config

    @property
    def oss_config(self):
        """Checks for user defined FairscaleOSSConfig and/or sets a default config object

        Returns
        -------
        FairscaleOSSConfig
            User set FairscaleOSSConfig or the defaulted version

        """
        config = self._configs.get("FairscaleOSSConfig")
        return config if config is not None else FairscaleOSSConfig()

    @property
    def sddp_config(self):
        """Checks for user defined FairscaleSDDPConfig and/or sets a default config object

        Returns
        -------
        FairscaleSDDPConfig
            User set FairscaleSDDPConfig or the defaulted version

        """
        config = self._configs.get("FairscaleSDDPConfig")
        return config if config is not None else FairscaleSDDPConfig()

    @property
    def fsdp_config(self):
        """Checks for user defined FairscaleFSDPConfig and/or sets a default config object

        Mutates the default attr class to contain the mixed_precision attribute that is derived from FP16 settings

        Returns
        -------
        FairscaleFSDPConfig mutated with mixed-precision state

        """
        config = self._configs.get("FairscaleFSDPConfig")
        # Swap in a default config if none
        if config is None:
            config = FairscaleFSDPConfig()
        # Handle FP16 settings if set via constructor -- these need to be morphed at runtime to a new attr class
        config_dict = attr.asdict(config)
        config_dict.update({"mixed_precision": self.is_fp16_amp})
        return _FairscaleFSDPConfig(**config_dict)

    @property
    def horovod_config(self):
        """Checks for user defined HorovodConfig and/or sets a default config object

        Returns
        -------
        HorovodConfig
            User set HorovodConfig or the defaulted version

        """
        config = self._configs.get("HorovodConfig")
        return config if config is not None else HorovodConfig()

    def __repr__(self):
        """Formats the status for pretty printing

        Returns
        -------
        str
            pretty formatted status string

        """
        return (
            f"STOKE STATE:\n"
            f"    CUDA AVAILABLE: {self.cuda}\n"
            f"    NCCL AVAILABLE: {self.nccl}\n"
            f"    GPU FLAG: {self.gpu}\n"
            f"    FP16 FLAG: {self.fp16}\n"
            f"    DISTRIBUTED BACKEND: {self.distributed}\n"
            f"    FAIRSCALE OSS: {self.oss}\n"
            f"    FAIRSCALE SDDP: {self.sharded}\n"
            f"    FAIRSCALE FSDP: {self.fully_sharded}\n"
            f'    DEEPSPEED ZeRO: {f"Stage {self.zero}" if self.is_distributed_deepspeed else f"False"}\n'
            f"    WORLD SIZE: {self.world_size}\n"
            f"    GRAD ACCUMULATION STEPS: {self.grad_accum}\n"
            f"    BATCH SIZE (PER DEVICE): {self.batch_size}\n"
            f"    EFFECTIVE BATCH SIZE (ALL DEVICES): {self.effective_batch_size}\n"
            f'    GRAD CLIP: ({", ".join(f"{k}: {v}" for k, v in attr.asdict(self.grad_clip).items()) if self.grad_clip is not None else "None"})'
        )
