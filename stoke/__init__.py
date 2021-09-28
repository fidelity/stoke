# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Stoke is a lightweight wrapper for PyTorch that provides a simple unified interface for context switching

Please refer to the documentation provided in the README.md
"""

from .configs import *
from .data import BucketedDistributedSampler
from .status import DistributedOptions, FP16Options
from .stoke import Stoke
from .utils import ParamNormalize

__all__ = [
    "Stoke",
    "ParamNormalize",
    "FP16Options",
    "DistributedOptions",
    "StokeOptimizer",
    "ClipGradNormConfig",
    "ClipGradConfig",
    "FairscaleOSSConfig",
    "FairscaleSDDPConfig",
    "FairscaleFSDPConfig",
    "HorovodConfig",
    "ApexConfig",
    "DeepspeedConfig",
    "DDPConfig",
    "AMPConfig",
    "DeepspeedAIOConfig",
    "DeepspeedActivationCheckpointingConfig",
    "DeepspeedFlopsConfig",
    "DeepspeedFP16Config",
    "DeepspeedPLDConfig",
    "DeepspeedOffloadOptimizerConfig",
    "DeepspeedOffloadParamConfig",
    "DeepspeedTensorboardConfig",
    "DeepspeedZeROConfig",
    "BucketedDistributedSampler",
]

from ._version import get_versions

__version__ = get_versions()["version"]

del get_versions
