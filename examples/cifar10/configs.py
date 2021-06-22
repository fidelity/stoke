# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 training script configs via spock"""

from typing import Optional, Tuple

from spock.config import spock

from stoke import DistributedOptions, FP16Options


@spock
class RunConfig:
    """Simple spock config class for some stoke options/settings

    Attributes:
        gpu: flag to use gpu
        grad_accum: number of gradient accumulation steps
        distributed: option to choose a distributed backend
        fp16: option to choose a fp16/mixed-precision backend
        oss: flag to use fairscale optimizer state sharding
        sddp: flag to use fairscale sharded DDP
        checkpoint_path: path to save model checkpoint
        checkpoint_name: name of model checkpoint
        num_epochs: number of epoch to train for

    """

    gpu: bool = False
    grad_accum: Optional[int]
    distributed: Optional[DistributedOptions]
    fp16: Optional[FP16Options]
    oss: bool = False
    sddp: bool = False
    checkpoint_path: str
    checkpoint_name: str
    num_epoch: int


@spock
class ZeROConfig:
    """Optional config for using Deepspeed ZeRO implementations

    Attributes:
        zero: option to use a deepspeed ZeRO stage
        contiguous_gradients: use contiguous gradients with ZeRO
        overlap_comm: overlap communication with ZeRO

    """

    zero: Optional[int] = 0
    contiguous_gradients: bool = False
    overlap_comm: bool = False


@spock
class OSSConfig:
    broadcast_fp16: bool = False


@spock
class SDDPConfig:
    reduce_fp16: bool = False


@spock
class DataConfig:
    batch_size: int
    n_workers: Optional[int]
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]
    root_dir: str
    crop_size: int
    crop_pad: int


@spock
class SGDConfig:
    lr: float
    momentum: float
    weight_decay: float
