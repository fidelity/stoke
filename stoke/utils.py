# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Stoke utility functions"""

import os
from enum import Enum
from typing import Any, Callable, List, Tuple, TypeVar, Union

import torch
from fairscale.optim.oss import OSS

# Taken from torch/utils/data/dataloader
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

# Taken from torch/utils/data/dataloader
_worker_init_fn_t = Callable[[int], None]
# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`,
# but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


class ParamNormalize(Enum):
    """Normalization enum for total number of model parameters used to help with a pretty print"""

    THOUSAND = 1e3
    MILLION = 1e6
    BILLION = 1e9
    TRILLION = 1e12


def zero_optimizer_grads(
    optimizer: Union[torch.optim.Optimizer, OSS],
    apex: bool = False,
    horovod: bool = False,
):
    """Zeros grads depending on if it is a base Torch optimizer or a Fused version from APEX

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        current optimizer object
    apex: bool, default: False
        if apex is active
    horovod: bool, default: False
        if horovod is active
    Returns
    -------
    None

    """
    if (optimizer.__class__.__name__.find("Fused") == -1) and not apex and not horovod:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()


def unrolled_print(msg: Union[str, List[str], Tuple[str]], single_line: bool = False):
    """Prints the msg if it's a string or iterable of strings

    Parameters
    ----------
    msg: Union[str, List[str], Tuple[str]
        string(s) to print
    single_line: bool, default: False
        if iterable print all on one line space and comma separated

    Returns
    -------
    None

    """
    if isinstance(msg, (list, tuple)):
        if single_line:
            msg = type(msg)(
                f"Stoke -- {val}" if idx == 0 else f"{val}"
                for idx, val in enumerate(msg)
            )
        else:
            msg = type(msg)(f"Stoke -- {val}" for idx, val in enumerate(msg))
        print(*msg, sep=", " if single_line else "\n")
    else:
        print(f"Stoke -- {msg}")


def make_folder(path: str):
    """

    Parameters
    ----------
    path: str
        path to write

    Returns
    -------

    """
    # Make the folder if it doesn't exist
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
