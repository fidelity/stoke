# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Stoke setup.py"""

import os

import setuptools

import versioneer


def _handle_reqs(req_path):
    """Handles any non standard refs to installs (git+ notation)

    Loosely based on https://stackoverflow.com/a/57500829

    Parameters
    ----------
    req_path: str
        path to a requirements file

    Returns
    -------
    list
        processed list of requirements

    """
    with open(req_path, "r") as fid:
        pre_reqs = fid.read().splitlines()
    EGG_MARK = "#egg="
    for idx, line in enumerate(pre_reqs):
        # Catch anything that is git+
        if line.startswith("git+"):
            if EGG_MARK in line:
                egg_idx = line.rindex(EGG_MARK)
                name = line[(egg_idx + len(EGG_MARK)) :]
                repo = line[:egg_idx]
                pre_reqs[idx] = f"{name} @ {repo}"
            else:
                raise SyntaxError(
                    "Dependency should have the format: \n"
                    "git+https://github.com/xxxx/xxxx#egg=package_name\n"
                    "-or-\n"
                    "git+ssh://git@github.com/xxxx/xxxx#egg=package_name"
                )
    return pre_reqs


# Process all the different reqs
install_reqs = _handle_reqs("REQUIREMENTS.txt")
mpi_reqs = _handle_reqs("./requirements/MPI.txt")

# Export some env variables
# Make sure horovod with pytorch get installed
os.environ["HOROVOD_WITH_PYTORCH"] = "1"
# Make sure horovod is using NCCL for ops
os.environ["HOROVOD_GPU_OPERATIONS"] = "NCCL"
# Make sure fairscale fused ADAM cuda kernels get included
os.environ["BUILD_CUDA_EXTENSIONS"] = "1"

with open("README.md", "r") as fid:
    long_description = fid.read()

setuptools.setup(
    name="stoke",
    description="Lightweight wrapper for PyTorch that provides a simple unified interface for context switching "
    "between devices (CPU, GPU), distributed modes (DDP, Horovod), mixed-precision (AMP, Apex), and "
    "extensions (fairscale, deepspeed).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="FMR LLC",
    url="https://github.com/fidelity/stoke",
    classifiers=[
        # "Development Status :: 3 - Alpha",
        "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Source": "https://github.com/fidelity/stoke",
        "Documentation": "https://fidelity.github.io/stoke/",
        "Bug Tracker": "https://github.com/fidelity/stoke/issues",
    },
    keywords=[
        "deep learning",
        "pytorch",
        "AI",
        "gpu",
        "ddp",
        "horovod",
        "deepspeed",
        "fairscale",
        "distributed",
        "fp16",
        "mixed-precision",
        "apex",
        "amp",
    ],
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    python_requires=">=3.6",
    install_requires=install_reqs,
    extras_require={"mpi": mpi_reqs},
)
