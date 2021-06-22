# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Stoke setup.py"""

import os

import setuptools
from pkg_resources import parse_requirements

import versioneer

# Export some env variables
# Make sure horovod with pytorch get installed
os.environ["HOROVOD_WITH_PYTORCH"] = "1"
# Make sure fairscle fused ADAM cuda kernels get included
os.environ["BUILD_CUDA_EXTENSIONS"] = "1"

with open("README.md", "r") as fid:
    long_description = fid.read()

with open("REQUIREMENTS.txt", "r") as fid:
    install_reqs = [str(req) for req in parse_requirements(fid)]

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
        "Development Status :: 3 - Alpha",
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
        "apex",
        "amp",
    ],
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    python_requires=">=3.6",
    install_requires=install_reqs,
)
