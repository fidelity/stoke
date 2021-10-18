[![Stoke](https://raw.githubusercontent.com/fidelity/stoke/master/resources/images/logo_and_text.png)](https://fidelity.github.io/stoke/)
> Add a little accelerant to your torch

[![License](https://img.shields.io/badge/License-Apache%202.0-9cf)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.6+-informational.svg)]()
[![Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Lint](https://github.com/fidelity/stoke/workflows/lint/badge.svg?branch=master)
![Docs](https://github.com/fidelity/stoke/workflows/docs/badge.svg)
---

## About

`stoke` is a lightweight wrapper for PyTorch that provides a simple declarative API for context switching between 
devices (e.g. CPU, GPU), distributed modes, mixed-precision, and PyTorch extensions. This allows you to switch from 
local full-precision CPU to mixed-precision distributed multi-GPU with extensions (like optimizer state sharding) 
by simply changing a few declarative flags. Additionally, `stoke` exposes configuration settings for every 
underlying backend for those that want configurability and raw access to the underlying libraries.

In short, `stoke` is the best of 
[PyTorch Lightning Accelerators](https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerators.html) 
disconnected from the rest of PyTorch Lightning. Write whatever PyTorch code you want, but leave device and backend 
context switching to `stoke`.

## Supports

 * Devices: CPU, GPU, multi-GPU
 * Distributed: [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel), [Horovod](https://horovod.readthedocs.io/en/stable/index.html), [deepspeed](https://github.com/microsoft/DeepSpeed) (via DDP)
 * Mixed-Precision: [AMP](https://pytorch.org/docs/stable/amp.html), [Nvidia Apex](https://github.com/NVIDIA/apex), [deepspeed](https://github.com/microsoft/DeepSpeed) (custom APEX like backend)
 * Extensions: [fairscale](https://github.com/facebookresearch/fairscale) (Optimizer State Sharding, Sharded DDP, Fully Sharded DDP), [deepspeed](https://github.com/microsoft/DeepSpeed) (ZeRO Stage 0-3, etc.)

## Benefits/Capabilities

- Declarative style API -- allows you to declare or specify the desired state and let `stoke` handle the rest
- Mirrors base PyTorch style `model`, `loss`, `backward`, and `step` calls
- Automatic device placement of model(s) and data
- Universal interface for saving and loading regardless of backend(s) or device
- Automatic handling of gradient accumulation and clipping
- Common `attrs` interface for all backend configuration parameters (with docstrings)
- Helper methods for printing synced losses, device specific print, number of model parameters
- Extra(s) - Custom torch.utils.data.distributed.Sampler: BucketedDistributedSampler which buckets data by 
  a sorted idx and then randomly samples from specific bucket(s) to prevent situations like grossly mismatched sequence 
  length leading to wasted computational overhead (ie excess padding)