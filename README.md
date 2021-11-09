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
devices (e.g. CPU, GPU), distributed modes, mixed-precision, and PyTorch extensions. It places no restrictions on code 
structure for model architecture, training/inference loops, loss functions, optimizer algorithm, etc. Stoke simply 
'wraps' your existing  PyTorch code to automatically handle the necessary underlying wiring for all of the 
supported backends.This allows you to switch from local full-precision CPU to mixed-precision distributed multi-GPU 
with extensions (like optimizer state sharding) by simply changing a few declarative flags. Additionally, `stoke` 
exposes configuration settings for every underlying backend for those that want configurability and raw access to 
the underlying libraries.

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
- Wrapped API Mirrors base PyTorch style `model`, `loss`, `backward`, and `step` calls
- Automatic device placement of model(s) and data
- Universal interface for saving and loading regardless of backend(s) or device
- Automatic handling of gradient accumulation and clipping
- Common `attrs` interface for all backend configuration parameters (with docstrings)
- Helper methods for printing synced losses, device specific print, number of model parameters
- Extra(s) - Custom torch.utils.data.distributed.Sampler: BucketedDistributedSampler which buckets data by 
  a sorted idx and then randomly samples from specific bucket(s) to prevent situations like grossly mismatched sequence 
  length leading to wasted computational overhead (ie excess padding)

## Installation

### (Required for FP16 Support) Install NVIDIA Apex

If you are planning on using mixed-precision (aka FP16), please install Apex so that `stoke` supports all FP16 methods. 
If you are not planning on using mixed precision, this step can actually be skipped (as all imports are in a try/except 
and are only conditionally imported).

Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start).

### (Optional) Underlying OpenMPI Support

**Note: MPI support is necessary if you plan to run Stoke across multiple compute nodes (e.g. 2 nodes with 4 GPUs each) 
with DDP, Horovod, or DeepSpeed backends**

Follow the instructions [here](https://www.open-mpi.org/faq/?category=building) or 
[here](https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf)

Also, refer to the Dockerfile [here](https://github.com/fidelity/stoke/blob/master/docker/stoke-gpu-mpi.Dockerfile) 

### via PyPi
```bash
pip install stoke
```

### via PyPi w/ Optional MPI Support

**Note: MPI support is necessary if you plan to run Stoke across multiple compute nodes (e.g. 2 nodes with 4 GPUs each) 
with DDP, Horovod, or DeepSpeed backends**

```bash
pip install stoke[mpi]
```

## Documentation and Examples

Full documentation can be found [here](https://fidelity.github.io/stoke/) and 
examples are [here](https://github.com/fidelity/stoke/blob/master/examples).

## Quick Start

#### Basic Definitions

Assuming some already existing common PyTorch objects (dataset: `torch.utils.data.Dataset`, model: `torch.nn.Module`, 
loss: `torch.nn.(SomeLossFunction)`):

```python
import torch

# Some existing user defined dataset using torch.utils.data.Dataset
class RandomData(torch.utils.data.Dataset):
    pass

# An existing model defined with torch.nn.Module
class BasicNN(torch.nn.Module):
    pass

# Our existing dataset from above
dataset = RandomData(...)

# Our existing model from above 
model = BasicNN(...)

# A loss function
loss = torch.nn.BCEWithLogitsLoss()
```

#### Optimizer Setup

`stoke` requires a slightly different way to define the optimizer (as it handles instantiation internally) by using
`StokeOptimizer`. Pass in the uninstantiated `torch.optim.*` class object and any **kwargs that need to be passed to the 
`__init__` call:

```python
from stoke import StokeOptimizer
from torch.optim import Adam

# Some ADAM parameters
lr = 0.001
beta1 = 0.9
beta2 = 0.98
epsilon = 1E-09

# Create the StokeOptimizer
opt = StokeOptimizer(
    optimizer=Adam,
    optimizer_kwargs={
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": epsilon
    }
)
```

#### Create Stoke Object

Now create the base `stoke` object. Pass in the model, loss(es), and `StokeOptimizer` from above as well as any
flags/choices to set different backends/functionality/extensions and any necessary configurations. As an example, 
we set the device type to GPU, use the PyTorch DDP backend for distributed multi-GPU training, toggle native PyTorch 
AMP mixed precision, add Fairscale optimizer-state-sharding (OSS), and turn on automatic gradient accumulation and 
clipping (4 steps and clip-by-norm). In addition, let's customize PyTorch DDP,  PyTorch AMP and Fairscale OSS with 
some of our own settings but leave all the others as default configurations.

```python
import os
from stoke import AMPConfig
from stoke import ClipGradNormConfig
from stoke import DDPConfig
from stoke import DistributedOptions
from stoke import FairscaleOSSConfig
from stoke import FP16Options
from stoke import Stoke

# Custom AMP configuration
# Change the initial scale factor of the loss scaler
amp_config = AMPConfig(
    init_scale=2.**14
)

# Custom DDP configuration
# Automatically swap out batch_norm layers with sync_batch_norm layers
# Notice here we have to deal with the local rank parameter that DDP needs (from env or cmd line)
ddp_config = DDPConfig(
    local_rank=os.getenv('LOCAL_RANK'),
    convert_to_sync_batch_norm=True
)

# Custom OSS configuration
# activate broadcast_fp16 -- Compress the model shards in fp16 before sharing them in between ranks
oss_config = FairscaleOSSConfig(
    broadcast_fp16=True
)

# Configure gradient clipping using the configuration object
grad_clip = ClipGradNormConfig(
    max_norm=5.0,
    norm_type=2.0
)

# Build the object with the correct options/choices (notice how DistributedOptions and FP16Options are already provided
# to make choices simple) and configurations (passed to configs as a list)
stoke_obj = Stoke(
    model=model,
    optimizer=opt,
    loss=loss,
    batch_size_per_device=32,
    gpu=True,
    fp16=FP16Options.amp.value,
    distributed=DistributedOptions.ddp.value,
    fairscale_oss=True,
    grad_accum_steps=4,
    grad_clip=grad_clip,
    configs=[amp_config, ddp_config, oss_config]
)
```

#### Build PyTorch DataLoader

Next we need to create a `torch.utils.data.DataLoader` object. Similar to the optimizer definition this has to be done
a little differently with `stoke` for it to correctly handle each of the different backends. `stoke` provides a mirrored
wrapper to the native `torch.utils.data.DataLoader` class (as the `DataLoader` method) that will return a correctly 
configured `torch.utils.data.DataLoader` object. Since we are using a distributed backend (DDP) we need to provide a 
`DistributedSampler` or similar class to the `DataLoader`. Note that the `Stoke` object that we just created has the 
properties `.rank` and `.world_size` which provide common interfaces to this information regardless of the backend!

```python
from torch.utils.data.distributed import DistributedSampler

# Create our DistributedSampler
# Note: dataset is the torch.utils.data.Dataset from the first section
sampler = DistributedSampler(
    dataset=dataset,
    num_replicas=stoke_obj.world_size,
    rank=stoke_obj.rank
)

# Call the DataLoader method on the stoke_obj to correctly create a DataLoader instance
# The DataLoader object already known the batch size from the Stoke object creation
data_loader = stoke_obj.DataLoader(
    dataset=dataset,
    collate_fn=lambda batch: dataset.collate_fn(batch), # note: this is optional depending on your dataset
    sampler=sampler,
    num_workers=4
)
```

#### Add a LR Scheduler

Stoke provides access to each of the underlying PyTorch instances/objects/classes it's managing. Any created `Stoke`
object has multiple `@property` methods that return the underlying attribute(s) such as `.optimzer`, `.loss_access`,
`.model_access`, `.step_loss`, etc. Therefore, to use a PyTorch LR Scheduler it's as simple as getting the underlying
optimizer and passing it to the LR Scheduler constructor:

```python
from torch.optim.lr_scheduler import OneCycleLR


scheduler = OneCycleLR(
  stoke_obj.optimizer, 
  max_lr=0.001, 
  pct_start = 0.9, 
  epochs=100, 
  steps_per_epoch=len(data_loader)
)
```

#### Run a Training Loop

At this point, we've successfully configured `stoke`! Since `stoke` handled wrapping/building your `torch.nn.Module` and 
`torch.utils.data.DataLoader`, device placement is handled automatically (in our example the model and data are moved
to GPUs). The following simple training loop should look fairly standard, except that the model `forward`, `loss`, 
`backward`, and `step` calls are all called on the `Stoke` object instead of each individual component (as it 
internally maintains the model, loss, and optimizer and all necessary code for all 
backends/functionality/extensions). In addition, we use one of many helper functions built into `stoke` to print the 
synced and gradient accumulated loss across all devices (an all-reduce across all devices with ReduceOp.SUM and divided
by world_size -- that is print only on rank 0 by default)

```python
epoch = 0
# Iterate until number epochs
while epoch < 100:
    # Loop through the dataset
    for x, y in data_loader:
        # Use the Stoke wrapped version(s) of model, loss, backward, and step
        # Forward
        out = stoke_obj.model(x)
        # Loss
        loss = stoke_obj.loss(out, y.to(dtype=torch.float).unsqueeze(1))
        # Detach loss and sync across devices -- only after grad accum step has been called 
        stoke_obj.print_mean_accumulated_synced_loss()
        # Backward
        stoke_obj.backward(loss)
        # stoke_obj.dump_model_grads()
        # Optimizer Step
        stoke_obj.step()
        # Scheduler Step -> Note this is the order for PyTorch 1.10, for < 1.10 the scheduler step is before the
        # optimizer step
        scheduler.step()
    epoch += 1
```

#### Save/Load

`stoke` provides a unified interface to save and load model checkpoints regardless of backend/functionality/extensions.
Simply call the `save` or `load` methods on the `Stoke` object.

```python
# Save the model w/ a dummy extra dict
path, tag = stoke_obj.save(
    path='/path/to/save/dir',
    name='my-checkpoint-name',
    extras={'foo': 'bar'}
    )

# Attempt to load a saved checkpoint -- returns the extras dictionary
extras = stoke_obj.load(
    path=path,
    tag=tag
)
```

### Launchers

See the documentation [here](https://fidelity.github.io/stoke/docs/Launchers/)

## Compatibility Matrix

Certain combinations of backends/functionality are not compatible with each other. The below table indicates which 
combinations should work together:

| Backends/Devices | CPU | GPU | PyTorch DDP | Deepspeed DDP | Horovod | Deepspeed FP16 | Native AMP | NVIDIA APEX | Deepspeed ZeRO | Fairscale |
| ---------------- | --- | --- | ----------- | ------------- | ------- | -------------- | ---------  | ----------- | -------------- | --------- |
| CPU | | | | | | | | | |
| GPU | | | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | |
| PyTorch DDP | | &#10004; | | | | | &#10004; | &#10004; | | &#10004; |
| Deepspeed DDP | | &#10004; | | | | &#10004; | | | &#10004; | |
| Horovod | | &#10004; | | | | | &#10004; | &#10004; | | |
| DeepspeedFP16 | | &#10004; | | &#10004; | | &#10004; | | | &#10004; | |
| Native AMP | | &#10004; | &#10004; | | &#10004; | | | | | &#10004; |
| NVIDIA APEX | | &#10004; | &#10004; | | &#10004; | | | | | |
| Deepspeed ZeRO | | &#10004; | | &#10004; | | &#10004; | | | | |
| Fairscale | | &#10004; | &#10004; | | | | &#10004; | | | |


___
`stoke` is developed and maintained by the **Artificial Intelligence Center of Excellence at Fidelity Investments**.