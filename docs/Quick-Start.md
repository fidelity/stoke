# Quick Start

This is a quick and dirty guide to getting up and running with `stoke`. Read the 
documentation [here](https://fidelity.github.io/stoke/) for full details and refer to the 
examples [here](https://github.com/fidelity/stoke/blob/master/examples).

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

Now create the base `stoke` object. Pass in the model, and loss(es), and `StokeOptimizer` from above, 
flags/choices to set different backends/functionality/extensions, and any necessary configurations. As an example, 
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
    fp16=FP16Options.amp,
    distributed=DistributedOptions.ddp,
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
DistributedSampler or similar class to the DataLoader. Note that the `Stoke` object that we just created has the 
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
data_loader = stoke_obj.DataLoader(
    dataset=dataset,
    collate_fn=lambda batch: dataset.collate_fn(batch),
    batch_size=32,
    sampler=sampler,
    num_workers=4
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
by world_size -- printed only on rank 0 by default)

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
        # Step
        stoke_obj.step()
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