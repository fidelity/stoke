# Example CIFAR10 Training with Stoke

[Home](https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 
10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches 
contain the remaining images in random order, but some training batches may contain more images from one class than 
another. Between them, the training batches contain exactly 5000 images from each class.

This example uses a simple ResNet-152 model ([arxiv](https://arxiv.org/pdf/1512.03385.pdf)) and trains for a user
defined number of epochs.


### Install Requirements
pip install the necessary requirements

```shell
pip install -r CIFAR10_REQUIREMENTS.tx
```

### Run
For instance, a simple single GPU run would be something like:

```shell
python stoke/examples/cifar10/train.py -c stoke/examples/cifar10/config/single-gpu.yaml
```

or using the torch distributed launcher (with 2 GPUs):

```shell
python -m torch.distributed.launch --nproc_per_node=2 --use_env stoke/examples/cifar10/train.py \
-c stoke/examples/cifar10/config/ddp-gpu.yaml
```

### Included Configuration Examples
- CPU: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/base.yaml)
- Single GPU: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/single-gpu.yaml)
- DDP Multi-GPU: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/ddp-gpu.yaml)
- DDP Multi-GPU + AMP FP16: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/ddp-fp16-amp-gpu.yaml)
- DDP Multi-GPU + APEX O1 FP16: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/ddp-fp16-apex01-gpu.yaml)
- DDP Multi-GPU + AMP FP16 + Fairscale OSS + Fairscale SDDP: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/ddp-fp16-amp-oss-sddp.yaml)
- Deepspeed Multi-GPU w/ Deepspeed FP16 + ZeRO Stage 2: [config](https://github.com/fidelity/stoke/blob/master/examples/cifar10/config/deepspeed-fp16-zero-stage-2.yaml)
