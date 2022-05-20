# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 training script demonstrating a few different stoke options

Based loosely on: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from configs import *
from model import resnet152
from spock.builder import ConfigArgBuilder
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data.distributed import DistributedSampler

from stoke import DeepspeedConfig, DeepspeedZeROConfig, Stoke, StokeOptimizer


def train(train_dataloader, cifar_stoke: Stoke, epoch: int):
    cifar_stoke.print_on_devices(f"Starting Epoch {epoch + 1}")
    cifar_stoke.model_access.train()
    for idx, (x, y) in enumerate(train_dataloader):
        # Call the model through the stoke object interface
        outputs = cifar_stoke.model(x)
        # Call the loss through the stoke object interface
        loss = cifar_stoke.loss(outputs, y)
        # Print some loss info
        cifar_stoke.print_ema_loss(prepend_msg=f"Step {idx+1} -- EMA Loss")
        # Call backward through the stoke object interface
        cifar_stoke.backward(loss=loss)
        # Call step through the stoke object interface
        cifar_stoke.step()
    return epoch + 1


def predict(test_dataloader, cifar_stoke: Stoke):
    # Switch to eval mode
    cifar_stoke.model_access.eval()
    total_y = 0
    total_correct = 0
    # Wrap with no grads context just to be safe
    with torch.no_grad():
        for x, y in test_dataloader:
            outputs = cifar_stoke.model(x)
            _, preds = torch.max(outputs.detach(), dim=1)
            total_y += y.size(0)
            total_correct += torch.sum(preds == y).item()
    cifar_stoke.print_on_devices(
        msg=f"Current Test Accuracy: {((total_correct/total_y) * 100):.3f}"
    )


def main():
    # Use spock to grab all the configs
    configs = ConfigArgBuilder(
        DataConfig, OSSConfig, RunConfig, SDDPConfig, SGDConfig, ZeROConfig
    ).generate()
    # Create the resnet-152 model
    model = resnet152()
    # Define the loss function
    loss = CrossEntropyLoss()
    # Make the StokeOptimizer object
    optimizer = StokeOptimizer(
        optimizer=SGD,
        optimizer_kwargs={
            "lr": configs.SGDConfig.lr,
            "momentum": configs.SGDConfig.momentum,
            "weight_decay": configs.SGDConfig.weight_decay,
        },
    )
    # Handle some extra config objects so we can easily switch between different stoke options from the config yaml(s)
    extra_configs = [
        DeepspeedConfig(
            zero_optimization=DeepspeedZeROConfig(
                stage=configs.ZeROConfig.zero,
                contiguous_gradients=configs.ZeROConfig.contiguous_gradients,
                overlap_comm=configs.ZeROConfig.overlap_comm,
            ),
            dump_state=True,
        )
    ]
    # Build the base stoke object
    cifar_stoke = Stoke(
        model=model,
        optimizer=optimizer,
        loss=loss,
        batch_size_per_device=configs.DataConfig.batch_size,
        gpu=configs.RunConfig.gpu,
        fp16=configs.RunConfig.fp16,
        distributed=configs.RunConfig.distributed,
        fairscale_oss=configs.RunConfig.oss,
        fairscale_sddp=configs.RunConfig.sddp,
        configs=extra_configs,
        grad_accum_steps=configs.RunConfig.grad_accum,
        verbose=True,
    )
    # Set up a transform pipeline for CIFAR10 training data -- do some simple augmentation for illustration
    transform_train = tv_transforms.Compose(
        [
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=configs.DataConfig.normalize_mean,
                std=configs.DataConfig.normalize_std,
            ),
        ]
    )
    # Set up a transform pipeline for CIFAR10 test data
    transform_test = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=configs.DataConfig.normalize_mean,
                std=configs.DataConfig.normalize_std,
            ),
        ]
    )
    # Get CIFAR10 training data from torchvision
    training_dataset = tv_datasets.CIFAR10(
        root=configs.DataConfig.root_dir,
        train=True,
        download=True,
        transform=transform_train,
    )
    # Get CIFAR10 test data from torchvision
    test_dataset = tv_datasets.CIFAR10(
        root=configs.DataConfig.root_dir,
        train=False,
        download=True,
        transform=transform_test,
    )
    # If distributed then roll a sampler else None
    train_sampler = (
        DistributedSampler(
            dataset=training_dataset,
            num_replicas=cifar_stoke.world_size,
            rank=cifar_stoke.rank,
        )
        if configs.RunConfig.distributed is not None
        else None
    )
    # Construct the DataLoader
    train_loader = cifar_stoke.DataLoader(
        dataset=training_dataset,
        sampler=train_sampler,
        num_workers=configs.DataConfig.n_workers
        if configs.DataConfig.n_workers is not None
        else 0,
    )
    # If distributed then roll a sampler else None
    test_sampler = (
        DistributedSampler(
            dataset=test_dataset,
            num_replicas=cifar_stoke.world_size,
            rank=cifar_stoke.rank,
        )
        if configs.RunConfig.distributed is not None
        else None
    )
    test_loader = cifar_stoke.DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        num_workers=configs.DataConfig.n_workers
        if configs.DataConfig.n_workers is not None
        else 0,
    )
    # Initial overall acc which should be ~10% given the 10 CIFAR10 classes
    predict(test_dataloader=test_loader, cifar_stoke=cifar_stoke)
    n_epochs = 0
    while n_epochs < configs.RunConfig.num_epoch:
        n_epochs = train(
            train_dataloader=train_loader, cifar_stoke=cifar_stoke, epoch=n_epochs
        )
        # Reset the ema stats after each epoch
        cifar_stoke.reset_ema()
        # Check test loss every epoch
        predict(test_dataloader=test_loader, cifar_stoke=cifar_stoke)


if __name__ == "__main__":
    main()
