# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles any data (e.g. loader, sampler, etc.) related classes"""

import itertools
from math import ceil
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import horovod.torch as hvd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset
from torch.utils.data.distributed import Sampler

from stoke.status import DistributedOptions, FP16Options
from stoke.utils import T_co, _collate_fn_t, _worker_init_fn_t


class StokeDataLoader(DL):
    """Provides a shim interface to torch.utils.data.DataLoader with mapped kwargs

    Attributes
    ----------
    _gpu: bool
    _fp16: Optional[FP16Options]

    See Also
    --------
    torch.utils.data.DataLoader: base DataLoader class that this inherits from (check for all attributes)

    """

    def __init__(
        self, dataset: Dataset[T_co], gpu: bool, fp16: Optional[FP16Options], **kwargs
    ):
        """Maps to torch.utils.data.DataLoader __init__

        Shim is necessary to automatically handle device placement since the gpu/fp16 flags can't be
        determined until the StokeStatus object is available which is post init. This could be disconnected from
        this class but it would require the user to forward on device or fp16 configs which breaks the
        paradigm that the flags only need to be set and never handled

        Parameters
        ----------
        dataset: Dataset
            dataset from which to load the data
        gpu: bool
            flag to signify the device should be gpu
        fp16: Optional[FP16Options], default: None
            Choice of mixed-precision backend
        **kwargs

        Returns
        -------
        StokeDataLoader
            wrapped torch.utils.data.DataLoader object

        """
        # Call super init for the actual torch DataLoader - using **kwargs
        super(StokeDataLoader, self).__init__(dataset, **kwargs)
        self._gpu = gpu
        self._fp16 = fp16

    def __iter__(self):
        """Underlying iter of the DataLoader that yields samples

        Wrap the base __iter__ with a call to place on the device if flagged

        Yields
        ------
        Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]]
            data placed on the correct device

        """
        # Iterate using the base class iter but override the yield by pushing to device prior if gpu flag is true
        for val in super().__iter__():
            yield val if not self._gpu else self._place_data_on_gpu(val)

    def _place_data_on_gpu(
        self,
        data: Union[
            torch.Tensor,
            List[torch.Tensor],
            Tuple[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
    ):
        """Determine data structure and then place on the correct device (cast in the context of deepspeed FP16 as it
        wants half dtype as input)

        Parameters
        ----------
        data: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]]
            current data coming from the underlying __iter__

        Returns
        -------
        data: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]]
            data moved to the correct device

        """
        if isinstance(data, torch.Tensor):
            # Move to the correct cuda device w/ the correct type -- deepspeed FP16 requires a cast to half if fp16
            if self._fp16 == "deepspeed":
                return data.to(device="cuda", dtype=torch.half)
            else:
                return data.to(device="cuda", dtype=data.dtype)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._place_data_on_gpu(data=val) for val in data)
        elif isinstance(data, dict):
            return {k: self._place_data_on_gpu(v) for k, v in data.items()}
        elif ~(hasattr(data, "to")):
            return data
        else:
            raise TypeError(
                f"Stoke -- Unsupported data type passed to _place_data_on_gpu "
                f"(torch.Tensor, tuple, list, dict), currently {type(data)}"
            )


class BucketedDistributedSampler(Sampler[T_co]):
    """Sampler that buckets samples by sorted_idx and then randomly samples from a specific bucket to prevent excess
    padding leading to wasted computation

    Borrowing heavily from the base DistributedSampler
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler

    Attributes
    ----------
    num_replicas: int, default: None
        number of replicas
    rank: int, default: None
        current device rank
    epoch: int
        current training epoch
    drop_last: bool, default: False
        whether to drop last set of samples that don't fit into a batch
    shuffle: bool, default: True
        flag to shuffle dataset
    seed: int, default: 0
        seed to use for generators
    buckets: int
        number of buckets to break the dataset into
    sorted_n_samples: list
        sorted list of samples by the characteristic to bucket by (e.g. seq len)
    batch_size: int
        batch size that will be used (needed to make sure slices are correct)
    allow_bucket_overlap: bool, default: False
        allow for the residual samples (those that are not divisible by batch and num_replicas) to be assembled into
        an un-bucketed batch
    slice_size: int
        computed from batch size and number of replicas
    num_samples_per_bucket: int
        computed value that represents the number of samples in a single bucket
    num_slices_per_bucket: int
        computed value that represents the number of slices available in a bucket
    bucket_idx: list
        computed value that make a contiguous list of indices in each bucket
    rounded_num_samples_per_bucket: int
        computed value post round for number of samples in a single bucket
    rounded_num_samples_per_replica: int
        computed value post round for number of slices available in a bucket

    """

    def __init__(
        self,
        dataset: Dataset,
        buckets: int,
        batch_size: int,
        sorted_idx: List,
        backend: DistributedOptions,
        allow_bucket_overlap: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        info_rank: int = 0,
    ) -> None:
        """Init for BucketedDistributedSampler

        Parameters
        ----------
        dataset: Dataset
            dataset from which to load the data.
        buckets: int
            number of buckets to break the dataset into
        batch_size: int
            batch size that will be used (needed to make sure slices are correct)
        sorted_idx: list
            sorted list of samples by the characteristic to bucket by (e.g. seq le
        backend: DistributedOptions
            which backend is being used (as rank, world size, etc. need to be used)
        allow_bucket_overlap: bool, default: False
            allow for the residual samples (those that are not divisible by batch and num_replicas) to be assembled into
            an un-bucketed batch
        num_replicas: int, default: None
            number of replicas
        rank: int, default: None
            current device rank
        shuffle: bool, default: True
            flag to shuffle dataset
        seed: int, default: 0
            seed to use for generators
        drop_last: bool, default: False
            whether to drop last set of samples that don't fit into a
        info_rank: int, default: 0
            which device to print information on

        """
        # If the backend isnt DDP there needs to be an additional import
        num_replicas, rank = self._conditional_distributed(
            backend=backend, num_replicas=num_replicas, rank=rank
        )
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.buckets = buckets
        self.sorted_n_samples = sorted_idx
        # Batch size is needed here so a contiguous iter of buckets can be formed
        self.batch_size = batch_size
        # This is a flag to batch up the dropped samples (that would be 'wasted') if drop_last is flagged
        self.allow_bucket_overlap = allow_bucket_overlap
        # Calculate the size of each slice that will be indexed across the replicas
        self.slice_size = self.batch_size * self.num_replicas
        # Calculate the size of the buckets (rounded or not based on drop last)
        self.num_samples_per_bucket = self._get_size(
            len(dataset), self.buckets, self.drop_last
        )
        # Calculate the number of slices per bucket
        self.num_slices_per_bucket = self._get_size(
            self.num_samples_per_bucket, self.slice_size, self.drop_last
        )
        if self.num_samples_per_bucket < self.slice_size:
            raise ValueError(
                f"Stoke -- Resulting number of slices (batch * replicas) per bucket "
                f"({self.num_samples_per_bucket}) is less than the batch size "
                f"({self.batch_size})"
            )
        if self.num_slices_per_bucket < 2:
            raise ValueError(
                f"Stoke -- Number of slices per bucket {self.num_slices_per_bucket} is less than 2 "
                f"which is not recommended"
            )
        if self.num_samples_per_bucket < 100:
            raise ValueError(
                f"Stoke -- Number of samples per bucket {self.num_samples_per_bucket} is less than 100 "
                f"which is not recommended as this might lead to dropping of excessive data"
            )
        # Split into buckets and turn into lists
        self.bucket_idx = [
            list(val) for val in np.array_split(self.sorted_n_samples, self.buckets)
        ]
        # Calculate the post rounded numbers
        self.rounded_num_samples_per_bucket = (
            self.slice_size * self.num_slices_per_bucket
        )
        self.rounded_num_samples_per_replica = (
            self.num_slices_per_bucket * self.batch_size * self.buckets
        )
        # Add the bucket overlap samples
        if self.allow_bucket_overlap:
            self.rounded_num_samples_per_replica += (
                (len(dataset) - (self.rounded_num_samples_per_bucket * self.buckets))
                // self.slice_size
            ) * self.batch_size
        if self.rank == info_rank:
            print(
                f"Stoke -- BucketedDistributedSampler -- # Samples Per Bucket: "
                f"{self.rounded_num_samples_per_bucket}, # of Samples Per Replica: "
                f"{self.rounded_num_samples_per_replica}"
            )

    def _conditional_distributed(
        self,
        backend: DistributedOptions,
        num_replicas: Optional[int],
        rank: Optional[int],
    ):
        """

        Parameters
        ----------
        backend: DistributedOptions
            which backend is being used
        num_replicas: int, default: None
            total number of replicas
        rank: int, default: None
            current device rank

        Returns
        -------
        Tuple[int, int]
            num_replicas, rank
        """
        return self._check_backend(backend, num_replicas, rank)

    def _get_backend_functions(self, backend: DistributedOptions):
        """Gets backend functions if needed

        Parameters
        ----------
        backend: DistributedOptions
            which backend is being used

        Returns
        -------
        Tuple[bool, int, int]
            is_init, num_replicas, rank

        """
        if backend.value == "ddp" or backend.value == "deepspeed":
            return (
                torch.distributed.is_initialized,
                torch.distributed.get_world_size,
                torch.distributed.get_rank,
            )
        else:
            return hvd.is_initialized, hvd.size, hvd.rank

    def _check_backend(
        self,
        backend: DistributedOptions,
        num_replicas: Optional[int],
        rank: Optional[int],
    ):
        """Checks the backend for correct device info

        Parameters
        ----------
        backend: DistributedOptions
            which backend is being used
        num_replicas: int, default: None
            total number of replicas
        rank: int, default: None
            current device rank

        Returns
        -------
        Tuple[int, int]
            num_replicas, rank

        """
        if num_replicas is None or rank is None:
            is_avail, get_world_size, get_rank = self._get_backend_functions(
                backend=backend
            )
        if num_replicas is None:
            if not is_avail():
                raise RuntimeError(
                    "Requires distributed package (torch.dist or hvd) to be available"
                )
            num_replicas = get_world_size()
        if rank is None:
            if not is_avail():
                raise RuntimeError(
                    "Requires distributed package (torch.dist or hvd) to be available"
                )
            rank = get_rank()
        return num_replicas, rank

    @staticmethod
    def _get_size(data_len: int, split_var: int, drop_last: bool = False):
        """Gets the size of a split

        Parameters
        ----------
        data_len: int
            current dataset length
        split_var: int
            how many to split into
        drop_last: bool, default: False
            drop last hanging samples if not batch_size

        Returns
        -------
        num_samples: int

        """
        if drop_last:
            num_samples = data_len // split_var
        else:
            num_samples = ceil(data_len / split_var)
        return num_samples

    def __iter__(self) -> Iterator[T_co]:
        """Handles assembling the batches from a bucketed perspective

        Shuffle bucket order->Pad if necessary->Slice across replicas->Possibly batch up residuals->shuffle bucketed
        batches->Unroll into list->Make iter

        Returns
        -------
        Iterator[T_co]

        """
        # Shuffle the bucketed idx
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Permute each bucket
            indices = [
                [val[idx] for idx in torch.randperm(len(val), generator=g).tolist()]
                for val in self.bucket_idx
            ]
        else:
            indices = self.bucket_idx
        # Iterate over the buckets
        for idx, val in enumerate(indices):
            # If this is true we need to handle padding
            if (self.num_slices_per_bucket * self.slice_size) > len(val):
                split_val = self._handle_padding(val)
                indices[idx] = list(itertools.chain(*split_val))
                assert len(indices[idx]) == self.rounded_num_samples_per_bucket
        # Now slice across replicas
        final_indices = []
        for val in indices:
            for idx in range(self.num_slices_per_bucket):
                replica_slice = val[
                    (idx * self.slice_size) : ((idx + 1) * self.slice_size)
                ][self.rank : self.slice_size : self.num_replicas]
                final_indices.append(replica_slice)
        # If bucket overlap is allowed then we just batch up the residual indices
        if self.drop_last and self.allow_bucket_overlap:
            residual_idx = list(
                itertools.chain(
                    *[val[self.rounded_num_samples_per_bucket :] for val in indices]
                )
            )
            if len(residual_idx) > self.slice_size:
                # Cut by slices then by replicas
                residual_idx = [
                    residual_idx[
                        (idx * self.slice_size) : ((idx + 1) * self.slice_size)
                    ][self.rank : self.slice_size : self.num_replicas]
                    for idx in range(len(residual_idx) // self.slice_size)
                ]
                # Append to the final indices
                final_indices.extend(residual_idx)
        # Shuffle the bucketed batches
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Permute the bucket order
            final_indices = [
                final_indices[val]
                for val in torch.randperm(len(final_indices), generator=g)
            ]
        # Unroll into a single list
        final_indices = list(itertools.chain(*final_indices))
        assert len(final_indices) == self.rounded_num_samples_per_replica
        return iter(final_indices)

    def _handle_padding(self, idx_list: List):
        """Handles padding out if a batch is short

        Parameters
        ----------
        idx_list: List
            list of indices

        Returns
        -------
        split_val: List
            list with correctly padded sizes

        """
        split_val = []
        for idx in range(self.num_slices_per_bucket):
            if idx == (self.num_slices_per_bucket - 1):
                # Get the short batch
                short_batch = idx_list[(idx * self.slice_size) :]
                # Short batch replica slice sizes
                short_len = [
                    self.batch_size - len(list(val))
                    for val in np.array_split(short_batch, self.num_replicas)
                ]
                # Pop the necessary values from the entire bucket
                pad_values = [
                    idx_list[s_idx : (self.num_replicas * s_len) : self.num_replicas]
                    for s_idx, s_len in enumerate(short_len)
                ]
                # If not a consistent list then we need to reorder so that the step size alignment slicing
                # of the replicas works
                if len(set(short_len)) != 1:
                    # here we need to find the first larger idx and reorder
                    first_idx = short_len.index(max(set(short_len)))
                    # Reorder
                    pad_values = pad_values[first_idx:] + pad_values[0:first_idx]
                extended_batch = short_batch + [
                    pad
                    for pad in list(
                        itertools.chain(*itertools.zip_longest(*pad_values))
                    )
                    if pad is not None
                ]
                split_val.append(extended_batch)
            else:
                split_val.append(
                    idx_list[(idx * self.slice_size) : ((idx + 1) * self.slice_size)]
                )
        return split_val

    def __len__(self) -> int:
        return self.rounded_num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Parameters
        ----------
        epoch: int
            Epoch number

        """
        self.epoch = epoch
