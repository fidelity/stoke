# -*- coding: utf-8 -*-

# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

"""Handles all config objects"""

from enum import Enum
from typing import Dict, Optional, Type

import attr
import torch

try:
    from typing import TypedDict
except ImportError:
    from mypy_extensions import TypedDict


class HorovodOps(Enum):
    """Horovod ops options"""

    Average = "Average"
    Sum = "Sum"
    Adasum = "Adasum"


class OffloadDevice(Enum):
    """Offload device options"""

    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class BackendOptions(Enum):
    """Communication backend options"""

    nccl = "nccl"
    mpi = " mpi"
    gloo = "gloo"


@attr.s(auto_attribs=True)
class AMPConfig:
    """PyTorch AMP configuration class

    Attributes
    ----------
    backoff_factor : float, default: 0.5
        Factor by which the scale is multiplied during update if inf/NaN gradients occur in an iteration
    growth_factor : float, default: 2.0
        Factor by which the scale is multiplied during update if no inf/NaN gradients occur for growth_interval consecutive iterations.
    growth_interval : int, default: 2000
        Number of consecutive iterations without inf/NaN gradients that must occur for the scale to be multiplied by
        growth_factor
    init_scale : float, default: 2.**16
        Initial scale factor

    """

    backoff_factor: float = 0.5
    growth_factor: float = 2.0
    growth_interval: int = 2000
    init_scale: float = 2.0 ** 16


@attr.s(auto_attribs=True)
class ApexConfig:
    """Nvidia APEX configuration class

    Attributes
    ----------
    cast_model_outputs: Optional[torch.dtype], default: None
        Option to ensure that the outputs of your model(s) are always cast to a particular type regardless of opt_level
    convert_to_sync_batch_norm: bool, default: False
        Automatically convert all batch norm calls to apex.parallel.SyncBatchNorm calls
        https://nvidia.github.io/apex/parallel.html#apex.parallel.SyncBatchNorm
    max_loss_scale: float, default: 2.**24
        Sets a ceiling for the loss scale values that can be chosen by dynamic loss scaling
    min_loss_scale: Optional[float], default: None
        Sets a floor for the loss scale values that can be chosen by dynamic loss scaling. The default value of None
        means that no floor is imposed
    scaler_per_loss: bool, default: False
        Option to impose a scaler for each loss instead of a global scaler
    verbosity: int, default: 0
        Set to 0 to suppress Amp-related output

    """

    cast_model_outputs: Optional[torch.dtype] = None
    convert_to_sync_batch_norm: bool = False
    max_loss_scale: float = 2.0 ** 24
    min_loss_scale: Optional[float] = None
    scaler_per_loss: bool = False
    verbosity: int = 0


@attr.s(auto_attribs=True)
class ClipGradConfig:
    """Gradient clipping by value configuration class

    Attributes
    ----------
    clip_value: float
        maximum allowed absolute value of the gradients [-clip_value, clip_value]

    """

    clip_value: float


@attr.s(auto_attribs=True)
class ClipGradNormConfig:
    """Gradient clipping by p-norm configuration class

    Attributes
    ----------
    max_norm: float
        max norm of the gradients
    norm_type: float
        type of the used p-norm

    """

    max_norm: float
    norm_type: float


@attr.s(auto_attribs=True)
class DDPConfig:
    """PyTorch DistributedDataParallel configuration class

    Attributes
    ----------
    local_rank: Optional[int]
        Current local rank of the device (provided here, as LOCAL_RANK env var, or parsed from --local_arg)
    convert_to_sync_batch_norm: bool, default: False
        Automatically convert all batch norm calls to torch.nn.SyncBatchNorm calls
        https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    backend: BackendOptions, default: 'nccl'
        Which communication backend to use
    broadcast_buffers: bool, default: True
        Flag that enables syncing (broadcasting) buffers of the module at beginning of the forward function
    bucket_cap_mb: int, default: 25
        DistributedDataParallel will bucket parameters into multiple buckets so that gradient reduction of each bucket
        can potentially overlap with backward computation. bucket_cap_mb controls the bucket size in MegaBytes (MB)
    find_unused_parameters: bool, default: False
        Traverse the autograd graph from all tensors contained in the return value of the wrapped module’s forward
        function. Parameters that don’t receive gradients as part of this graph are preemptively marked as being ready
        to be reduced. Note that all forward outputs that are derived from module parameters must participate in
        calculating loss and later the gradient computation. If they don’t, this wrapper will hang waiting for autograd
        to produce gradients for those parameters. Any outputs derived from module parameters that are otherwise unused
        can be detached from the autograd graph using torch.Tensor.detach
    gradient_as_bucket_view: bool, default: False
        When set to True, gradients will be views pointing to different offsets of allreduce communication
        buckets. This can reduce peak memory usage, where the saved memory size will be equal to the total gradients
        size. Moreover, it avoids the overhead of copying between gradients and allreduce communication buckets. When
        gradients are views, detach_() cannot be called on the gradients. If hitting such errors, please fix it by
        referring to the zero_grad() function in torch/optim/optimizer.py as a solution.
    init_method: str, default: 'env://'
        URL specifying how to initialize the process group

    """

    local_rank: Optional[int]
    convert_to_sync_batch_norm: bool = False
    backend: BackendOptions = "nccl"
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = False
    init_method: str = "env://"


@attr.s(auto_attribs=True)
class DeepspeedAIOConfig:
    """Deepspeed asynchronous I/O configuration class

    Attributes
    ----------
    block_size: int, default: 1048576
        I/O block size in bytes
    ignore_unused_parameters: bool, default: True
        Unused parameters in modules may be unexpected in static networks, but could be normal in dynamic networks.
        This controls whether or not training should terminate with an error message when unused parameters are
        detected.
    overlap_events: bool, default: True
        Submit requests to storage device in an overlapped fashion without waiting for completion of earlier requests.
    queue_depth: int, default: 8
        I/O queue depth
    single_submit: bool, default: False
        Submit requests to storage device as multiple individual requests as opposed to one block of requests.
    thread_count: int, default: 1
        Intra-request parallelism for each read/write submitted by a user thread.

    """

    block_size: int = 1048576
    ignore_unused_parameters: bool = True
    overlap_events: bool = True
    queue_depth: int = 8
    single_submit: bool = False
    thread_count: int = 1


@attr.s(auto_attribs=True)
class DeepspeedActivationCheckpointingConfig:
    """Deepspeed activation checkpointing configuration class

    Attributes
    ----------
    contiguous_memory_optimization: bool, default: False
        Copies partitioned activations so that they are contiguous in memory
    cpu_checkpointing: bool, default: False
        Offloads partitioned activations to CPU if partition_activations is enabled
    number_checkpoints: Optional[int], default: None
        Total number of activation checkpoints used to allocate memory buffer for contiguous_memoty_optimization
    partition_activations: bool, default: False
        Enables partition activation when used with model parallelism
    profile: bool, default: False
        Logs the forward and backward time for each checkpoint function
    synchronize_checkpoint_boundary: bool, default: False
        Inserts torch.cuda.synchronize() at each checkpoint boundary

    """

    contiguous_memory_optimization: bool = False
    cpu_checkpointing: bool = False
    number_checkpoints: Optional[int] = None
    partition_activations: bool = False
    profile: bool = False
    synchronize_checkpoint_boundary: bool = False


@attr.s(auto_attribs=True)
class DeepspeedFlopsConfig:
    """Deepspeed flops profiler configuration class

    Attributes
    ----------
    detailed: bool, default: True
        Whether to print the detailed model profile
    module_depth: int, default: -1
        The depth of the model at which to print the aggregated module information. When set to -1, it prints
        information from the top module to the innermost modules (the maximum depth).
    output_file: Optional[str], default: None
        Path to the output file. If None, the profiler prints to stdout
    profile_step: int, default: 1
        The global training step at which to profile.
    top_modules: int, default: 1
        Limits the aggregated profile output to the number of top modules specified.

    Notes
    -----
    Warm up steps are needed for accurate time measurement

    """

    detailed: bool = True
    module_depth: int = -1
    output_file: Optional[str] = None
    profile_step: int = 1
    top_modules: int = 1


@attr.s(auto_attribs=True)
class DeepspeedFP16Config:
    """Deepspeed FP16 configuration class

    Attributes
    ----------
    hysteresis: int, default: 2
        represents the delay shift in dynamic loss scaling
    initial_scale_power: int, default: 32
        power of the initial dynamic loss scale value. The actual loss scale is computed as 2 ** initial_scale_power
    loss_scale: float, default: 0.0
        loss scaling value for FP16 training (0.0 --> dynamic scaling)
    loss_scale_window: int, default: 1000
        the window over which to raise/lower the dynamic loss scale value
    min_loss_scale: int, default: 1000
        minimum dynamic loss scale value

    """

    hysteresis: int = 2
    initial_scale_power: int = 32
    loss_scale: float = 0.0
    loss_scale_window: int = 1000
    min_loss_scale: int = 1000


@attr.s(auto_attribs=True)
class DeepspeedOffloadOptimizerConfig:
    """Deepspeed optimizer offloading configuration class

    Attributes
    ----------
    buffer_count: int, default: 4
        Number of buffers in buffer pool for optimizer state offloading to NVMe. This should be at least the number
        of states maintained per parameter by the optimizer. For example, Adam optimizer has 4 states (parameter,
        gradient, momentum, and variance).
    device: OffloadDevice, default: 'cpu'
        Device memory to offload optimizer state
    fast_init: bool, default: False
        Enable fast optimizer initialization when offloading to NVMe
    nvme_path: str, default: '/local_nvme'
        Filesystem path for NVMe device for optimizer state offloading
    pin_memory: bool, default: False
        Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead.
    pipeline: bool, default: False
        pipeline activated (will default to True if either pipeline_read or pipeline_write is set
    pipeline_read: bool, default: False
        activate pipeline read (deepspeed has limited docs for what this does)
    pipeline_write: bool, default: False
        activate pipeline write(deepspeed has limited docs for what this does)

    """

    buffer_count: int = 4
    device: OffloadDevice = "cpu"
    fast_init: bool = False
    nvme_path: str = "/local_nvme"
    pin_memory: bool = False
    pipeline: bool = False
    pipeline_read: bool = False
    pipeline_write: bool = False


@attr.s(auto_attribs=True)
class DeepspeedOffloadParamConfig:
    """Deepspeed parameter offloading configuration class

    Attributes
    ----------
    buffer_count: int, default: 5
        Number of buffers in buffer pool for parameter offloading to NVMe
    buffer_size: int, default: int(1E8)
        Size of buffers in buffer pool for parameter offloading to NVMe
    device: OffloadDevice, default: 'cpu'
        Device memory to offload model parameters
    max_in_cpu: int, default: int(1E9)
        Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.
    nvme_path: str, default: '/local_nvme'
        Filesystem path for NVMe device for parameter offloading
    pin_memory: bool, default: False
        Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead.

    """

    buffer_count: int = 5
    buffer_size: int = int(1e8)
    device: OffloadDevice = "cpu"
    max_in_cpu: int = int(1e9)
    nvme_path: str = "/local_nvme"
    pin_memory: bool = False


@attr.s(auto_attribs=True)
class DeepspeedPLDConfig:
    """
    Attributes
    ----------
    theta: float, default: 1.0
        Hyper-parameter that controls the trade-off between training time and robustness. The lower the theta value,
        the faster the training speed
    gamma: float, default: 0.001
        Hyper-parameter that controls how fast the drop ratio increases

    """

    theta: float = 1.0
    gamma: float = 0.001


@attr.s(auto_attribs=True)
class DeepspeedTensorboardConfig:
    """Deepspeed Tensorboard configuration class

    Attributes
    ----------
    output_path: str, default: ''
        Tensorboard output path
    job_name: str, default: 'DeepSpeedJobName'
        Tensorboard job name

    """

    output_path: str = ""
    job_name: str = "DeepSpeedJobName"


@attr.s(auto_attribs=True)
class DeepspeedZeROConfig:
    """Deepspeed ZeRO configuration class

    Attributes
    ----------
    allgather_bucket_size: int, default: int(5E8)
        Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes
    allgather_partitions: bool, default: True
        Chooses between allgather collective or a series of broadcast collectives to gather updated parameters
        from all the GPUs at the end of each step
    contiguous_gradients: bool, default: False
        Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward
        pass. Only useful when running very large models.
    ignore_unused_parameters: bool, default: True
        Now just used in stage2 complete_grad_norm_calculation_for_cpu_offload
        Enable this option to avoid -- https://github.com/microsoft/DeepSpeed/issues/707
    legacy_stage1: bool, default: False
        Use deepspeed < v0.3.17 zero stage 1, kept for backwards compatability reasons
    offload_optimizer: Optional[DeepspeedOffloadOptimizerConfig], default: None
        Enable offloading of optimizer state to CPU or NVMe, and optimizer computation to CPU. This frees up GPU
        memory for larger models or batch sizes. Valid only with stage 3
    offload_param: Optional[DeepspeedOffloadParamConfig], default: None
        Enable offloading of model parameters to CPU or NVMe. This frees up GPU memory for larger models or batch
        sizes. Valid only with stage 3.
    overlap_comm: bool, default: False
        Attempts to overlap the reduction of the gradients with backward computation
    reduce_bucket_size: int, default: int(5E8)
        Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large
        model sizes
    reduce_scatter: bool, default: True
        Uses reduce or reduce scatter instead of allreduce to average gradients
    stage: int, default: 0
        Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer state
        partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning,
        respectively
    stage3_max_live_parameters: int, default: int(1E9)
        The maximum number of parameters resident per GPU before releasing. Smaller values use less memory, but
        perform more communication.
    stage3_max_reuse_distance: int, default: int(1E9)
        Do not release a parameter if it will be reused within this threshold of parameters. Smaller values use less
        memory, but perform more communication.
    stage3_prefetch_bucket_size: int, default: int(5E8)
        The size of the fixed buffer for prefetching parameters. Smaller values use less memory, but can increase
        stalls due to communication.
    stage3_param_persistence_threshold: int, default: int(1E6)
        Do not partition parameters smaller than this threshold. Smaller values use less memory, but can greatly
        increase communication (especially latency-bound messages).
    stage3_gather_fp16_weights_on_model_save: bool, default: False
        Consolidate the weights before saving the model by save_fp16_model(). Since the weights are partitioned
        across GPUs, they aren’t part of state_dict, so this function automatically gather the weights when this
        option is enabled and then saves the fp16 model weights.
    sub_group_size: int, default: int(1E12)
        sub_group_size controls the granularity in which parameters are updated during optimizer steps. Parameters are
        grouped into buckets of sub_group_size and each buckets is updated one at a time.

    """

    allgather_bucket_size: int = int(5e8)
    allgather_partitions: bool = True
    contiguous_gradients: bool = False
    ignore_unused_parameters: bool = True
    legacy_stage1: bool = False
    offload_optimizer: Optional[DeepspeedOffloadOptimizerConfig] = None
    offload_param: Optional[DeepspeedOffloadParamConfig] = None
    overlap_comm: bool = False
    reduce_bucket_size: int = int(5e8)
    reduce_scatter: bool = True
    stage: int = 0
    stage3_max_live_parameters: int = int(1e9)
    stage3_max_reuse_distance: int = int(1e9)
    stage3_prefetch_bucket_size: int = int(5e8)
    stage3_param_persistence_threshold: int = int(1e6)
    stage3_gather_fp16_weights_on_model_save: bool = False
    sub_group_size: int = int(1e12)


@attr.s(auto_attribs=True)
class DeepspeedConfig:
    """Deepspeed configuration class

    Composed of other configuration classes related to specific functionality

    Attributes
    ----------
    activation_checkpointing: Optional[DeepspeedActivationCheckpointingConfig], default: DeepspeedActivationCheckpointingConfig()
        Enables and configures activation checkpointing
    aio: Optional[DeepspeedAIOConfig], default: DeepspeedAIOConfig()
        Configuring the asynchronous I/O module for offloading parameter and optimizer states to persistent
        (NVMe) storage
    auto_mpi_discovery: bool, default: True
        if distributed environment variables are not set, attempt to discover them from MPI
    disable_allgather: bool, default: False
        Disables allgather
    dist_backend: BackendOptions, default: 'nccl'
        Which communication backend to use
    distributed_port: int, default: 29500
        torch distributed backend port
    dump_state: bool, default: False
        Print out state information of DeepSpeed object after initialization
    flops_profiler: Optional[DeepspeedFlopsConfig], default: None
        Enables and configures the flops profiler. This would also enable wall_clock_breakdown
    fp16: Optional[DeepspeedFP16Config], default: None
        Enables and configures mixed precision/FP16 training that leverages NVIDIA’s Apex package
    fp32_allreduce: bool, default: False
        During gradient averaging perform allreduce with 32 bit values
    gradient_predivide_factor: float, default: 1.0
        Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability
        when scaling to large numbers of GPUs
    init_method: str, default: 'env://'
        URL specifying how to initialize the process group
    prescale_gradients: float, default: 1.0
        Scale gradients before doing allreduce
    progressive_layer_drop: Optional[DeepspeedPLDConfig], default: None
        Enables and configures progressive layer dropping
    sparse_gradients: bool, default: False
        Enable sparse compression of torch.nn.Embedding gradients
    steps_per_print: int, default: 10
        Print train loss every N steps
    tensorboard: Optional[DeepspeedTensorboardConfig], default: None
        Enables and configures tensorboard support
    verbose: bool, default: True
        flag to make deepspeed engine verbose with information
    wall_clock_breakdown: bool, default: False
        Enable timing of the latency of forward/backward/update training phases
    zero_optimization: Optional[DeepspeedZeROConfig], default: DeepspeedZeROConfig()
        Enables and configures ZeRO memory optimizations

    Notes
    -----
    Deepspeed does not use Apex’s AMP mode whihc allows for more flexibility in mixed precision training modes. FP16
    here is similar to AMP’s O2 mode

    """

    activation_checkpointing: Optional[
        DeepspeedActivationCheckpointingConfig
    ] = DeepspeedActivationCheckpointingConfig()
    aio: Optional[DeepspeedAIOConfig] = DeepspeedAIOConfig()
    auto_mpi_discovery: bool = True
    disable_allgather: bool = False
    dist_backend: BackendOptions = "nccl"
    distributed_port: int = 29500
    dump_state: bool = False
    flops_profiler: Optional[DeepspeedFlopsConfig] = None
    fp16: Optional[DeepspeedFP16Config] = None
    fp32_allreduce: bool = False
    gradient_predivide_factor: float = 1.0
    init_method: str = "env://"
    prescale_gradients: bool = False
    progressive_layer_drop: Optional[DeepspeedPLDConfig] = None
    sparse_gradients: bool = False
    steps_per_print: int = 10
    tensorboard: Optional[DeepspeedTensorboardConfig] = None
    verbose: bool = True
    wall_clock_breakdown: bool = False
    zero_optimization: Optional[DeepspeedZeROConfig] = DeepspeedZeROConfig()


@attr.s(auto_attribs=True)
class FairscaleOSSConfig:
    """Fairscale optimizer state sharding configuration class

    Attributes
    ----------
    broadcast_fp16: bool, default: False
        Compress the model shards in fp16 before sharing them in between ranks. This is safe to use when PyTorch AMP
        is activated. Without torch AMP this will lead to a slight degradation in terms of accuracy.

    """

    broadcast_fp16: bool = False


@attr.s(auto_attribs=True)
class FairscaleSDDPConfig:
    """Fairscale sharded data parallel configuration class

    Attributes
    ----------
    auto_refresh_trainable: bool, default: True
        Check whether the parameters trainability (requires_grad) has changed and update both ShardedDDP and OSS
        automatically if this is the case. If set to False, refresh_trainable() needs to be called anytime a
        parameter is frozen or unfrozen
    broadcast_buffers: bool, default: True
        Whether to additionally broadcast model buffers in between ranks at the beginning of each forward pass. Same
        setting as in Pytorch DDP, this is in addition to the broadcast and reduction of the model parameters.
    reduce_buffer_size: int, default: 2 ** 23
        he max size of the buffer used to batch the small parameter tensors, in number of elements. This will impact
        the long term memory consumption, because these buckets correspond to parameters which will not be sharded.
        Set to 0 to remove all bucketing, 1M to 8M is usually reasonable.
    reduce_fp16: bool, default: False
        cast the grads to fp16 before reducing. Not needed if the model is already fp16, but will probably improve
        performance for multi node jobs using PyTorch AMP. The effect is similar to DDP’s fp16_compress_hook and
        will also save some memory.
    sync_models_at_startup: bool, default: True
        Synchronize the models in between the ranks when starting up. Not needed if each rank has the same seed, or
        the training restarts from a saved state

    """

    auto_refresh_trainable: bool = True
    broadcast_buffers: bool = True
    reduce_buffer_size: int = 2 ** 23
    reduce_fp16: bool = False
    sync_models_at_startup: bool = True


@attr.s(auto_attribs=True)
class HorovodConfig:
    """Horovod configuration class

    Attributes
    ----------
    compression: bool, default: False
        Compression algorithm used during allreduce to reduce the amount of data sent during the each parameter
        update step.
    convert_to_sync_batch_norm: bool, default: False
        Automatically convert all batch norm calls to horovod.torch.SyncBatchNorm calls
        https://horovod.readthedocs.io/en/stable/api.html#horovod.torch.SyncBatchNorm
    gradient_predivide_factor: float, default: 1.0
        If op == Average, gradient_predivide_factor splits the averaging before and after the sum. Gradients are scaled
        by 1.0 / gradient_predivide_factor before the sum and gradient_predivide_factor / size after the sum.
    op: HorovodOps, default: 'Average'
        The reduction operation to use when combining gradients across different ranks.

    """

    compression: bool = False
    convert_to_sync_batch_norm: bool = False
    gradient_predivide_factor: float = 1.0
    op: HorovodOps = "Average"


class StokeOptimizer(TypedDict):
    """Stoke optimizer wrapper class

    Given all the different backends and extensions the optimizer might need to be instantiated in a different way
    thus this typed dict holds the configuration without instantiation

    Attributes
    ----------
    optimizer: Type[torch.optim.Optimizer]
        un-instantiated torch.optim.Optimizer class
    optimizer_kwargs: Dict
        any keyword args to be unrolled into the optimizer at instantiation time

    """

    optimizer: Type[torch.optim.Optimizer]
    optimizer_kwargs: Dict
