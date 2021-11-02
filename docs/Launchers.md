# Launchers

Stoke supports the following launchers...

### PyTorch DDP

#### PyTorch >= v1.10.0

PyTorch greatly simplified the launcher with the addition of `torchrun`:

```shell
torchrun train.py
```

#### PyTorch <= v1.10.0

Prefer the `torch.distributed.launch` utility described 
[here](https://pytorch.org/docs/stable/distributed.html#launch-utility) (Note: the local_rank requirement
propagates through to `stoke`)

```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --use_env train.py
```

### Horovod
Refer to the docs [here](https://horovod.readthedocs.io/en/stable/running_include.html)

```shell
horovodrun -np 4 -H localhost:4 python train.py
```
or
```shell
horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
```

### Horovod w/ OpenMPI
Refer to the docs [here](https://horovod.readthedocs.io/en/stable/mpi.html). Can also be used
with k8s via the [MPI Operator](https://github.com/kubeflow/mpi-operator)

```shell
mpirun -np 4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```
or
```shell
mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

### Deepspeed w/ OpenMPI
Prefer the OpenMPI version [here](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables) over the 
native launcher. Deepspeed will automatically discover devices, etc. via mpi4py. Can also be used
with k8s via the [MPI Operator](https://github.com/kubeflow/mpi-operator)

```shell
mpirun -np 4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```
or
```shell
mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```


### PyTorch DDP w/ OpenMPI
Leverage Deepspeed functionality to automatically discover devices, etc. via mpi4py. Can also be used
with k8s via the [MPI Operator](https://github.com/kubeflow/mpi-operator)

```shell
mpirun -np 4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```
or
```shell
mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```