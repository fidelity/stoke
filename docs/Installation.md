# Installation

### Requirements

* Python: 3.6+
* Pip Dependencies: attrs, deepspeed, fairscale, horovod, mypy_extensions (python_version < '3.8'),
  torch>=1.8.1
* Optional Pip Dependencies: mpi4py
* Build Dependencies: apex (NVIDIA)
* Tested OS: Unix (Ubuntu 16.04, Ubuntu 18.04), OSX (10.14.6)

### (Required) Install NVIDIA Apex

Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start).

### (Optional) OpenMPI Support

Follow the instructions [here](https://www.open-mpi.org/faq/?category=building) or 
[here](https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf)

Also, refer to the Dockerfile [here](https://github.com/fidelity/stoke/blob/master/docker/stoke-gpu-mpi.Dockerfile)

### PyPi
```bash
pip install stoke
```

### PyPi From Source
```bash
pip install git+https://github.com/fidelity/stoke
```

### Build From Source
```bash
git clone https://github.com/fidelity/stoke
cd stoke
pip install setuptools wheel
python setup.py bdist_wheel
pip install /dist/stoke-X.X.XxX-py3-none-any.whl
```