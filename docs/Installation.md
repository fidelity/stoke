# Installation

### Requirements

* Python: 3.6+
* Pip Dependencies: attrs, deepspeed, fairscale, horovod[pytorch], mypy_extensions (python_version < '3.8'), mpi4py,
  torch>=1.8.1
* Build Dependencies: apex (NVIDIA)
* Tested OS: Unix (Ubuntu 16.04, Ubuntu 18.04), OSX (10.14.6)

### Install NVIDIA Apex

Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start).

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