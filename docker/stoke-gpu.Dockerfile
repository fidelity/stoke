############
# BASE IMAGE
############
# Pull from the base ubuntu image w/ the correct ARCH and CUDA
FROM nvidia/cuda:11.1-base-ubuntu18.04 as base

##########
# ENV VARS
##########
ENV LANG C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

######################
# INSTALL DEPENDENCIES
######################
# Use bash instead of sh
# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# All CUDA dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-11-1 \
        cuda-libraries-11-1 \
        libcudnn8=8.1.0.77-1+cuda11.2 \
        libnccl2=2.8.4-1+cuda11.1 \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# All CUDA-dev dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-11-1 \
        cuda-compiler-11-1 \
        libcudnn8-dev=8.1.0.77-1+cuda11.2 \
        libnccl-dev=2.8.4-1+cuda11.1 \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Pickup additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++-7 \
        git \
        ibverbs-providers \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libibverbs1 \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libsndfile1 \
        libzmq3-dev \
        openssh-client \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub and reconfigure dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

############
# PIP/Python
############
# Add deadsnakes which has all Python versions
RUN add-apt-repository ppa:deadsnakes/ppa
# Install python X.X-dev and pip via apt-get
RUN apt-get update && apt-get install -y python3.8-dev python3.8-distutils python3-pip python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Create the symlink to the correct version of python X.X
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
# Upgrade pip -- this will link the pip cmd to the correct version of python X.X
RUN python3.8 -m pip --no-cache-dir install --upgrade pip
# Upgrade setuptools
RUN pip --no-cache-dir install --upgrade setuptools
# Some tools expect a "python" binary
RUN ln -s $(which python3.8) /usr/local/bin/python

################
# PYTORCH -- GPU
################
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

################
# HOROVOD -- GPU
################
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs
RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch] && ldconfig

#############
# NVIDIA APEX
#############
WORKDIR /usr/src/code/
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /usr/src/code/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install stoke w/o MPI support
RUN pip install --no-cache-dir --trusted-host pypi.python.org stoke