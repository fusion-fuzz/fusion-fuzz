# TVM is built from source at setup time (CMake + Ninja) against the
# assertion-enabled LLVM that projects/clang's from-source build already
# provides on the same repository mount (its llvm-config reports
# --assertion-mode ON), so no LLVM is built or installed here: the image
# only needs a C++ toolchain, CMake/Ninja and the Python runtime TVM's
# Python package depends on.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    zlib1g-dev \
    libzstd-dev \
    libxml2-dev \
    libtinfo-dev \
    libedit-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    gdb \
    file \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# FFL's own deps plus what python/tvm imports at runtime.
RUN pip3 install --no-cache-dir pyyaml requests \
    numpy scipy cloudpickle ml_dtypes packaging psutil tornado typing-extensions \
    decorator attrs pytest

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
