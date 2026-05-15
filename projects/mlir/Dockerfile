FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies and LLVM 21 toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget git curl sudo tmux vim \
    build-essential pkg-config unzip \
    python3 python3-pip python3-dev python3-setuptools \
    cmake ninja-build \
    zlib1g-dev libxml2-dev libedit-dev libncurses-dev \
    lsb-release software-properties-common gnupg \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 21 all \
    && rm llvm.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Compiler defaults — use LLVM 21
ENV CC=clang-21
ENV CXX=clang++-21
ENV PATH="/usr/lib/llvm-21/bin:${PATH}"

# 2. Python dependencies needed by the FFL framework
RUN pip3 install --no-cache-dir pyyaml requests numpy pybind11

# Add user "fuzz"
RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
