# Triton is built from source, but its LLVM/MLIR is downloaded prebuilt (see
# setup.py), so this image only needs a C++ toolchain, CMake/Ninja and the
# Python bits FFL itself uses — no multi-hour LLVM build like the clang
# adapter's image implies.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# clang/lld rather than gcc: Triton's prebuilt LLVM is built with clang, and
# linking a project this size with lld is markedly faster than with bfd.
# zlib/zstd are what LLVM's own object-file readers link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    curl \
    wget \
    zlib1g-dev \
    libzstd-dev \
    python3 \
    python3-pip \
    python3-dev \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    file \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pyyaml requests

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
