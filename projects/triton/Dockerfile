# Triton is built from source, but its LLVM/MLIR is downloaded prebuilt (see
# setup.py), so this image only needs a C++ toolchain, CMake/Ninja and the
# Python bits FFL itself uses — no multi-hour LLVM build like the clang
# adapter's image implies.
# 24.04 for clang 18: Triton main uses C++20 features (lambdas capturing
# structured bindings) that 22.04's clang 14 rejects.
FROM ubuntu:24.04

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
    python3-yaml \
    python3-requests \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    file \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN (userdel -r ubuntu 2>/dev/null || true) && useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
