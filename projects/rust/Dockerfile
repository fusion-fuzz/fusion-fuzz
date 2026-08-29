# rustc is built from source (see setup.py) rather than taken from
# rustlang/rust:nightly, because a released nightly has neither rustc's own
# debug assertions nor LLVM assertions compiled in — and those are most of
# what this fuzzer looks for.
#
# x.py downloads its own stage0 toolchain, so no Rust needs to be installed
# here; this image only carries what building rustc and LLVM needs.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# cmake/ninja/clang are for LLVM, in case bootstrap decides to build it from
# source rather than download the CI artifact. pkg-config + libssl-dev are
# what cargo links against.
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
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    python3 \
    python3-pip \
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
