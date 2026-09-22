# The Mojo compiler is open source in modular/modular (Mojo/lib, Mojo/tools)
# and builds with Bazel: ./bazelw fetches bazelisk, bazelisk fetches the
# pinned (buildbuddy) Bazel, and the repo's hermetic C++ toolchain and its
# LLVM/MLIR source are Bazel repositories, so the image only needs the
# fetch tools, a JVM-less Bazel's few host libraries and Python for FFL.
# The released `mojo` wheels are also installable here as a fallback
# toolchain (see setup.py); they need nothing beyond glibc 2.34+.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    zip \
    unzip \
    patch \
    file \
    gdb \
    libtinfo5 \
    libxml2 \
    zlib1g-dev \
    libzstd-dev \
    libedit-dev \
    libncurses-dev \
    sqlite3 \
    tmux \
    vim \
    sudo \
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
