# Plain Ubuntu 22.04 base — LLVM/Clang is built from source at setup time
# (see setup.py), so this image only needs the tools required to fetch and
# build llvm-project plus the Python deps FFL itself needs.
FROM ubuntu:22.04

# Set the maintainer label
LABEL maintainer="yuancheng@comp.nus.edu.sg"

# Prevent interactive prompts from appearing during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Build toolchain for llvm-project (cmake + ninja + a host C/C++ compiler)
# plus git/curl to fetch the source and python3 for FFL/LLVM build scripts.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyYAML/requests globally via pip (used by core/config_loader.py, core/llmgen.py)
RUN pip3 install --no-cache-dir pyyaml requests

# Setup User
RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

# Set up the workspace
WORKDIR /home/fuzz/WorkSpace

RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

# Switch to the user
USER fuzz

# Define the default command
CMD ["/bin/bash"]
