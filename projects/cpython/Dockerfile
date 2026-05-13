# Use the official Ubuntu 22.04 as a base image
FROM ubuntu:22.04

# Set the maintainer label
LABEL maintainer="yuancheng@comp.nus.edu.sg"

# Prevent interactive prompts from appearing during package installation
ARG DEBIAN_FRONTEND=noninteractive

# 1. Update and install basic tools required to add PPAs and download scripts
RUN apt-get update && \
    apt-get install -y \
    lsb-release \
    wget \
    software-properties-common \
    gnupg \
    git \
    vim \
    curl \
    sudo \
    build-essential \
    pkg-config \
    zip \
    unzip \
    tmux \
    gcovr \
    lcov \
    valgrind \
    # 2. Add Deadsnakes PPA for newer Python versions (Required for CPython JIT tools)
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    # 3. Install Python 3.12 (Build Host) and CPython Dependencies
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    uuid-dev \
    tcl-dev \
    tk-dev \
    libxml2-dev \
    libmpdec-dev \
    # 4. Install LLVM 21 using the official script
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 21 all \
    # 5. Configure Python 3.12 as the default 'python3'
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    # 6. Install Pip for Python 3.12
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm llvm.sh

# Set Clang 21 as the default compiler environment variables
ENV CC=clang-21
ENV CXX=clang++-21

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
