# Use the official Ubuntu 22.04 as a base image
FROM haskell:latest

# Set the maintainer label
LABEL maintainer="yuancheng@comp.nus.edu.sg"

# Prevent interactive prompts from appearing during package installation
ARG DEBIAN_FRONTEND=noninteractive

# 1. Update and install basic tools required to add PPAs and download scripts
RUN apt update && apt install -y \
    tmux \
    vim \
    sudo \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-psutil \
    curl \
    unzip \
    zip \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyYAML globally via pip
RUN pip3 install --break-system-packages pyyaml

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
