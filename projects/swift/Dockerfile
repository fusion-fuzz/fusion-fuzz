# Use the official Ubuntu 22.04 as a base image
FROM swiftlang/swift:nightly-noble

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
    libicu-dev \
    icu-devtools \
    libcurl4-openssl-dev \
    libedit-dev \
    libxml2-dev \
    libsqlite3-dev \
    swig \
    libncurses5-dev \
    pkg-config \
    libblocksruntime-dev \
    libbsd-dev \
    systemtap-sdt-dev \
    tzdata \
    rsync \
    curl \
    unzip \
    zip \
    uuid-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    clang-15 lld-15 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Export environment variables so the build script picks them up
RUN export CC=clang-15
RUN export CXX=clang++-15

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
