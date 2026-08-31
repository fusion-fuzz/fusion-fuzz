# SpiderMonkey (Firefox's JavaScript engine) is built from source; see
# setup.py. Unlike V8 it needs no gclient — the JS shell has a standalone
# autoconf build under js/src, so a shallow git clone is enough.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# The build needs a C++ toolchain, Rust (SpiderMonkey has Rust components
# — mozglue and the encoding/unicode crates), and the autoconf 2.13 that
# js/src/configure was generated with.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    llvm \
    cmake \
    ninja-build \
    make \
    autoconf2.13 \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    zlib1g-dev \
    libffi-dev \
    unzip \
    zip \
    xz-utils \
    file \
    sqlite3 \
    tmux \
    vim \
    sudo \
    procps \
    lsb-release \
    gnupg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# SpiderMonkey's build requires clang >= 19; noble ships 18.1.3 and the
# configure script refuses it outright. Install a supported one from the
# LLVM project's own apt repository.
ARG LLVM_VERSION=20
RUN curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
    && chmod +x /tmp/llvm.sh \
    && /tmp/llvm.sh ${LLVM_VERSION} \
    && apt-get install -y --no-install-recommends \
       clang-${LLVM_VERSION} lld-${LLVM_VERSION} llvm-${LLVM_VERSION}-dev \
       libclang-rt-${LLVM_VERSION}-dev \
    && update-alternatives --install /usr/bin/clang clang \
       /usr/bin/clang-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ \
       /usr/bin/clang++-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/lld lld \
       /usr/bin/lld-${LLVM_VERSION} 100 \
    && rm -f /tmp/llvm.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml requests

# 24.04 ships a default uid-1000 `ubuntu` user; remove it so `fuzz` can
# take uid 1000 and match the host user that owns the bind-mounted repo.
RUN userdel -r ubuntu 2>/dev/null || true; \
    useradd -m -u 1000 fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

USER fuzz

# Rust is a hard build dependency, and the version Mozilla pins moves, so
# it comes from rustup rather than the distro package.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/home/fuzz/.cargo/bin:${PATH}"

# SpiderMonkey generates C headers for its Rust components with cbindgen,
# and configure fails outright without it. Mozilla's own instruction is
# `mach bootstrap`, which wants the full Firefox tree; installing the tool
# directly is enough for the standalone js/src build.
RUN cargo install cbindgen --locked

WORKDIR /home/fuzz/WorkSpace

CMD ["/bin/bash"]
