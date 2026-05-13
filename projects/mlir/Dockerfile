FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive
# Limit parallelism so we don't OOM on machines with many cores (each compile
# unit with -O2 can consume 1-2 GB; 16 jobs × 2 GB = 32 GB, fits in 31 GB).
ARG JOBS=8

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

# 3. Build mlir-opt from source (no sanitizers to keep memory usage manageable).
#    The fuzzer driver still detects crashes via exit codes and stderr patterns.
#    Installed to /opt/mlir-install so setup.py can symlink it at runtime
#    without any further compilation.
RUN git clone --depth=1 https://github.com/llvm/llvm-project.git /opt/llvm-project \
    && cmake -S /opt/llvm-project/llvm -B /opt/mlir-build -G Ninja \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_TARGETS_TO_BUILD=host \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_PARALLEL_LINK_JOBS=1 \
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
        -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
        -DCMAKE_INSTALL_PREFIX=/opt/mlir-install \
    && cmake --build /opt/mlir-build --target mlir-opt -- -j${JOBS} \
    && mkdir -p /opt/mlir-install/bin \
    && cp /opt/mlir-build/bin/mlir-opt /opt/mlir-install/bin/ \
    && rm -rf /opt/mlir-build /opt/llvm-project

# Add user "fuzz"
RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
