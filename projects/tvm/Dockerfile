# TVM is built from source at setup time (CMake + Ninja). LLVM 18 comes
# from apt.llvm.org (Ubuntu 22.04's own is 14/15; TVM's backend follows
# released LLVM APIs, and projects/clang's assertion-enabled trunk LLVM
# does not compile with it yet — FFL_TVM_LLVM_CONFIG selects it when it
# does). tvm-ffi needs CMake >= 3.26, newer than the distro's, and its
# Python package is built with scikit-build-core + Ninja, so those come
# from pip into /usr/local, ahead of /usr/bin on PATH.
#
# The base is NVIDIA's CUDA 13.4 devel image (no GPU is needed or assumed):
# the GPU backends are fuzzed compile-only, and the toolkit supplies what
# the compile side needs — nvcc/ptxas/nvrtc for TVM's generated CUDA C,
# libdevice for the nvptx target. The Vulkan headers + SPIRV-Tools make
# USE_VULKAN=ON build (SPIR-V codegen, validated with spirv-val), and
# rocm-device-libs' bitcode lets the rocm target link (lld from LLVM 18).
FROM nvidia/cuda:13.4.1-devel-ubuntu22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    zlib1g-dev \
    libzstd-dev \
    libxml2-dev \
    libtinfo-dev \
    libedit-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    gdb \
    file \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    wget \
    gnupg \
    libvulkan-dev \
    spirv-tools \
    spirv-headers \
    rocm-device-libs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# LLVM 18 development packages (llvm-config, static libs, Polly) from apt.llvm.org.
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
       > /etc/apt/sources.list.d/llvm18.list \
    && apt-get update && apt-get install -y --no-install-recommends llvm-18-dev libpolly-18-dev lld-18 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# FFL's own deps, the build tools tvm-ffi's Python package needs, and what
# python/tvm imports at runtime.
RUN pip3 install --no-cache-dir pyyaml requests \
    "cmake>=3.28" ninja scikit-build-core \
    numpy scipy cloudpickle ml_dtypes packaging psutil tornado typing-extensions \
    decorator attrs pytest

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

# TVM's ROCm codegen looks for device bitcode under $ROCM_PATH/amdgcn/bitcode
# and links it with ld.lld. Ubuntu ships the bitcode in rocm-device-libs but
# not the directory layout TVM expects, so give it one. Compile-only: no
# ROCm runtime and no AMD GPU are involved.
RUN mkdir -p /opt/rocm && ln -s /usr/lib/x86_64-linux-gnu/amdgcn /opt/rocm/amdgcn \
    && ln -sf /usr/lib/llvm-18/bin/ld.lld /usr/local/bin/ld.lld

ENV ROCM_PATH=/opt/rocm \
    CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

USER fuzz

CMD ["/bin/bash"]
