# CUDA compile-only fuzzing needs a CUDA toolkit (nvcc, cudafe++, cicc,
# ptxas, the runtime headers and libdevice) but no GPU and no driver: every
# mode the driver draws stops at PTX, cubin or an object file. NVIDIA's
# devel image carries the whole toolkit, so nothing is downloaded at setup
# time. clang for the CUDA frontend comes from projects/clang's
# from-source, assertions-on build when that tree is present (same
# repository mount), with the distro clang as fallback (see setup.py).
FROM nvidia/cuda:13.4.1-devel-ubuntu22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    gdb \
    git \
    ca-certificates \
    curl \
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

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
