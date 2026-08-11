FROM rust:1.88-bookworm

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    python3-psutil \
    tmux \
    vim \
    sudo \
    curl \
    ca-certificates \
    && pip3 install --break-system-packages pyyaml \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install --git https://github.com/gfx-rs/wgpu.git --branch trunk naga-cli

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
