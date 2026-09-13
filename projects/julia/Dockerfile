# Julia (JuliaLang/julia master) is built from source by setup.py inside
# this image. The build downloads prebuilt dependencies (LLVM, OpenBLAS,
# SuiteSparse and the rest) rather than building them, which is the
# difference between a 40-minute build and an all-day one; Julia's own
# assertions are turned on separately and cost nothing.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    m4 \
    patch \
    perl \
    pkg-config \
    python3 \
    python3-pip \
    python3-yaml \
    python3-requests \
    git \
    ca-certificates \
    curl \
    wget \
    unzip \
    zip \
    file \
    libatomic1 \
    sqlite3 \
    tmux \
    vim \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The repository is bind-mounted and owned by uid 1000 on the host.
RUN userdel -r ubuntu 2>/dev/null || true
RUN useradd -m -u 1000 fuzz && echo "fuzz:fuzz" | chpasswd && usermod -aG sudo fuzz
WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace
USER fuzz
# Nothing the fuzzer runs may fetch a package from the registry.
ENV JULIA_DEPOT_PATH=/tmp/ffl-julia-depot JULIA_PKG_OFFLINE=true
CMD ["/bin/bash"]
