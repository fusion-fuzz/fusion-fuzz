# Plain Ubuntu 22.04 base — GCC is built from source at setup time (see
# setup.py), so this image only carries what is needed to fetch and build
# it, plus the Python deps FFL itself needs.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# Build prerequisites for GCC itself. Beyond the usual toolchain, GCC needs
# GMP/MPFR/MPC (its arbitrary-precision maths libraries) and flex/bison for
# the generated parsers; `gcc-multilib` supplies the 32-bit headers so a
# seed compiled with -m32 does not fail on the driver rather than in the
# compiler. `libisl-dev` enables Graphite loop transforms, another chunk of
# optimiser surface worth exercising.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    libisl-dev \
    flex \
    bison \
    texinfo \
    gcc-multilib \
    g++-multilib \
    git \
    ca-certificates \
    curl \
    wget \
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

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
