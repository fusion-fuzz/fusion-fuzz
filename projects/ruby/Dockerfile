# Ruby (ruby/ruby trunk) is built from source by setup.py inside this
# image. Two of Ruby's build-from-git requirements decide the base image:
# a BASERUBY >= 3.1 (Ubuntu 22.04 ships 3.0) and, for the AddressSanitizer
# build, clang >= 18 (doc/contributing/building_ruby.md: earlier clang
# mishandles fork under ASan). Ubuntu 24.04 has both in apt.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang-18 \
    lld-18 \
    llvm-18 \
    libclang-rt-18-dev \
    autoconf \
    bison \
    ruby \
    git \
    ca-certificates \
    curl \
    wget \
    pkg-config \
    libyaml-dev \
    libssl-dev \
    zlib1g-dev \
    libffi-dev \
    libgdbm-dev \
    libreadline-dev \
    libgmp-dev \
    python3 \
    python3-pip \
    python3-yaml \
    python3-requests \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    file \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The repository is bind-mounted and owned by uid 1000 on the host; the
# fuzz user must have that uid to write corpus.db and .fused/ into it.
# 24.04 pre-creates an `ubuntu` user at 1000, so it goes first.
RUN userdel -r ubuntu 2>/dev/null || true
RUN useradd -m -u 1000 fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz
ENV CC=clang-18 CXX=clang++-18
# The interpreter under test is executed, not just compiled: nothing it
# runs may fetch gems or reach the network.
ENV GEM_HOME=/tmp/ffl-gems

CMD ["/bin/bash"]
