# R (r-devel, the R Core Team's development trunk) is built from source by
# setup.py inside this image. R's build needs a Fortran compiler and the
# usual text/compression libraries; clang 18 and its sanitizer runtime are
# here for the AddressSanitizer build.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang-18 \
    libclang-rt-18-dev \
    gfortran \
    autoconf \
    automake \
    make \
    git \
    ca-certificates \
    curl \
    wget \
    pkg-config \
    libreadline-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    libicu-dev \
    libblas-dev \
    libtirpc-dev \
    liblapack-dev \
    gettext \
    rsync \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The repository is bind-mounted and owned by uid 1000 on the host; 24.04
# pre-creates an `ubuntu` user at that uid, so it goes first.
RUN userdel -r ubuntu 2>/dev/null || true
RUN useradd -m -u 1000 fuzz && echo "fuzz:fuzz" | chpasswd && usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace
USER fuzz
ENV CC=clang-18 CXX=clang++-18
# Nothing the fuzzer runs may fetch a package from CRAN.
ENV R_LIBS_USER=/tmp/ffl-rlibs R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null

CMD ["/bin/bash"]
