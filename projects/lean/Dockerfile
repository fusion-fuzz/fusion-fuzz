# Lean 4 (leanprover/lean4 master) is built from source by setup.py inside
# this image — twice, from the same checkout:
#
#   * build/relwithassert   CMAKE_BUILD_TYPE=RelWithAssert: -O3 with
#                           LEAN_DEBUG, i.e. the kernel's, the runtime's and
#                           the C emitted for the Lean-written frontend all
#                           keep their `lean_assert`s.
#   * build/asan            the same, plus -fsanitize=address,undefined on
#                           the C++ kernel/runtime and on the stage1 C code
#                           (the repo's own `sanitize` preset flags).
#
# Lean bootstraps from the C sources checked in under stage0/, so no
# existing Lean toolchain (elan) is needed. The dependency list is the one
# doc/make/ubuntu.md gives, plus lld, libclang-rt (the sanitizer runtimes
# and their headers) and llvm (the sanitizer preset links
# with -fuse-ld=lld; llvm-symbolizer turns ASan reports into frames).
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    libclang-rt-18-dev \
    llvm \
    cmake \
    ccache \
    pkgconf \
    libgmp-dev \
    libuv1-dev \
    libssl-dev \
    zlib1g-dev \
    git \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    python3-yaml \
    python3-psutil \
    python3-requests \
    sqlite3 \
    gdb \
    file \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The repository is bind-mounted and owned by uid 1000 on the host; the
# base image's own uid-1000 user is removed so `fuzz` can take it.
RUN userdel -r ubuntu 2>/dev/null || true
RUN useradd -m -u 1000 fuzz && echo "fuzz:fuzz" | chpasswd && usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz
ENV CCACHE_DIR=/home/fuzz/.ccache
# A Lean-level panic prints a demangled backtrace (the default); the C++
# side's `lean_assert` asks a debugger prompt on stdin, which the driver
# closes so the process aborts instead of waiting.
CMD ["/bin/bash"]
