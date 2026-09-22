# projects/xla/Dockerfile — build environment for the XLA adapter.
#
# XLA is built with Bazel from source (see setup.py). Bazel itself is fetched
# by bazelisk at the version XLA pins in .bazelversion, and XLA's own build
# downloads a hermetic clang toolchain (USE_HERMETIC_CC_TOOLCHAIN=1 in its
# .bazelrc), so the image only needs: a JVM-free bazelisk binary, python3,
# git, and the system libraries Bazel's downloaded toolchain expects. A
# system clang is installed too, for the non-hermetic (`--config=clang_local`)
# fallback and for the driver's symbolizer.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates gnupg lsb-release \
    software-properties-common unzip zip patch file \
    python3 python3-pip python3-dev python3-numpy \
    sudo tmux vim sqlite3 \
    libtinfo5 libxml2 zlib1g-dev \
    && wget -q https://apt.llvm.org/llvm.sh && chmod +x llvm.sh \
    && ./llvm.sh 18 && rm llvm.sh \
    && apt-get install -y --no-install-recommends lld-18 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# bazelisk: resolves .bazelversion (8.7.0 at the pinned checkout) and caches
# the matching bazel under ~/.cache/bazelisk.
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

RUN pip3 install --no-cache-dir pyyaml requests

ENV CLANG_PATH=/usr/lib/llvm-18/bin/clang
ENV PATH="/usr/lib/llvm-18/bin:${PATH}"

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz && \
    echo "fuzz ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
