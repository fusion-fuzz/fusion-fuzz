# V8 is built from source (see setup.py). The checkout is done with
# depot_tools/gclient rather than a plain git clone: V8's build needs a
# dozen sibling repositories (build/, buildtools/, third_party/...) that
# only gclient knows how to place.
# 24.04, not 22.04: depot_tools imports enum.StrEnum, which is Python
# 3.11+. Jammy ships 3.10 and every gclient invocation dies on the import
# before it does anything.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# depot_tools still shells out to python2-era tooling in places and wants
# a real git; the rest is what V8's own build/install-build-deps.sh pulls
# in for a headless Linux build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    lld \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    curl \
    wget \
    file \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-dev \
    libnss3-dev \
    libatk1.0-dev \
    libc++-dev \
    libc++abi-dev \
    lsb-release \
    xz-utils \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 24.04 marks the system interpreter externally managed (PEP 668). These
# are the two packages FusionFuzz itself needs and the container has no
# other Python consumer, so installing into it is the simpler choice.
RUN pip3 install --no-cache-dir --break-system-packages pyyaml requests

# 24.04 ships a default uid-1000 `ubuntu` user; remove it so `fuzz` can
# take uid 1000 and match the host user that owns the bind-mounted repo.
RUN userdel -r ubuntu 2>/dev/null || true; \
    useradd -m -u 1000 fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

# depot_tools must be on PATH before the build; it supplies gn, ninja
# wrappers, gclient and the fetch recipes.
RUN git clone --depth=1 https://chromium.googlesource.com/chromium/tools/depot_tools.git \
        /opt/depot_tools \
    && chown -R fuzz:fuzz /opt/depot_tools
ENV PATH="/opt/depot_tools:${PATH}"
# depot_tools phones home unless told not to; a fuzzing container should
# not be sending usage metrics.
ENV DEPOT_TOOLS_METRICS=0
# depot_tools self-updates before every gclient invocation and hangs
# indefinitely against the depth=1 clone above. It is already at HEAD.
ENV DEPOT_TOOLS_UPDATE=0
ENV VPYTHON_BYPASS="manually managed python not supported by chrome operations"

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
