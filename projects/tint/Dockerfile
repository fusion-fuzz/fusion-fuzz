# tint (Dawn's WGSL compiler) is built from source; see setup.py. Unlike V8
# it needs no gclient — Dawn's CMake build fetches its own dependencies with
# a plain Python script (DAWN_FETCH_DEPENDENCIES=ON) — and only the tint
# command-line tool is built, not the GPU backends.
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive

# clang + cmake + ninja build the compiler; python3 and git run Dawn's
# dependency fetch. No X11/mesa/wayland packages: setup.py turns the window
# system off entirely (tint never opens a surface), which is also what
# avoids GLFW's hard requirement on wayland-scanner.
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
    python3 \
    python3-pip \
    pkg-config \
    zlib1g-dev \
    xz-utils \
    file \
    sqlite3 \
    tmux \
    vim \
    sudo \
    unzip \
    zip \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml requests

# 24.04 ships a default uid-1000 `ubuntu` user; remove it so `fuzz` can
# take uid 1000 and match the host user that owns the bind-mounted repo.
RUN userdel -r ubuntu 2>/dev/null || true; \
    useradd -m -u 1000 fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

USER fuzz
WORKDIR /home/fuzz/WorkSpace

CMD ["/bin/bash"]

# clang and its sanitizer runtime.
#
# The package that carries compiler-rt differs by release: on 22.04 the
# `clang` package includes it, on 24.04 it is a separate
# libclang-rt-<version>-dev. So the version is read back from the clang that
# got installed rather than pinned here.
#
# The final compile is the point of this step, not a test of it: without
# compiler-rt a sanitized build gets thousands of objects in before failing
# on a missing sanitizer/asan_interface.h. Failing here instead makes the
# image refuse to be built rather than produce one that cannot instrument.
RUN apt-get update && apt-get install -y --no-install-recommends clang && \
    v="$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p' | head -1)" && \
    (apt-get install -y --no-install-recommends "libclang-rt-$v-dev" || true) && \
    printf '#include <sanitizer/asan_interface.h>\nint main(void){return 0;}\n' > /tmp/rtprobe.c && \
    clang -fsanitize=address,undefined -fno-sanitize=function /tmp/rtprobe.c -o /tmp/rtprobe && \
    rm -f /tmp/rtprobe.c /tmp/rtprobe && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
