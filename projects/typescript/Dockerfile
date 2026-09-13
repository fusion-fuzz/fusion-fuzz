# TypeScript (microsoft/TypeScript master) is the Go-native compiler since
# TypeScript 7: `tsc/` is a Go module (go 1.26). Nothing but a Go
# toolchain is needed to build it; setup.py does the build.
#
# In this campaign the running container reuses the ffe-go image instead
# of building this one: the fuzzer executes only the statically linked
# `tsc-bin` that setup.py produced, so at fuzz time nothing beyond
# python3 is needed, and ffe-go already carries a go1.26 toolchain at
# /usr/local/go-bootstrap for the build. This file is what to build when
# the image is wanted on its own (disk budget permitting).
FROM ubuntu:24.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive
ARG GO_VERSION=1.26.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl wget python3 python3-pip \
    python3-yaml python3-requests sqlite3 tmux vim sudo unzip zip file \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz \
    && /usr/local/go/bin/go version

# The repository is bind-mounted and owned by uid 1000 on the host.
RUN userdel -r ubuntu 2>/dev/null || true
RUN useradd -m -u 1000 fuzz && echo "fuzz:fuzz" | chpasswd && usermod -aG sudo fuzz
WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace
USER fuzz
ENV PATH=/usr/local/go/bin:/home/fuzz/go/bin:$PATH GOFLAGS=-mod=mod
# The module cache is filled once by setup.py's `go build`; the fuzzer
# never builds anything afterwards.
CMD ["/bin/bash"]
