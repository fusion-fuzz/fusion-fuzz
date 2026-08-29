# Go is built from source (see setup.py). Building Go trunk needs an
# existing Go toolchain to bootstrap from — src/make.bash sets
# bootgo=1.26.0 — and Ubuntu 22.04's apt only carries 1.18, so the
# bootstrap toolchain is downloaded here rather than installed.
FROM ubuntu:22.04

LABEL maintainer="yuancheng@comp.nus.edu.sg"

ARG DEBIAN_FRONTEND=noninteractive
ARG BOOTSTRAP_GO=1.26.0

# gcc is needed for cgo, which the standard library's net and os/user
# packages use; without it those build in pure-Go mode and a seed that
# exercises cgo paths cannot be compiled at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
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

# Bootstrap toolchain, installed where make.bash looks for it by default
# ($HOME/sdk/go<bootgo>) so setup.py does not have to pass GOROOT_BOOTSTRAP.
RUN curl -fsSL "https://go.dev/dl/go${BOOTSTRAP_GO}.linux-amd64.tar.gz" \
      -o /tmp/bootstrap.tar.gz \
    && mkdir -p /usr/local/go-bootstrap \
    && tar -C /usr/local/go-bootstrap --strip-components=1 -xzf /tmp/bootstrap.tar.gz \
    && rm /tmp/bootstrap.tar.gz \
    && /usr/local/go-bootstrap/bin/go version

ENV GOROOT_BOOTSTRAP=/usr/local/go-bootstrap

RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace
RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

USER fuzz
# Keep module downloads off: seeds import only the standard library, and a
# fused program must never reach the network mid-fuzz.
ENV GOPROXY=off GOFLAGS=-mod=mod

CMD ["/bin/bash"]
