FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
