# ffl-cangjie — Cangjie compiler fuzzing container
#
# Build (no SDK needed at build time):
#   docker build -t ffl-cangjie projects/cangjie/
#
# The Cangjie SDK is downloaded and installed by setup.py at run time:
#   docker run --rm -v $(pwd):/workspace -w /workspace ffl-cangjie \
#     python3 main.py --project cangjie --setup
#
# setup.py reads CANGJIE_SDK_URL from the environment (or falls back to a
# local tarball at projects/cangjie/cangjie-sdk-linux-x64.tar.gz).
# SDK installs to /opt/cangjie inside the container volume.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime and build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ca-certificates \
    libssl-dev \
    libc6-dev \
    libstdc++6 \
    python3 \
    python3-pip \
    sudo \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (mirrors root requirements.txt)
RUN pip3 install --no-cache-dir pyyaml>=6.0 requests>=2.28

# PATH/LD_LIBRARY_PATH for the SDK location chosen in setup.py (/opt/cangjie)
ENV PATH="/opt/cangjie/bin:/opt/cangjie/tools/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cangjie/runtime/lib:/opt/cangjie/tools/lib:${LD_LIBRARY_PATH:-}"

# Add a default user "fuzz" with password "fuzz" and add to sudo group
RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

WORKDIR /home/fuzz/WorkSpace

RUN chown -R fuzz /home/fuzz/WorkSpace

USER fuzz

CMD ["/bin/bash"]
