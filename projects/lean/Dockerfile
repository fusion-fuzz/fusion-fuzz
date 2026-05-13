FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies for Lean 4 (GMP, libstdc++, curl for elan)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgmp-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching the host UID/GID (passed at build time)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ffluser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ffluser

USER ffluser
WORKDIR /home/ffluser

# Install elan (Lean version manager) and bake in the default lean toolchain
ENV PATH="/home/ffluser/.elan/bin:${PATH}"
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --no-modify-path \
    && /home/ffluser/.elan/bin/lean --version

WORKDIR /workspace

CMD ["/bin/bash"]
