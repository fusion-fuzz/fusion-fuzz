FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies for host-compiled wgslc binary
RUN apt-get update && apt-get install -y \
    libicu74 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Add a standard user to prevent permission issues
ARG UID=1009
ARG GID=1009
RUN groupadd -g ${GID} ffluser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ffluser

# Set working directory
WORKDIR /workspace
RUN chown -R ffluser:ffluser /workspace

USER ffluser

CMD ["/bin/bash"]
