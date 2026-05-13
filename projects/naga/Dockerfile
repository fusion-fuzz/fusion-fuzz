FROM rust:latest

# Pre-install Rust nightly and rust-src to avoid network access during builds
# This is done at image build time when network is available
RUN rustup toolchain install nightly-x86_64-unknown-linux-gnu && \
    rustup component add rust-src --toolchain nightly-x86_64-unknown-linux-gnu

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
