# ffl-lfortran — LFortran compiler fuzzing container
#
# This image only carries the *toolchain* needed to build LFortran
# (cmake/ninja/g++ + LLVM-14 dev libs from apt) — it does NOT bake in an
# LLVM source build, conda, or a prebuilt LFortran binary, which is what
# LFortran's own docs recommend (conda + llvmdev). Using Ubuntu 22.04's
# system llvm-14-dev instead of conda keeps the image small.
#
# LFortran itself is built from a fresh clone of lfortran/lfortran's `main`
# branch (tip-of-main by default; set LFORTRAN_REF to pin a tag/commit) by
# setup.py at container-run time:
#   docker run --rm -v $(pwd):/workspace -w /workspace ffl-lfortran \
#     python3 main.py --project lfortran --setup
#
# Building from a git checkout (not a release tarball) means the generated
# parser/lexer/ASR headers don't exist yet, so setup.py replicates upstream's
# build0.sh codegen step itself (see _run_codegen() there) — everything
# except the LSP codegen step, which needs a `llanguage_server` pip package
# that doesn't resolve on PyPI and which we don't need anyway (WITH_LSP is
# never set). That needs: re2c, bison, python3 (stdlib only, no extra pip
# packages — verified against the actual codegen scripts' imports).
#
# setup.py installs LFortran to /opt/lfortran (PATH below) and reuses the
# same Fortran seed corpus as projects/flang (llvm-project's flang/test),
# so both projects fuzz off one shared, non-duplicated seed source.

FROM ubuntu:22.04

# Set the maintainer label
LABEL maintainer="yuancheng@comp.nus.edu.sg"

# Prevent interactive prompts from appearing during package installation
ARG DEBIAN_FRONTEND=noninteractive

# 1. Update and install basic tools + LFortran's build dependencies.
#    llvm-14-dev / libzstd-dev / zlib1g-dev / libunwind-dev are needed to
#    build LFortran itself (see cmake's WITH_LLVM/WITH_ZSTD/WITH_ZLIB/
#    WITH_LIBUNWIND_FORCE branches) — no clang, no rapidjson, no conda.
#    re2c / bison are needed for setup.py's codegen step (parser/tokenizer
#    generation), only required because we build from a git checkout of
#    main instead of a pre-generated release tarball.
RUN apt update && apt install -y \
    tmux \
    vim \
    sudo \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-psutil \
    curl \
    unzip \
    zip \
    llvm-14-dev \
    libzstd-dev \
    zlib1g-dev \
    libunwind-dev \
    python3-yaml \
    re2c \
    bison \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Where setup.py installs the built `lfortran` binary (cmake
# CMAKE_INSTALL_PREFIX). Created + chowned here so the unprivileged `fuzz`
# user can `ninja install` into it at setup time.
ENV PATH="/opt/lfortran/bin:${PATH}"
RUN mkdir -p /opt/lfortran

# Setup User
RUN useradd -m fuzz && \
    echo "fuzz:fuzz" | chpasswd && \
    usermod -aG sudo fuzz

RUN chown -R fuzz:fuzz /opt/lfortran

# Set up the workspace
WORKDIR /home/fuzz/WorkSpace

RUN chown -R fuzz:fuzz /home/fuzz/WorkSpace

# Switch to the user
USER fuzz

# Define the default command
CMD ["/bin/bash"]
