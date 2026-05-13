#!/usr/bin/env bash
# build-mlir-sanitized.sh
# Build MLIR from llvm-project with AddressSanitizer + UBSan.
set -euo pipefail

### ---- Config (override via env or flags) -------------------------------------
: "${SRC_ROOT:=$PWD/llvm-project}"      # git clone target
: "${BUILD_DIR:=$PWD/llvm-mlir-build}"  # CMake build dir
: "${INSTALL_DIR:=$PWD/llvm-mlir-install}"
: "${JOBS:=$(nproc || sysctl -n hw.ncpu || echo 8)}"
: "${CC:=clang}"
: "${CXX:=clang++}"
: "${BUILD_TYPE:=RelWithDebInfo}"       # or Debug
: "${SANITIZERS:=Address;Undefined}"    # ASan + UBSan via LLVM_USE_SANITIZER
BRANCH="main"

usage () {
  cat <<EOF
Usage: [VAR=val ...] $0 [--clone|--no-clone] [--branch BRANCH] [--clean]
Env overrides:
  SRC_ROOT=$SRC_ROOT
  BUILD_DIR=$BUILD_DIR
  INSTALL_DIR=$INSTALL_DIR
  JOBS=$JOBS
  CC=$CC
  CXX=$CXX
  BUILD_TYPE=$BUILD_TYPE
  SANITIZERS=$SANITIZERS
Examples:
  JOBS=32 BUILD_TYPE=Debug $0 --clone
  SRC_ROOT=~/src/llvm-project BUILD_DIR=/tmp/llvmb $0 --no-clone --clean
EOF
  exit 1
}

DO_CLONE=1
DO_CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone) DO_CLONE=1 ;;
    --no-clone) DO_CLONE=0 ;;
    --clean) DO_CLEAN=1 ;;
    --branch) BRANCH="${2:-main}"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac; shift
done

### ---- Check deps -------------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 2; }; }
need git
need cmake
need ninja
need "$CC"
need "$CXX"

### ---- Clone repo (shallow) ---------------------------------------------------
if [[ $DO_CLONE -eq 1 ]]; then
  if [[ -d "$SRC_ROOT/.git" ]]; then
    echo "[INFO] llvm-project already exists at $SRC_ROOT"
  else
    echo "[INFO] Cloning llvm-project ($BRANCH) into $SRC_ROOT ..."
    git clone --depth=1 --branch "$BRANCH" https://github.com/llvm/llvm-project.git "$SRC_ROOT"
  fi
fi

[[ -d "$SRC_ROOT/llvm" ]] || { echo "ERROR: $SRC_ROOT/llvm not found"; exit 3; }

### ---- Clean build dir if requested ------------------------------------------
if [[ $DO_CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
  echo "[INFO] Cleaning $BUILD_DIR ..."
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

### ---- Configure --------------------------------------------------------------
echo "[INFO] Configuring CMake in $BUILD_DIR ..."
cmake -S "$SRC_ROOT/llvm" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_USE_SANITIZER="$SANITIZERS" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cat <<EONOTE

[NOTE] CMake configured with:
  - BUILD_TYPE         = $BUILD_TYPE
  - LLVM_ENABLE_PROJECTS= mlir
  - LLVM_USE_SANITIZER = $SANITIZERS
  - ASSERTIONS/RTTI    = ON/ON
  - TARGETS_TO_BUILD   = host

EONOTE

### ---- Build (key MLIR tools & libs) -----------------------------------------
echo "[INFO] Building MLIR (parallel: $JOBS) ..."
cmake --build "$BUILD_DIR" -- -j "$JOBS"

# Optionally build specific tools explicitly (uncomment if you want a shorter build)
# cmake --build "$BUILD_DIR" --target mlir-opt mlir-translate mlir-tblgen -- -j "$JOBS"

### ---- Install ---------------------------------------------------------------
echo "[INFO] Installing to $INSTALL_DIR ..."
cmake --install "$BUILD_DIR"

cat <<'EONEXT'

✅ Done.

Run-time tips (recommended for readable reports):
  export ASAN_OPTIONS=abort_on_error=1:detect_leaks=1:strict_init_order=1
  export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

Use the tools from:
  <install>/bin/mlir-opt
  <install>/bin/mlir-translate

If you hit linker errors for sanitizer runtimes:
  - Ensure you’re using Clang/clang++ (not GCC).
  - Make sure your system Clang has compiler-rt installed (most distros do).
  - You can also add: -DLLVM_ENABLE_RUNTIMES="compiler-rt" to build sanitizer runtimes,
    but do NOT sanitize compiler-rt itself (LLVM_USE_SANITIZER already avoids that).

To run MLIR tests (optional, slower):
  cmake --build <build> --target check-mlir -j$(nproc)

EONEXT

