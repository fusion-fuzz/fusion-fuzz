
    set -e
    echo "Cleaning build directory..."
    rm -rf /workspace/projects/mlir/llvm-mlir-build/*
    echo "Configuring CMake..."
    cmake -S /workspace/projects/mlir/llvm-project/llvm -B /workspace/projects/mlir/llvm-mlir-build -G Ninja \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_TARGETS_TO_BUILD=host \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_USE_SANITIZER="Address;Undefined" \
        -DCMAKE_INSTALL_PREFIX=/workspace/projects/mlir/llvm-mlir-install

    echo "Building MLIR (parallel: 16)..."
    cmake --build /workspace/projects/mlir/llvm-mlir-build -- -j 16

    echo "Installing..."
    cmake --install /workspace/projects/mlir/llvm-mlir-build
    