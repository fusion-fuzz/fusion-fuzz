import os
import sys
import subprocess
import multiprocessing
import shutil


def setup(project_root):
    """
    Sets up the MLIR environment (runs inside the ffl-mlir container):
    1. Clones llvm-project if needed (for seed sources).
    2. Symlinks the pre-built mlir-opt from /opt/mlir-install if available.
    3. Falls back to building mlir-opt with cmake + ninja.
    """
    print(f"Setting up MLIR in: {project_root}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:120]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    src_root = os.path.join(project_root, "llvm-project")
    build_dir = os.path.join(project_root, "llvm-mlir-build")
    install_dir = os.path.join(project_root, "llvm-mlir-install")
    mlir_opt = os.path.join(install_dir, "bin", "mlir-opt")

    # 1. Already done
    if os.path.exists(mlir_opt):
        print(f"mlir-opt already exists at {mlir_opt}")
        print("MLIR setup complete.")
        return

    # 2. Clone llvm-project for seed sources (projects/mlir/llvm-project/mlir/test)
    if not os.path.exists(src_root):
        print("Cloning llvm-project (needed for seed sources)...")
        _run(f"git clone --depth=1 https://github.com/llvm/llvm-project.git {src_root}")

    # 3. Use mlir-opt pre-built in the Docker image
    docker_mlir_opt = "/opt/mlir-install/bin/mlir-opt"
    system_mlir_opt = (
        docker_mlir_opt if os.path.exists(docker_mlir_opt)
        else shutil.which("mlir-opt")
    )
    if system_mlir_opt:
        print(f"Found pre-built mlir-opt at {system_mlir_opt}, creating symlink...")
        os.makedirs(os.path.join(install_dir, "bin"), exist_ok=True)
        os.symlink(system_mlir_opt, mlir_opt)
        print("MLIR setup complete.")
        return

    # 4. Fallback: build from source
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(install_dir, exist_ok=True)

    jobs = multiprocessing.cpu_count()
    build_script = f"""
set -e
echo "Configuring CMake..."
cmake -S {src_root}/llvm -B {build_dir} -G Ninja \\
    -DCMAKE_C_COMPILER=gcc \\
    -DCMAKE_CXX_COMPILER=g++ \\
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
    -DLLVM_ENABLE_PROJECTS=mlir \\
    -DLLVM_ENABLE_ASSERTIONS=ON \\
    -DLLVM_ENABLE_RTTI=ON \\
    -DLLVM_OPTIMIZED_TABLEGEN=ON \\
    -DLLVM_TARGETS_TO_BUILD=host \\
    -DLLVM_BUILD_TOOLS=ON \\
    -DCMAKE_INSTALL_PREFIX={install_dir}
echo "Building mlir-opt (parallel: {jobs})..."
cmake --build {build_dir} --target mlir-opt -- -j {jobs}
mkdir -p {install_dir}/bin
cp {build_dir}/bin/mlir-opt {install_dir}/bin/
"""
    try:
        _run(build_script)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

    print("MLIR setup complete.")
    print(f"mlir-opt available at: {mlir_opt}")
