import multiprocessing
import os
import shutil
import subprocess


def _run(cmd_str, cwd=None):
    print(f"[run] {cmd_str[:160]}")
    subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)


def setup(project_root):
    """
    Sets up the Clang fuzzing environment (runs inside the fuzz-clang container):
    1. Clones llvm-project's main branch (full shallow clone — a from-source
       build needs llvm/, cmake/, etc., not just clang/test).
    2. Builds clang from source with cmake + ninja and installs it to
       projects/clang/llvm-clang-install/bin/{clang,clang++} (see
       projects/clang/driver.py, which invokes these binaries directly).
    3. Uses clang/test (part of the same checkout) as the seed source.
    """
    print(f"Setting up Clang in: {project_root}")

    src_root = os.path.join(project_root, "llvm-project")
    build_dir = os.path.join(project_root, "llvm-clang-build")
    install_dir = os.path.join(project_root, "llvm-clang-install")
    clang_bin = os.path.join(install_dir, "bin", "clang")
    clangxx_bin = os.path.join(install_dir, "bin", "clang++")
    seed_dir = os.path.join(src_root, "clang", "test")

    if os.path.exists(clang_bin):
        print(f"clang already built at {clang_bin}")
    else:
        llvm_cmake = os.path.join(src_root, "llvm", "CMakeLists.txt")
        if os.path.exists(src_root) and not os.path.exists(llvm_cmake):
            # Old checkouts sparse-cloned only clang/test (no from-source
            # build). A build needs llvm/, cmake/, etc. — re-clone in full.
            print(f"{src_root} is missing llvm/ (old clang/test-only checkout) — re-cloning full source.")
            shutil.rmtree(src_root)

        if not os.path.exists(src_root):
            print("Cloning llvm-project (main branch)...")
            _run(
                "git clone --depth=1 --branch main "
                f"https://github.com/llvm/llvm-project.git {src_root}"
            )

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(install_dir, exist_ok=True)

        # Cap parallelism: linking clang is memory-hungry, so building on a
        # 16-core/32GB-class machine with full -j<nproc> risks OOM.
        jobs = max(1, min(multiprocessing.cpu_count(), 8))

        build_script = f"""
set -e
cmake -S {src_root}/llvm -B {build_dir} -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DLLVM_ENABLE_PROJECTS=clang \\
    -DLLVM_ENABLE_ASSERTIONS=ON \\
    -DLLVM_TARGETS_TO_BUILD=host \\
    -DLLVM_OPTIMIZED_TABLEGEN=ON \\
    -DLLVM_PARALLEL_LINK_JOBS=2 \\
    -DCMAKE_INSTALL_PREFIX={install_dir}
cmake --build {build_dir} --target install-clang -- -j{jobs}
"""
        _run(build_script)

    if not os.path.exists(clang_bin):
        raise RuntimeError(f"Clang build failed: {clang_bin} not found")

    if not os.path.exists(clangxx_bin):
        os.symlink("clang", clangxx_bin)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found — seed collection will find 0 seeds.")
    else:
        print(f"Clang setup complete. clang: {clang_bin}, seeds: {seed_dir}")
