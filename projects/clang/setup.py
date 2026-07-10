import os
import shutil
import subprocess


def _run(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def setup(project_root):
    """
    Sets up Clang fuzzing seeds (runs inside the ffl-clang container, which
    already ships a system clang/clang++ toolchain — see Dockerfile).

    Does NOT build LLVM from source: the container's prebuilt clang/clang++
    are used directly (see projects/clang/driver.py). This only sparse-clones
    llvm-project to pull in clang's own regression-test corpus (clang/test)
    as the seed source, mirroring how projects/mlir/setup.py sources seeds
    from mlir/test.
    """
    print(f"Setting up Clang in: {project_root}")

    for tool in ("clang", "clang++"):
        found = shutil.which(tool)
        if found:
            print(f"Found {tool}: {found}")
        else:
            print(f"Warning: '{tool}' not found in PATH. "
                  f"Install it or update projects/clang/Dockerfile.")

    src_root = os.path.join(project_root, "llvm-project")
    seed_dir = os.path.join(src_root, "clang", "test")

    if os.path.exists(seed_dir):
        print(f"Seed source already present at {seed_dir}")
        return

    if not os.path.exists(src_root):
        print("Sparse-cloning llvm-project (clang/test only)...")
        _run([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
            "https://github.com/llvm/llvm-project.git", src_root,
        ])
        _run(["git", "sparse-checkout", "set", "clang/test"], cwd=src_root)
        _run(["git", "checkout"], cwd=src_root)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found after clone — seed collection will find 0 seeds.")
    else:
        print(f"Clang setup complete. Seed source: {seed_dir}")
