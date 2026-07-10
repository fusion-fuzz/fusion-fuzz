import os
import shutil
import subprocess


def _run(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def setup(project_root):
    """
    Sets up Flang fuzzing seeds (runs inside the ffl-flang container, which
    already ships a system flang toolchain — see Dockerfile).

    Does NOT build LLVM from source: the container's prebuilt flang is used
    directly (see projects/flang/driver.py). This only sparse-clones
    llvm-project to pull in flang's own regression-test corpus
    (flang/test) as the seed source, mirroring projects/clang/setup.py.
    """
    print(f"Setting up Flang in: {project_root}")

    found = shutil.which("flang")
    if found:
        print(f"Found flang: {found}")
    else:
        print("Warning: 'flang' not found in PATH. "
              "Install it or update projects/flang/Dockerfile.")

    src_root = os.path.join(project_root, "llvm-project")
    seed_dir = os.path.join(src_root, "flang", "test")

    if os.path.exists(seed_dir):
        print(f"Seed source already present at {seed_dir}")
        return

    if not os.path.exists(src_root):
        print("Sparse-cloning llvm-project (flang/test only)...")
        _run([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
            "https://github.com/llvm/llvm-project.git", src_root,
        ])
        _run(["git", "sparse-checkout", "set", "flang/test"], cwd=src_root)
        _run(["git", "checkout"], cwd=src_root)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found after clone — seed collection will find 0 seeds.")
    else:
        print(f"Flang setup complete. Seed source: {seed_dir}")
