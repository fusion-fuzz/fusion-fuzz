import os
import subprocess


def _run(cmd, cwd=None):
    print(f"[run] {cmd[:160]}")
    subprocess.run(["sh", "-c", cmd], check=True, cwd=cwd)


def setup(project_root):
    """
    Clone wgpu's trunk branch for Naga WGSL seeds. The Dockerfile installs
    naga-cli; setup.py only prepares the local seed tree consumed by parser.py.
    """
    src_root = os.path.join(project_root, "wgpu")
    seed_root = os.path.join(src_root, "naga", "tests")
    if not os.path.exists(src_root):
        _run(
            "git clone --depth=1 --branch trunk "
            f"https://github.com/gfx-rs/wgpu.git {src_root}"
        )
    elif not os.path.exists(os.path.join(src_root, ".git")):
        raise RuntimeError(f"{src_root} exists but is not a git checkout")
    else:
        _run("git fetch --depth=1 origin trunk && git checkout trunk && git pull --ff-only", cwd=src_root)

    if not os.path.exists(seed_root):
        print(f"Warning: Naga WGSL seed root not found at {seed_root}")
    else:
        print(f"Naga setup complete. Seeds: {seed_root}")
