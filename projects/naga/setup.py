import os
import shutil
import subprocess
import sys


def setup(project_root):
    """
    Sets up the Naga environment (runs inside the ffl-naga container):
    1. Clones wgpu repository if needed.
    2. Builds naga-cli with cargo.
    3. Copies binary to project root.
    4. Collects .wgsl seed files.
    """
    print(f"[Naga] Setting up in: {project_root}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    wgpu_dir = os.path.join(project_root, "wgpu")
    final_bin = os.path.join(project_root, "naga")

    # 1. Check if binary already exists
    if os.path.exists(final_bin):
        print(f"[Naga] naga binary already present at {final_bin}")
    else:
        # 2. Clone wgpu
        if not os.path.exists(wgpu_dir):
            print("[Naga] Cloning wgpu repository...")
            _run(f"git clone https://github.com/gfx-rs/wgpu.git {wgpu_dir}")
        else:
            print("[Naga] wgpu repository already exists, updating...")
            _run("git fetch origin && git checkout origin/trunk", cwd=wgpu_dir)

        # 3. Build naga-cli
        print("[Naga] Building naga-cli...")
        build_cmd = (
            "RUSTUP_TOOLCHAIN=1.93.1-x86_64-unknown-linux-gnu "
            "cargo build -p naga-cli --release -j2"
        )
        try:
            _run(build_cmd, cwd=wgpu_dir)
        except subprocess.CalledProcessError as e:
            print(f"[Naga] Build failed: {e}")
            sys.exit(1)

        # 4. Copy binary
        target_bin = os.path.join(wgpu_dir, "target", "release", "naga")
        if os.path.exists(target_bin):
            shutil.copy2(target_bin, final_bin)
            print(f"[Naga] Binary copied to: {final_bin}")
        else:
            print(f"[Naga] Error: Binary not found at {target_bin}")
            sys.exit(1)

    # 5. Collect .wgsl seed files
    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    seed_source_dirs = [
        os.path.join(wgpu_dir, "naga", "tests"),
        os.path.join(wgpu_dir, "examples"),
        os.path.join(wgpu_dir, "tests"),
        os.path.join(wgpu_dir, "benches"),
        os.path.join(wgpu_dir, "wgpu-core"),
        os.path.join(wgpu_dir, "wgpu-hal"),
        os.path.join(wgpu_dir, "naga", "src"),
    ]

    collected = skipped = 0
    for source_dir in seed_source_dirs:
        if not os.path.exists(source_dir):
            continue
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith(".wgsl"):
                    continue
                src_path = os.path.join(root, fname)
                rel = os.path.relpath(src_path, wgpu_dir)
                safe_name = rel.replace(os.sep, "__")
                dst_path = os.path.join(seeds_dir, safe_name)
                if os.path.exists(dst_path):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                    collected += 1
                except Exception as e:
                    print(f"[Naga] Warning: could not copy {src_path}: {e}")

    print(f"[Naga] Collected {collected} new .wgsl seeds ({skipped} already existed).")
    print("[Naga] Setup complete!")
