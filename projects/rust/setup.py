import os
import shutil
import subprocess


def setup(project_root):
    """
    Sets up the Rust fuzzing environment (runs inside the ffl-rust container):
    1. Clones rust-lang/rust into /rust-src if needed.
    2. Configures and builds rustc (stage1) from source.
    3. Collects .rs seed files.
    """
    print(f"Setting up Rust in: {project_root}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    rust_src = "/rust-src"
    rustc_bin = f"{rust_src}/build/x86_64-unknown-linux-gnu/stage1/bin/rustc"

    # 1. Check if binary already exists
    if os.path.exists(rustc_bin):
        print(f"rustc already exists at {rustc_bin}")
        _run(f"{rustc_bin} --version")
    else:
        # 2. Clone rust-lang/rust
        _run(
            f"[ -d {rust_src}/.git ] || "
            f"git clone https://github.com/rust-lang/rust.git {rust_src}"
        )

        # 3. Configure and build rustc
        build_script = f"""
set -e
cd {rust_src}
[ -f config.toml ] && rm config.toml
[ -f bootstrap.toml ] && rm bootstrap.toml
echo "Configuring Rust build..."
./configure \\
    --enable-debug \\
    --enable-debug-assertions \\
    --enable-overflow-checks \\
    --set llvm.download-ci-llvm=true \\
    --set change-id=148671
echo "Building rustc (stage1)..."
./x.py build library compiler
echo "rustc build complete."
"""
        _run(build_script)
        _run(f"{rustc_bin} --version")

    # 4. Collect .rs seed files
    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    seed_source_dirs = [
        os.path.join(rust_src, "tests", "ui"),
        os.path.join(rust_src, "tests", "codegen"),
    ]
    collected = skipped = 0
    for source_dir in seed_source_dirs:
        if not os.path.exists(source_dir):
            continue
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith(".rs"):
                    continue
                src_path = os.path.join(root, fname)
                rel = os.path.relpath(src_path, rust_src)
                safe_name = rel.replace(os.sep, "__")
                dst_path = os.path.join(seeds_dir, safe_name)
                if os.path.exists(dst_path):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                    collected += 1
                except Exception as e:
                    print(f"Warning: could not copy {src_path}: {e}")

    print(f"Collected {collected} new .rs seeds ({skipped} already existed).")
    print("Rust setup complete.")
