import os
import shutil
import subprocess

BOOTSTRAP_GO_VERSION = "1.24.6"


def setup(project_root):
    """
    Sets up the Go fuzzing environment (runs inside the ffl-go container):
    1. Clones the Go repository if needed.
    2. Downloads a bootstrap Go toolchain and builds Go from source.
    3. Collects .go seed files from the Go standard library tests.
    """
    print(f"Setting up Go in: {project_root}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    go_dir = os.path.join(project_root, "go")
    go_bin = os.path.join(go_dir, "bin", "go")

    # 1. Check if binary already exists
    if os.path.exists(go_bin):
        print(f"Go binary already exists at {go_bin}")
        _run(f"{go_bin} version")
    else:
        # 2. Clone Go repository
        if not os.path.exists(go_dir):
            _run(f"git clone https://github.com/golang/go.git {go_dir}")

        # 3. Build Go from source
        bootstrap_dir = os.path.join(project_root, "go-bootstrap")
        build_script = f"""
set -e
ARCH=$(uname -m)
case "$ARCH" in x86_64) GOARCH=amd64;; aarch64) GOARCH=arm64;; *) GOARCH=$ARCH;; esac
TARBALL="go{BOOTSTRAP_GO_VERSION}.linux-$GOARCH.tar.gz"
URL="https://go.dev/dl/$TARBALL"
mkdir -p {bootstrap_dir}
echo "Downloading bootstrap Go from $URL..."
curl -fL "$URL" -o {bootstrap_dir}/$TARBALL
tar -C {bootstrap_dir} -xzf {bootstrap_dir}/$TARBALL
rm {bootstrap_dir}/$TARBALL
export GOROOT_BOOTSTRAP={bootstrap_dir}/go
echo "Building Go from source..."
cd {go_dir}/src && bash make.bash
echo "Go built successfully."
"""
        _run(build_script)
        _run(f"{go_bin} version")

    # 4. Collect .go seed files
    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    seed_source_dirs = [
        os.path.join(go_dir, "test"),
        os.path.join(go_dir, "src", "cmd", "compile", "internal"),
        os.path.join(go_dir, "src", "go", "parser", "testdata"),
        os.path.join(go_dir, "src", "go", "printer", "testdata"),
    ]
    collected = skipped = 0
    for source_dir in seed_source_dirs:
        if not os.path.exists(source_dir):
            continue
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith(".go"):
                    continue
                src_path = os.path.join(root, fname)
                rel = os.path.relpath(src_path, go_dir)
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

    print(f"Collected {collected} new .go seeds ({skipped} already existed).")
    print("Go setup complete.")
