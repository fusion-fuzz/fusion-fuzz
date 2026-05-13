import os
import shutil
import subprocess
import sys


def setup(project_root):
    """
    Sets up the WGSLC fuzzing environment (runs inside the ffl-wgslc container):
    1. Clones the WebKit repository if needed.
    2. Patches OptionsJSCOnly.cmake and generator/main.rb.
    3. Builds wgslc with cmake + ninja.
    4. Collects .wgsl seed files.
    """
    print(f"[WGSLC] Setting up in: {project_root}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    webkit_dir = os.path.join(project_root, "WebKit")
    build_dir = os.path.join(webkit_dir, "build")
    wgslc_bin = os.path.join(project_root, "wgslc")

    # 1. Check if binary already exists
    if os.path.exists(wgslc_bin):
        print(f"[WGSLC] wgslc binary already present at {wgslc_bin}")
    else:
        # 2. Clone WebKit
        if not os.path.exists(webkit_dir):
            _run(f"git clone --depth 1 https://github.com/WebKit/WebKit.git {webkit_dir}")

        # 3. Apply patches
        patch_script = f"""
set -e
OPTIONS_FILE="{webkit_dir}/Source/cmake/OptionsJSCOnly.cmake"
if grep -q 'PATCHED for wgslc' "$OPTIONS_FILE" 2>/dev/null; then
    echo "OptionsJSCOnly.cmake already patched."
else
    sed -i 's/set(ENABLE_WEBGPU OFF)/set(ENABLE_WEBGPU ON)  # PATCHED for wgslc/' "$OPTIONS_FILE" && \\
    echo "Patched OptionsJSCOnly.cmake." || echo "Warning: could not patch OptionsJSCOnly.cmake"
fi
MAIN_RB="{webkit_dir}/Source/WebGPU/WGSL/generator/main.rb"
if [ -f "$MAIN_RB" ] && grep -q 'ARGV.length != 3' "$MAIN_RB"; then
    sed -i 's/ARGV.length != 3/ARGV.length < 2 || ARGV.length > 3/' "$MAIN_RB"
    sed -i 's|output_overloads = ARGV\\[2\\]|output_overloads = ARGV.length == 3 ? ARGV[2] : output_declarations.sub(/TypeDeclarations\\.h$/, "TypeOverloads.h")|' "$MAIN_RB"
    echo "Patched generator/main.rb."
fi
"""
        _run(patch_script)

        # 4. Build wgslc
        build_script = f"""
set -e
mkdir -p {build_dir}
if [ ! -f {build_dir}/CMakeCache.txt ]; then
    echo "Running cmake configuration..."
    cmake -S {webkit_dir} -B {build_dir} \\
        -DPORT=JSCOnly -DCMAKE_BUILD_TYPE=Release -G Ninja \\
        -DENABLE_WEBGPU=ON \\
        -DCMAKE_C_FLAGS=-Wno-unknown-warning-option \\
        -DCMAKE_CXX_FLAGS=-Wno-unknown-warning-option \\
        -DCMAKE_EXE_LINKER_FLAGS=-latomic \\
        -DCMAKE_SHARED_LINKER_FLAGS=-latomic
fi
echo "Building wgslc..."
ninja -C {build_dir} wgslc -j$(nproc)
FOUND=$(find {build_dir} -name wgslc -type f | head -1)
if [ -n "$FOUND" ]; then
    cp "$FOUND" {wgslc_bin}
    chmod +x {wgslc_bin}
    echo "wgslc binary copied to {wgslc_bin}"
else
    echo "Error: wgslc binary not found after build." && exit 1
fi
"""
        _run(build_script)

    # 5. Collect .wgsl seed files
    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    seed_source_dirs = [
        os.path.join(webkit_dir, "LayoutTests"),
        os.path.join(webkit_dir, "Source", "WebGPU"),
        os.path.join(webkit_dir, "Source", "ThirdParty"),
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
                rel = os.path.relpath(src_path, webkit_dir)
                safe_name = rel.replace(os.sep, "__")
                dst_path = os.path.join(seeds_dir, safe_name)
                if os.path.exists(dst_path):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                    collected += 1
                except Exception as e:
                    print(f"[WGSLC] Warning: could not copy {src_path}: {e}")

    print(f"[WGSLC] Collected {collected} new .wgsl seeds ({skipped} already existed).")
    print("[WGSLC] Setup complete!")
