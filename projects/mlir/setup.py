import os
import shutil
import sys
import subprocess
import multiprocessing


#: Sanitizers to build the compiler *itself* with (LLVM_USE_SANITIZER).
#: Without it a use-after-free or an out-of-bounds read inside the compiler
#: only becomes a finding when it happens to segfault, and any
#: ASAN_OPTIONS/UBSAN_OPTIONS the driver exports do nothing at all.
#: Assertions catch broken invariants the developers thought to check for;
#: this catches the memory errors nobody wrote a check for. Costs roughly
#: 2x build time and 2-4x compile time — set FFL_MLIR_SANITIZERS=none to opt out.
DEFAULT_SANITIZERS = "Address;Undefined"


def _can_sanitize(cc):
    """Whether this compiler can actually build sanitized code.

    Existing is not enough: a clang built from `LLVM_ENABLE_PROJECTS` alone
    has no compiler-rt, so it has no sanitizer headers or runtime and fails
    only once something includes <sanitizer/asan_interface.h>, thousands of
    objects into the build. gcc ships those headers with libasan and
    handles Address;Undefined, so it is a real fallback."""
    if not cc:
        return False
    src = "#include <sanitizer/asan_interface.h>\nint main(void){return 0;}\n"
    try:
        r = subprocess.run([cc, "-fsanitize=address,undefined",
                            # the flags LLVM's cmake adds; gcc rejects them
                            "-fno-sanitize=function",
                            "-x", "c", "-", "-o", os.devnull],
                           input=src, text=True, capture_output=True,
                           timeout=120)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _sanitizer_toolchain():
    """The first (cc, cxx) that passes _can_sanitize, or (None, None)."""
    pairs = []
    c, x = shutil.which("clang"), shutil.which("clang++")
    if c and x:
        pairs.append((c, x))
    for c, x in pairs:
        if _can_sanitize(c):
            return c, x
        print(f"  ({c} cannot build sanitized code; trying the next)")
    return None, None


def _sanitizers():
    value = os.environ.get("FFL_MLIR_SANITIZERS", DEFAULT_SANITIZERS).strip()
    return "" if value.lower() in ("", "none", "off", "0") else value


def _sanitizer_opts():
    """The toolchain and sanitizer cmake flags, as one block so the compiler
    is chosen in exactly one place.

    LLVM_USE_SANITIZER is only really supported building with clang, so a
    sanitized build needs clang; without one, fall back to gcc and an
    uninstrumented build rather than configuring something broken."""
    san = _sanitizers()
    cc = cxx = None
    if san:
        cc, cxx = _sanitizer_toolchain()
        if not (cc and cxx):
            print("  (no compiler here can build sanitized code: "
                  "building without sanitizers)")
            san = ""
    if not san:
        cc, cxx = "gcc", "g++"
    opts = ["-DCMAKE_C_COMPILER=%s" % cc, "-DCMAKE_CXX_COMPILER=%s" % cxx]
    if san:
        opts.insert(0, '-DLLVM_USE_SANITIZER="%s"' % san)
    return " \\\n    ".join(opts) + " \\\n    "


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

    # 2. Clone llvm-project
    if not os.path.exists(src_root):
        print("Cloning llvm-project...")
        _run(f"git clone --depth=1 https://github.com/llvm/llvm-project.git {src_root}")

    # 3. Build from source
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(install_dir, exist_ok=True)

    build_script = f"""
set -e
echo "Configuring CMake..."
cmake -S {src_root}/llvm -B {build_dir} -G Ninja \\
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
    -DLLVM_ENABLE_PROJECTS=mlir \\
    -DLLVM_ENABLE_ASSERTIONS=ON \\
    {_sanitizer_opts()}    -DLLVM_ENABLE_RTTI=ON \\
    -DLLVM_OPTIMIZED_TABLEGEN=ON \\
    -DLLVM_TARGETS_TO_BUILD=host \\
    -DLLVM_BUILD_TOOLS=ON \\
    -DCMAKE_INSTALL_PREFIX={install_dir}
echo "Building mlir-opt (parallel: 4)..."
cmake --build {build_dir} --target mlir-opt -- -j4
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
