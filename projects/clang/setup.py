import glob
import multiprocessing
import os
import shutil
import subprocess


#: Sanitizers to build the compiler *itself* with (LLVM_USE_SANITIZER).
#: Without it a use-after-free or an out-of-bounds read inside the compiler
#: only becomes a finding when it happens to segfault, and any
#: ASAN_OPTIONS/UBSAN_OPTIONS the driver exports do nothing at all.
#: Assertions catch broken invariants the developers thought to check for;
#: this catches the memory errors nobody wrote a check for. Costs roughly
#: 2x build time and 2-4x compile time — set FFL_CLANG_SANITIZERS=none to opt out.
DEFAULT_SANITIZERS = "Address;Undefined"


def _sanitizers():
    value = os.environ.get("FFL_CLANG_SANITIZERS", DEFAULT_SANITIZERS).strip()
    return "" if value.lower() in ("", "none", "off", "0") else value


def _sanitizer_toolchain():
    """A (cc, cxx) that can actually build sanitized code, or (None, None).

    Candidates in order: a clang on PATH, a clang this adapter built
    earlier (FFL_CLANG_BOOTSTRAP overrides its location, which is what to
    point at a copy of the previous install when this build overwrites it
    in place), then gcc.

    Each candidate is probed rather than assumed, and gcc is deliberately
    not among them. LLVM_USE_SANITIZER makes LLVM's cmake emit
    clang-specific flags — `-fno-sanitize=function`, `-fsanitize-blacklist=`
    — that gcc rejects outright, so a gcc "sanitized" build dies a few
    objects in even though gcc itself handles -fsanitize=address perfectly
    well. The clang this adapter builds is no good either: it comes from
    `LLVM_ENABLE_PROJECTS=clang` with no compiler-rt, so it has no sanitizer
    headers at all. What is needed is a clang shipping compiler-rt, which on
    Ubuntu means the `clang` package.
    """
    candidates = []
    cc, cxx = shutil.which("clang"), shutil.which("clang++")
    if cc and cxx:
        candidates.append((cc, cxx))
    roots = []
    if os.environ.get("FFL_CLANG_BOOTSTRAP"):
        roots.append(os.environ["FFL_CLANG_BOOTSTRAP"])
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "llvm-clang-install"))
    for root in roots:
        c = os.path.join(root, "bin", "clang")
        x = os.path.join(root, "bin", "clang++")
        if os.access(c, os.X_OK) and os.access(x, os.X_OK):
            candidates.append((c, x))
    for c, x in candidates:
        if _can_sanitize(c):
            print(f"  (sanitized build using {c})")
            return c, x
        print(f"  ({c} cannot build sanitized code; trying the next)")
    return None, None


def _can_sanitize(cc):
    """Whether this compiler can actually build sanitized code. Probing
    costs one compile and turns a late build failure into a clear message
    here. See _sanitizer_toolchain."""
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


def _sanitizer_opts():
    """cmake flags selecting a sanitized build, plus the host toolchain it
    needs."""
    san = _sanitizers()
    if not san:
        return ""
    cc, cxx = _sanitizer_toolchain()
    if not (cc and cxx):
        print("  (no compiler here can build sanitized code: "
              "building without sanitizers)")
        return ""
    return ('-DLLVM_USE_SANITIZER="%s" \\\n    '
            '-DCMAKE_C_COMPILER=%s \\\n    '
            '-DCMAKE_CXX_COMPILER=%s \\\n    ' % (san, cc, cxx))


def _run(cmd_str, cwd=None):
    print(f"[run] {cmd_str[:160]}")
    subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)


def _ensure_resource_headers(clang_bin, build_dir):
    """Make sure <resource-dir>/include exists, copying from the build tree
    if the install missed it. clang resolves stddef.h & friends only there."""
    try:
        res_dir = subprocess.run([clang_bin, "-print-resource-dir"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        print(f"Warning: could not query clang resource dir: {e}")
        return
    if not res_dir:
        return
    target = os.path.join(res_dir, "include")
    if os.path.isdir(target) and os.listdir(target):
        return
    # Same version-suffixed layout under the build tree.
    src = os.path.join(build_dir, "lib", "clang", os.path.basename(res_dir), "include")
    if not os.path.isdir(src):
        cands = glob.glob(os.path.join(build_dir, "lib", "clang", "*", "include"))
        src = cands[0] if cands else None
    if not src:
        print(f"Warning: clang resource headers not found under {build_dir}; "
              "seeds including <stddef.h> will fail to compile.")
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copytree(src, target, dirs_exist_ok=True)
    print(f"Installed clang resource headers: {src} -> {target}")


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

        # Backends to build. "host" (X86 only) is cheaper, but then clang
        # never generates the target intrinsic headers (arm_neon.h,
        # arm_sve.h, riscv_vector.h, ...) and rejects every seed that
        # includes one or asks for that triple on the RUN line — 3698 of
        # 31251 seeds in the clang/test corpus, ~21% of all executions in a
        # measured run. Building these five covers the corpus's own
        # -triple distribution (ppc64le, riscv64, aarch64, i386, x86_64).
        targets = os.environ.get("FFL_LLVM_TARGETS", "X86;AArch64;ARM;RISCV;PowerPC")

        build_script = f"""
set -e
cmake -S {src_root}/llvm -B {build_dir} -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DLLVM_ENABLE_PROJECTS=clang \\
    -DLLVM_ENABLE_ASSERTIONS=ON \\
    {_sanitizer_opts()}    -DLLVM_TARGETS_TO_BUILD="{targets}" \\
    -DLLVM_OPTIMIZED_TABLEGEN=ON \\
    -DLLVM_PARALLEL_LINK_JOBS=2 \\
    -DCMAKE_INSTALL_PREFIX={install_dir}
cmake --build {build_dir} --target install-clang -- -j{jobs}
# install-clang alone installs only the driver binary. Clang's own resource
# headers (stddef.h, limits.h, stdarg.h, and every target intrinsic header:
# arm_neon.h, riscv_vector.h, immintrin.h, ...) ship as a separate target.
# Without them, every seed that includes one of those dies on
# "fatal error: 'stddef.h' file not found" before the frontend does any of
# the work we are fuzzing — that single header alone accounted for ~31% of
# all executions in a measured run.
cmake --build {build_dir} --target install-clang-resource-headers -- -j{jobs}
"""
        _run(build_script)

    if not os.path.exists(clang_bin):
        raise RuntimeError(f"Clang build failed: {clang_bin} not found")

    if not os.path.exists(clangxx_bin):
        os.symlink("clang", clangxx_bin)

    # Guard against an install tree built before the resource-headers target
    # was added above (or a partial install): copy them straight out of the
    # build tree rather than forcing a full rebuild.
    _ensure_resource_headers(clang_bin, build_dir)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found — seed collection will find 0 seeds.")
    else:
        print(f"Clang setup complete. clang: {clang_bin}, seeds: {seed_dir}")
