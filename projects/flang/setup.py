import json
import multiprocessing
import os
import shutil
import subprocess


#: Backends to build. flang's own test corpus is full of target-specific
#: tests (PowerPC vector intrinsics, RISC-V and AArch64 triples on the RUN
#: lines), and projects/flang/driver.py draws a `--target=` from this same
#: set — a triple whose backend was not built is rejected before the
#: frontend does any of the work being fuzzed.
DEFAULT_TARGETS = "X86;AArch64;PowerPC;RISCV"

#: Sanitizers to build the compiler *itself* with (LLVM_USE_SANITIZER).
#: This is the single biggest oracle upgrade available: without it, a
#: use-after-free or an out-of-bounds read inside flang only turns into a
#: finding when it happens to segfault, and the ASAN_OPTIONS/UBSAN_OPTIONS
#: the driver already exports do nothing at all. Costs roughly 2x build
#: time and 2-4x compile time — set FFL_FLANG_SANITIZERS=none to opt out.
DEFAULT_SANITIZERS = "Address;Undefined"

BUILD_INFO = "build-info.json"


def _run(cmd_str, cwd=None):
    print(f"[run] {cmd_str[:200]}")
    subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)


def _git_commit(src_root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=src_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _find_flang(*dirs):
    """flang's tool target was renamed from `flang-new` to `flang` in LLVM 20;
    accept either so an older checkout still works."""
    for d in dirs:
        for name in ("flang", "flang-new"):
            path = os.path.join(d, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def _sanitizers():
    value = os.environ.get("FFL_FLANG_SANITIZERS", DEFAULT_SANITIZERS).strip()
    return "" if value.lower() in ("", "none", "off", "0") else value


def _host_toolchain():
    """cmake options selecting the compiler and linker to build LLVM with.

    Prefer clang and lld when the container already has them: LLVM_USE_SANITIZER
    is only really supported building with clang, clang builds LLVM
    substantially faster than gcc, and lld links it in a fraction of the
    time and memory that GNU ld needs — which matters here because an
    instrumented flang is a very large link.
    """
    opts = []
    cc, cxx = shutil.which("clang"), shutil.which("clang++")
    if cc and cxx:
        opts += [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]
        # LLVM_ENABLE_LLD passes -fuse-ld=lld; clang finds ld.lld next to
        # itself, so probe rather than assume it is on PATH.
        probe = subprocess.run(
            [cc, "-fuse-ld=lld", "-xc", "-", "-o", os.devnull],
            input="int main(void){return 0;}", text=True,
            capture_output=True)
        if probe.returncode == 0:
            opts.append("-DLLVM_ENABLE_LLD=ON")
    return opts


#: Module sources needing a PowerPC target (vector/MMA intrinsics), and the
#: ones that are CUDA Fortran. Everything else in flang-rt/lib/runtime
#: compiles for the host with no extra flags.
_PPC_MODULES = ("__ppc_types", "__ppc_intrinsics", "mma")
_CUDA_MODULES = ("__cuda_device", "cooperative_groups", "cuda_runtime_api",
                 "cudadevice")

_MODULE_CANARY = """program p
  use iso_fortran_env
  use iso_c_binding
  use ieee_arithmetic
  use omp_lib
  implicit none
  integer(int64) :: n
  type(team_type) :: t
  n = 1_int64
end program p
"""


#: openmp/module/*.var placeholders. These feed the version constants
#: omp_get_* reports; they do not affect any semantics flang checks, so the
#: values openmp/CMakeLists.txt itself sets are used verbatim.
_LIBOMP_SUBSTITUTIONS = {
    "@LIBOMP_VERSION_MAJOR@": "5",
    "@LIBOMP_VERSION_MINOR@": "0",
    "@LIBOMP_VERSION_BUILD@": "0",
    "@LIBOMP_OMP_YEAR_MONTH@": "201611",
    "@LIBOMP_BUILD_DATE@": "No_Timestamp",
}


def _build_openmp_modules(binary, src_root, module_dir, env):
    """Also generate omp_lib / omp_lib_kinds.

    They live in openmp/module as CMake-templated `.var` sources, built by
    libomp rather than by flang, so a compiler-only build has none — and
    every seed that USEs omp_lib (flang's own Semantics/OpenMP tree is full
    of them) dies on "Cannot parse module file for module 'omp_lib'".
    Substituting the five version placeholders by hand is far cheaper than
    adding the whole openmp runtime to the build for two .mod files.
    """
    omp_dir = os.path.join(src_root, "openmp", "module")
    if not os.path.isdir(omp_dir):
        return
    staging = os.path.join(module_dir, ".omp-src")
    os.makedirs(staging, exist_ok=True)
    try:
        generated = []
        for name in ("omp_lib.F90", "omp_lib_impl.F90"):
            template = os.path.join(omp_dir, name + ".var")
            if not os.path.exists(template):
                continue
            with open(template) as fh:
                text = fh.read()
            for key, value in _LIBOMP_SUBSTITUTIONS.items():
                text = text.replace(key, value)
            out = os.path.join(staging, name)
            with open(out, "w") as fh:
                fh.write(text)
            generated.append(out)
        # omp_lib.F90 declares the interfaces (omp_lib.mod, omp_lib_kinds.mod);
        # omp_lib_impl.F90 holds the separate module procedure bodies and has
        # to come second.
        for source in generated:
            subprocess.run([binary, "-fsyntax-only", "-fopenmp",
                            "-module-dir", module_dir, source],
                           capture_output=True, env=env, cwd=module_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build_intrinsic_modules(binary, src_root, build_dir):
    """Compile flang's intrinsic modules (iso_fortran_env, iso_c_binding,
    ieee_arithmetic, __fortran_builtins, ...) into <bin>/../include/flang.

    Building the `flang` target alone does not produce these: since the
    Fortran runtime moved to the flang-rt *runtime* project, the .mod files
    are a by-product of building flang-rt, which a compiler-only build
    never does. Without them flang answers any seed that USEs a standard
    module — or merely mentions a coarray type — with

        fatal internal error: INTERNAL: The __fortran_builtins module was
        not found, and the type '__builtin_team_type' was required

    which matches the crash oracle, so the missing modules cost validity
    *and* manufacture findings. (The same trap as clang's separate
    install-clang-resource-headers target.)

    Compiling the module sources directly is what flang-rt's own
    `flang-rt-mod` target does, minus building the runtime library itself
    with a sanitizer-instrumented compiler.
    """
    runtime_dir = os.path.join(src_root, "flang-rt", "lib", "runtime")
    if not os.path.isdir(runtime_dir):
        # Pre-flang-rt checkouts keep the modules in flang/module and build
        # them as part of the compiler.
        print("No flang-rt/lib/runtime — assuming this checkout builds its "
              "intrinsic modules with the compiler.")
        return

    # getIntrinsicDir() in flang/lib/Frontend/CompilerInvocation.cpp resolves
    # exactly this path relative to the executable, and it is the fallback
    # -fc1 uses when no -fintrinsic-modules-path was passed.
    module_dir = os.path.join(build_dir, "include", "flang")
    os.makedirs(module_dir, exist_ok=True)

    env = dict(os.environ)
    # A sanitizer-instrumented flang otherwise fails these on LeakSanitizer
    # reports from the compiler itself, which has nothing to do with the
    # module being compiled.
    env["ASAN_OPTIONS"] = "detect_leaks=0"

    sources = sorted(f[:-4] for f in os.listdir(runtime_dir) if f.endswith(".f90"))
    print(f"Generating intrinsic modules from {len(sources)} sources -> {module_dir}")

    def flags_for(name):
        if name in _PPC_MODULES:
            return ["--target=powerpc64le-unknown-linux-gnu"]
        if name in _CUDA_MODULES:
            return ["--offload-host-only", "-xcuda"]
        return []

    # Two passes: a module that USEs one compiled later in the list fails
    # the first time and succeeds the second. Individual failures are not
    # fatal — the canary below decides whether the result is usable.
    for _ in range(2):
        for name in sources:
            subprocess.run(
                [binary, "-fsyntax-only", *flags_for(name),
                 "-module-dir", module_dir,
                 os.path.join(runtime_dir, name + ".f90")],
                capture_output=True, env=env, cwd=module_dir,
            )

    _build_openmp_modules(binary, src_root, module_dir, env)

    count = len([f for f in os.listdir(module_dir) if f.endswith(".mod")])
    canary = os.path.join(module_dir, "ffl-module-canary.f90")
    with open(canary, "w") as fh:
        fh.write(_MODULE_CANARY)
    try:
        for mode in ([], ["-fc1"]):
            probe = subprocess.run([binary, *mode, "-fsyntax-only", canary],
                                   capture_output=True, text=True, env=env,
                                   cwd=module_dir)
            if probe.returncode != 0:
                raise RuntimeError(
                    f"Intrinsic modules unusable via {'flang -fc1' if mode else 'the driver'} "
                    f"({count} .mod files in {module_dir}):\n{probe.stderr[:800]}")
    finally:
        os.unlink(canary)
    print(f"{count} intrinsic modules generated and verified.")


def _write_build_info(project_root, binary, src_root, sanitizers, expensive):
    """Record how the compiler was built, for projects/flang/driver.py.

    The driver has to know: an ASan-instrumented binary reserves terabytes
    of *virtual* address space for its shadow map, so the `ulimit -v` the
    driver uses to cap memory would kill it at startup, before it ever
    reads the test case. The driver switches to ASan's own RSS limit when
    this file says the build is sanitized."""
    info = {
        "binary": binary,
        "sanitizers": sanitizers,
        "assertions": True,
        "expensive_checks": expensive,
        "commit": _git_commit(src_root),
        "source": src_root,
    }
    path = os.path.join(project_root, BUILD_INFO)
    with open(path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Wrote {path}")
    return info


def setup(project_root):
    """
    Builds flang from llvm-project trunk, with every bug oracle that is
    cheap enough to leave on permanently.

    Why from source rather than the distro package the container used to
    ship: a release build has no assertions, so flang's own internal
    invariants (`CHECK(...)`, the MLIR and LLVM IR verifiers, LLVM's
    ABI-breaking debug checks) are compiled out, and the only crashes
    reachable are the ones that happen to segfault. Assertions are where
    almost every reportable flang bug surfaces — of 29 findings triaged on
    the packaged build, 15 were assertion failures that a release build
    would have silently miscompiled or ignored.

    What gets turned on:

      LLVM_ENABLE_ASSERTIONS       flang's CHECK()/DIE(), plus the MLIR and
                                   LLVM IR verifiers after every pass.
      LLVM_ABI_BREAKING_CHECKS     LLVM's own internal consistency checks.
      MLIR_ENABLE_EXPENSIVE_...    MLIR rewrite-pattern API misuse checks;
                                   flang lowers through MLIR, so this
                                   covers the HLFIR/FIR pipeline.
      LLVM_USE_SANITIZER           ASan+UBSan *in the compiler itself*
                                   (see DEFAULT_SANITIZERS).

    Environment overrides:
      FFL_LLVM_TARGETS             backends to build (default: X86;AArch64;PowerPC;RISCV)
      FFL_FLANG_SANITIZERS         "none" to build without sanitizers
      FFL_FLANG_EXPENSIVE_CHECKS   "1" to add LLVM_ENABLE_EXPENSIVE_CHECKS
                                   (very slow — data-structure invariant
                                   checks on every operation; worth a
                                   dedicated campaign, not the default)
      FFL_BUILD_JOBS               parallelism (default: min(nproc, 8))

    Expect a multi-hour first build and ~100GB of disk with sanitizers on.
    The seed corpus is flang/test from the same checkout, so the seeds
    always match the compiler being fuzzed.
    """
    print(f"Setting up Flang in: {project_root}")

    src_root = os.path.join(project_root, "llvm-project")
    build_dir = os.path.join(project_root, "llvm-flang-build")
    install_dir = os.path.join(project_root, "llvm-flang-install")
    seed_dir = os.path.join(src_root, "flang", "test")

    sanitizers = _sanitizers()
    expensive = os.environ.get("FFL_FLANG_EXPENSIVE_CHECKS", "").strip() in ("1", "on", "ON", "true")

    binary = _find_flang(os.path.join(install_dir, "bin"), os.path.join(build_dir, "bin"))
    if binary:
        print(f"flang already built at {binary}")
        _build_intrinsic_modules(binary, src_root, os.path.dirname(os.path.dirname(binary)))
        _write_build_info(project_root, binary, src_root, sanitizers, expensive)
    else:
        llvm_cmake = os.path.join(src_root, "llvm", "CMakeLists.txt")
        if not os.path.exists(llvm_cmake):
            # Either there is no checkout, or it is the old sparse
            # flang/test-only one, which is enough to harvest seeds but not
            # to build anything. Clone alongside and swap only once the new
            # checkout is complete: this tree is also the seed source, and
            # a clone interrupted halfway through would otherwise leave the
            # project with neither a buildable source tree nor seeds.
            staging = src_root + ".new"
            shutil.rmtree(staging, ignore_errors=True)
            print("Cloning llvm-project (main branch)...")
            _run("git clone --depth=1 --branch main "
                 f"https://github.com/llvm/llvm-project.git {staging}")
            if os.path.exists(src_root):
                shutil.rmtree(src_root)
            os.rename(staging, src_root)

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(install_dir, exist_ok=True)

        # Linking LLVM is memory-hungry; a full -j<nproc> on a 16-core box
        # invites the OOM killer.
        jobs = int(os.environ.get("FFL_BUILD_JOBS", 0)) or \
            max(1, min(multiprocessing.cpu_count(), 8))
        targets = os.environ.get("FFL_LLVM_TARGETS", DEFAULT_TARGETS)

        # flang needs clang (it reuses clang's driver library) and mlir
        # (it lowers Fortran through HLFIR/FIR, which are MLIR dialects).
        opts = [
            "-DCMAKE_BUILD_TYPE=Release",
            '-DLLVM_ENABLE_PROJECTS="clang;mlir;flang"',
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_ABI_BREAKING_CHECKS=WITH_ASSERTS",
            "-DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON",
            f'-DLLVM_TARGETS_TO_BUILD="{targets}"',
            "-DLLVM_OPTIMIZED_TABLEGEN=ON",
            # Sanitized objects are far larger, so their links need far more
            # memory; two at once is what puts a 32GB-class box into swap.
            f"-DLLVM_PARALLEL_LINK_JOBS={1 if sanitizers else 2}",
            # A warning promoted to an error in flang's own build is not a
            # finding, it is a broken build against a moving trunk.
            "-DFLANG_ENABLE_WERROR=OFF",
            "-DLLVM_INCLUDE_BENCHMARKS=OFF",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        opts += _host_toolchain()
        if sanitizers:
            opts.append(f'-DLLVM_USE_SANITIZER="{sanitizers}"')
        if expensive:
            opts.append("-DLLVM_ENABLE_EXPENSIVE_CHECKS=ON")

        print(f"Building flang: sanitizers={sanitizers or 'none'}, "
              f"expensive_checks={expensive}, targets={targets}, jobs={jobs}")
        _run("set -e\ncmake -S {src}/llvm -B {build} -G Ninja \\\n    {opts}\n"
             "cmake --build {build} --target flang -- -j{jobs}".format(
                 src=src_root, build=build_dir, jobs=jobs,
                 opts=" \\\n    ".join(opts)))

        binary = _find_flang(os.path.join(build_dir, "bin"))
        if not binary:
            # Pre-LLVM-20 checkouts still call the tool target flang-new.
            _run(f"cmake --build {build_dir} --target flang-new -- -j{jobs}")
            binary = _find_flang(os.path.join(build_dir, "bin"))

        if not binary:
            raise RuntimeError(
                f"Flang build failed: no flang binary under {build_dir}/bin")

        _build_intrinsic_modules(binary, src_root, build_dir)
        _write_build_info(project_root, binary, src_root, sanitizers, expensive)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found — seed collection will find 0 seeds.")
    else:
        print(f"Flang setup complete. flang: {binary}, seeds: {seed_dir}")
