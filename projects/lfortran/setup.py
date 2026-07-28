import os
import shutil
import subprocess
import sys
import multiprocessing


# Tracks the tip of main by default — set LFORTRAN_REF to a tag/commit/branch
# (e.g. LFORTRAN_REF=v0.63.0) to pin to something reproducible instead.
DEFAULT_REF = "main"

INSTALL_PREFIX = "/opt/lfortran"
LFORTRAN_BIN = os.path.join(INSTALL_PREFIX, "bin", "lfortran")


def _run(cmd, cwd=None, env=None):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _ninja_job_count():
    """
    Memory-aware job count. LFortran's Debug-build translation units (huge
    generated asr.h/ast.h, full debug symbols, no optimization) can each use
    several GB of RAM, and several of the heaviest ones (asr_to_llvm.cpp,
    lfortran.cpp, ast_body_visitor.cpp, ast_symboltable_visitor.cpp) tend to
    get scheduled concurrently near the end of the build — capping
    parallelism by CPU count alone can get `ninja` OOM-killed mid-build.
    Empirically: -j20 on a 24 GB container SIGKILLed at object 116/123, and
    even -j8 (a naive 3 GB/job budget against 24 GB) still SIGKILLed on a
    single-file retry of asr_to_llvm.cpp alone — only -j16 against 48 GB
    completed cleanly. Budget ~4 GB/job for margin; if this still OOMs,
    raise the container's memory limit (`docker run -m`) rather than
    lowering this further, since -j8/24 GB wasn't safe either.
    """
    cpu_jobs = multiprocessing.cpu_count()
    mem_bytes = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                val = f.read().strip()
            if val != "max":
                mem_bytes = int(val)
                break
        except (FileNotFoundError, ValueError):
            continue
    if mem_bytes is None:
        return max(1, cpu_jobs // 2)  # limit unreadable: be conservative
    mem_gb = mem_bytes / (1024 ** 3)
    mem_jobs = max(1, int(mem_gb // 4))
    return max(1, min(cpu_jobs, mem_jobs))


def _llvm_cmake_dir():
    """
    Resolve LLVM_DIR for cmake from whichever llvm-config apt gave us
    (llvm-config-14 on Ubuntu 22.04), rather than hardcoding a path that
    could shift across point releases.
    """
    llvm_config = shutil.which("llvm-config-14") or shutil.which("llvm-config")
    if not llvm_config:
        raise RuntimeError(
            "llvm-config not found on PATH — install llvm-14-dev "
            "(see projects/lfortran/Dockerfile)"
        )
    out = subprocess.run([llvm_config, "--cmakedir"], capture_output=True, text=True, check=True)
    cmake_dir = out.stdout.strip()
    print(f"Using LLVM cmake config at: {cmake_dir}")
    return cmake_dir


def _run_codegen(src_dir):
    """
    Replicates upstream's build0.sh (verified against the actual checked-out
    script — see git history), MINUS the LSP codegen step. We don't pass
    -DLFORTRAN_BUILD_ALL=yes to cmake (which would run build0.sh verbatim)
    because that step unconditionally invokes
    src/server/generator/generate_lsp_code.py, which imports a package
    called `llanguage_server` that does not resolve on PyPI — and we don't
    need it: it only generates LSP server code, and we never pass
    -DWITH_LSP=yes. Everything else build0.sh does is plain stdlib-only
    Python (verified: asdl_cpp.py, wasm_instructions_visitor.py, and
    intrinsic_func_registry_util_gen.py import nothing but os/sys/re/pathlib)
    plus re2c/bison, so we just run those steps directly instead.
    """
    lfortran_dir = os.path.join(src_dir, "src", "lfortran")
    parser_dir = os.path.join(lfortran_dir, "parser")

    _run(["bash", "ci/version.sh"], cwd=src_dir)
    _run(["python3", "src/libasr/asdl_cpp.py", "grammar/AST.asdl", "src/lfortran/ast.h"], cwd=src_dir)
    _run(["python3", "src/libasr/asdl_cpp.py", "src/libasr/ASR.asdl", "src/libasr/asr.h"], cwd=src_dir)
    _run(["python3", "src/libasr/wasm_instructions_visitor.py"], cwd=src_dir)
    _run(["python3", "src/libasr/intrinsic_func_registry_util_gen.py"], cwd=src_dir)
    _run(["re2c", "-W", "-b", "parser/tokenizer.re", "-o", "parser/tokenizer.cpp"], cwd=lfortran_dir)
    _run(["re2c", "-W", "-b", "parser/preprocessor.re", "-o", "parser/preprocessor.cpp"], cwd=lfortran_dir)
    _run(["bison", "-Wall", "-d", "parser.yy"], cwd=parser_dir)


def _clone_and_build(project_root, ref):
    src_dir = os.path.join(project_root, "lfortran-src")

    if not os.path.exists(src_dir):
        print("Cloning lfortran/lfortran (full history — ci/version.sh "
              "needs `git describe` for the version string)...")
        _run(["git", "clone", "https://github.com/lfortran/lfortran.git", src_dir])
    else:
        print(f"Source already cloned at {src_dir}; fetching latest...")
        _run(["git", "fetch", "origin"], cwd=src_dir)

    _run(["git", "checkout", ref], cwd=src_dir)
    if ref == DEFAULT_REF:
        # Track the tip of main on every fresh build, not whatever commit
        # happened to be checked out when the clone was first made.
        _run(["git", "reset", "--hard", f"origin/{ref}"], cwd=src_dir)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=src_dir,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"Building lfortran @ {commit} (ref={ref})")

    _run_codegen(src_dir)

    build_dir = os.path.join(src_dir, "build")
    # Always start from a clean build dir: a stale CMakeCache.txt from a
    # prior failed/different configure (e.g. one that set
    # -DLFORTRAN_BUILD_ALL=yes) would otherwise silently persist and
    # override whatever flags we pass below.
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    llvm_dir = _llvm_cmake_dir()
    njobs = str(_ninja_job_count())
    print(f"Building with -j{njobs} (memory-capped, see _ninja_job_count)")

    # No -DLFORTRAN_BUILD_ALL=yes: the generated sources _run_codegen() just
    #   produced above make cmake see the same state a release tarball
    #   ships with, so it skips straight to configuring the actual build.
    # WITH_LLVM: full backend (execution, -c/-S, --show-llvm/--show-asm),
    #   not just the frontend (--show-ast/--show-asr).
    # WITH_STACKTRACE: symbolized crash locations on SIGSEGV/SIGABRT/ICE
    #   (uses LLVM's own symbolizer + link.h — no extra apt packages).
    # CMAKE_BUILD_TYPE=Debug: enables LFORTRAN_ASSERT/LCOMPILERS_ASSERT,
    #   which is exactly the kind of internal-invariant check fuzzing
    #   wants to trip.
    _run([
        "cmake", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DWITH_LLVM=yes",
        "-DWITH_STACKTRACE=yes",
        f"-DLLVM_DIR={llvm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_PREFIX}",
        src_dir,
    ], cwd=build_dir)

    _run(["ninja", f"-j{njobs}"], cwd=build_dir)
    _run(["ninja", "install"], cwd=build_dir)


def _setup_shared_fortran_corpus(project_root):
    """
    LFortran and Flang are both Fortran frontends, so they fuzz off the
    *same* Fortran seed corpus (llvm-project's flang/test) rather than
    each sparse-cloning their own copy. If projects/flang has already
    cloned it, reuse that directory as-is; otherwise clone it directly
    into projects/flang/llvm-project (the same path projects/flang/setup.py
    uses), so whichever project sets up first creates it once for both.
    See projects/lfortran/config.yaml: paths.seed_source.
    """
    flang_dir = os.path.join(project_root, "..", "flang")
    src_root = os.path.join(flang_dir, "llvm-project")
    seed_dir = os.path.join(src_root, "flang", "test")

    if os.path.exists(seed_dir):
        print(f"Shared Fortran seed corpus already present at {seed_dir}")
        return

    print("Sparse-cloning llvm-project (flang/test only) for the shared Fortran corpus...")
    _run([
        "git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
        "https://github.com/llvm/llvm-project.git", src_root,
    ])
    _run(["git", "sparse-checkout", "set", "flang/test"], cwd=src_root)
    _run(["git", "checkout"], cwd=src_root)

    if not os.path.exists(seed_dir):
        print(f"Warning: {seed_dir} not found after clone — seed collection will find 0 seeds.")
    else:
        print(f"Shared Fortran seed corpus ready at {seed_dir}")


def setup(project_root):
    """
    Sets up the LFortran fuzzing environment (runs inside the ffl-lfortran
    container, see Dockerfile):
    1. Clones lfortran/lfortran and builds the tip of `main` by default
       (WITH_LLVM=yes, Debug build w/ assertions, symbolized stacktraces)
       and installs it to /opt/lfortran, which is already on PATH. Set
       LFORTRAN_REF to a tag/commit/branch to pin to something reproducible
       instead of always tracking main.
    2. Ensures the shared Fortran seed corpus (projects/flang's sparse
       llvm-project/flang/test checkout) exists, so both projects/flang
       and projects/lfortran fuzz off identical seed sources.
    """
    print(f"Setting up LFortran in: {project_root}")

    if os.path.exists(LFORTRAN_BIN) and not os.environ.get("LFORTRAN_FORCE_REBUILD"):
        print(f"LFortran already built and installed at {LFORTRAN_BIN} "
              "(set LFORTRAN_FORCE_REBUILD=1 to rebuild against the latest ref)")
    else:
        ref = os.environ.get("LFORTRAN_REF", DEFAULT_REF)
        _clone_and_build(project_root, ref)
        if not os.path.exists(LFORTRAN_BIN):
            print(f"ERROR: build finished but {LFORTRAN_BIN} not found.")
            sys.exit(1)

    result = subprocess.run([LFORTRAN_BIN, "--version"], capture_output=True, text=True)
    print(f"lfortran available: {(result.stdout + result.stderr).strip()}")

    _setup_shared_fortran_corpus(project_root)

    print("LFortran setup complete.")
