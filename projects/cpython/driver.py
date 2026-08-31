"""
projects/cpython/driver.py — run a fused Python program under the
instrumented interpreter and report whether what came back is a bug.

This file owns execution; judgement lives in projects/cpython/analyzer.py.

CPython is the one target in this repo whose seeds are *executed* rather
than compiled, which changes two things:

  * Most failures are ordinary Python exceptions and mean nothing about
    CPython. The oracle's real work is spotting the few that do — see
    analyzer.py, particularly the C-API contract violations that arrive
    looking like a normal traceback.

  * A seed can bind a port, fork, or delete files. Those are containment
    problems that no compile-only adapter has, and they are the reason the
    concurrency here stays below the 16 the other adapters use.

Interpreter flags
-----------------
The previous version built a `FUZZ_FLAGS` list and then never used it:

    flags = ""  # self._get_random_flags()

so every execution ran with none. They are back, because each one changes
a different part of the interpreter's startup and bytecode path — `-O`/
`-OO` strip asserts and docstrings, `-b`/`-bb` change bytes/str comparison
behaviour, `-X showrefcount` exercises the refcount bookkeeping a pydebug
build carries, `-I`/`-E`/`-S` change how the runtime is initialised.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.cpython.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_cpython_analyzer",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class CPythonDriver(BaseDriver):
    """Drives the pydebug+ASan interpreter built by projects/cpython/setup.py."""

    # Startup and bytecode flags. Each reaches a different part of the
    # interpreter; -P is 3.11+, which the built trunk interpreter has even
    # though an older system python would reject it.
    FUZZ_FLAGS = [
        "-b", "-bb", "-B", "-E", "-I", "-O", "-OO", "-P", "-s", "-S", "-u",
    ]
    # -X options, verified against Python/initconfig.c in the cloned tree.
    X_OPTIONS = [
        "-X showrefcount",      # pydebug refcount bookkeeping
        "-X tracemalloc",
        "-X dev",               # development mode: extra checks
        "-X faulthandler",      # dumps a C traceback on a fatal signal
        "-X importtime",
        "-X int_max_str_digits=0",
        "-X pycache_prefix=/tmp/ffl_pycache",
    ]

    # A fused program can allocate without bound. The cap is applied
    # through ASan's own allocator rather than `ulimit -v`, because an
    # ASan-instrumented process reserves a very large virtual address
    # space up front and a -v cap would stop it from starting at all.
    DEFAULT_MEM_LIMIT_MB = 2048

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        # `memory_limit_mb` is the historical spelling in this project's
        # config; `mem_limit_mb` is what the newer adapters use. Accept both
        # so an old config keeps working.
        self.memory_limit_mb = int(
            exec_cfg.get("mem_limit_mb",
                         exec_cfg.get("memory_limit_mb", self.DEFAULT_MEM_LIMIT_MB)) or 0)
        self.python_bin = os.path.join(
            self.ffl_root, "projects", "cpython", "cpython", "build", "python")

    def _get_random_flags(self, facts):
        flags = random.sample(self.FUZZ_FLAGS, random.randint(0, 3))
        if random.random() < 0.35:
            flags.append(random.choice(self.X_OPTIONS))
        # -X faulthandler turns a fatal signal into a C-level traceback,
        # which is most of a crash signature. Worth forcing whenever the
        # seed is the kind that segfaults.
        if facts["uses_ctypes"] or facts["touches_code_objects"]:
            flags.append("-X faulthandler")
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            seed_file = os.path.join(workdir, f"{seed.id}.py")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)

            asan_opts = ("abort_on_error=1:detect_leaks=0:"
                         "allocator_may_return_null=1:symbolize=1")
            if self.memory_limit_mb:
                asan_opts += f":hard_rss_limit_mb={self.memory_limit_mb}"
            env = (
                f"ASAN_OPTIONS='{asan_opts}' "
                "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' "
                # Keeps a seed that reads stdin from hanging until the
                # timeout: it gets EOF immediately instead.
                "PYTHONUNBUFFERED=1 "
                # Never write .pyc next to the seed; the workdir is deleted
                # and a stale cache would be read by the next execution.
                "PYTHONDONTWRITEBYTECODE=1 "
            )
            flags = self._get_random_flags(facts)
            cmd = (f"ulimit -c 0; {env}{self.python_bin} {flags} "
                   f"{seed_file} < /dev/null").strip()
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        # Both streams: a Python traceback goes to stderr, but ASan and the
        # debug allocator write to stdout in some configurations, and the
        # previous version only ever looked at stderr.
        verdict = classify((stderr or "") + "\n" + (stdout or ""))
        res = ExecutionResult(rc, stdout, stderr, time.time() - start,
                              verdict["is_bug"], verdict["signature"])
        res.command = cmd
        res.seed_file = seed_file
        return res

    # ── crash oracle ──────────────────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        return classify((stderr or "") + "\n" + (stdout or ""))["is_bug"]

    def extract_crash_signature(self, stdout, stderr, return_code):
        sig = classify((stderr or "") + "\n" + (stdout or ""))["signature"]
        return sig or super().extract_crash_signature(stdout, stderr, return_code)
