"""
projects/gcc/driver.py — run a fused program through GCC and report whether
what came back is a bug.

This file owns *execution*: picking gcc vs g++, choosing flags, containing
the compile, and cleaning up. The judgement of what the output means lives
in projects/gcc/analyzer.py, which is pure text analysis and testable on
its own.

Flag selection is the other half of the bug-finding story. A fused program
that only ever meets `gcc -fsyntax-only` exercises the front end and
nothing else, while most GCC ICEs live in the middle end and the back end.
So each execution draws a random point in the (mode x optimisation x pass
flags) space, which is what puts fused code in front of the vectoriser,
the inliner, VRP and RTL expansion.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, not as a package
# member, so there is no parent package for a relative import to resolve
# against. The absolute form works because FFL always runs from the repo
# root; the fallback covers loading this file directly (e.g. from a test).
try:
    from projects.gcc.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_gcc_analyzer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class GCCDriver(BaseDriver):
    """Drives the gcc/g++ built by projects/gcc/setup.py."""

    STD_C = ["c89", "c99", "c11", "c17", "c23",
             "gnu89", "gnu99", "gnu11", "gnu17", "gnu23"]
    STD_CXX = ["c++98", "c++11", "c++14", "c++17", "c++20", "c++23",
               "gnu++11", "gnu++17", "gnu++20"]
    OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Og", "-Ofast"]

    # How far to push each invocation. -fsyntax-only is cheap and hammers
    # the front end; the rest reach the middle and back ends, where GCC's
    # checking machinery (and most of its ICEs) live. Weighted towards the
    # cheap mode because throughput here is bounded by compile time.
    MODES = ["-fsyntax-only", "-S -o /dev/null", "-c -o /dev/null"]
    MODE_WEIGHTS = [40, 35, 25]

    # Optimiser knobs worth varying. Restricted to flags that are legal on
    # their own and need no extra libraries, so that a rejected program is
    # rejected by the compiler rather than by the driver.
    MISC_FLAGS = [
        "-Wall", "-Wextra",
        "-fno-strict-aliasing", "-fstrict-aliasing",
        "-funroll-loops", "-ftree-vectorize", "-fno-tree-vectorize",
        "-fipa-pta", "-fno-inline", "-finline-functions",
        "-fstack-protector-all", "-fno-omit-frame-pointer",
        "-ffast-math", "-frounding-math", "-fwrapv", "-ftrapv",
        "-g", "-fPIC", "-fexceptions",
        "-fgraphite-identity", "-floop-nest-optimize",
    ]

    # A single translation unit can drive GCC's memory use into the tens of
    # gigabytes (runaway template instantiation, unbounded constexpr).
    # Without a cap the *system* OOM killer fires, and under cgroup v2 that
    # can take down every process in the container, orchestrator included.
    # Capping memory keeps the failure local: GCC reports "out of memory
    # allocating ..." and exits, which analyzer.classify then files as
    # resource exhaustion rather than as a bug. How the cap is applied
    # depends on whether the compiler is sanitized — see __init__.
    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        mem_mb = config.get("execution", {}).get("mem_limit_mb",
                                                 self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_kb = int(mem_mb) * 1024
        install = os.path.join(self.ffl_root, "projects", "gcc", "gcc-install")
        self.gcc_bin = os.path.join(install, "bin", "gcc")
        self.gxx_bin = os.path.join(install, "bin", "g++")
        # setup.py drops this marker when built with FFL_GCC_SANITIZE=1.
        # An ASan-instrumented cc1 maps ~20TB of virtual address space for
        # its shadow region before running any of its own code, so the
        # `ulimit -v` cap below would prevent it from starting at all —
        # every execution would come back as a launch failure and the run
        # would find nothing. Under ASan we cap resident memory through
        # ASan's own allocator instead.
        self.sanitized = os.path.exists(os.path.join(install, ".ffl_sanitized"))
        # The freshly built compiler needs its own libstdc++/libgcc at run
        # time, not the host GCC's older ones.
        self.lib_path = os.pathsep.join([os.path.join(install, "lib64"),
                                         os.path.join(install, "lib")])

    # ── command construction ──────────────────────────────────────────

    def _build_command(self, seed_file, content, extension):
        facts = analyze_seed(content, extension)
        binary = self.gxx_bin if facts["is_cxx"] else self.gcc_bin
        stds = self.STD_CXX if facts["is_cxx"] else self.STD_C

        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0],
                 random.choice(self.OPT_LEVELS)]
        if random.random() > 0.3:
            flags.append(f"-std={random.choice(stds)}")
        flags.extend(random.sample(self.MISC_FLAGS, random.randint(0, 3)))
        # The seed's own dg-options last, so they win any conflict: they are
        # what the test was written to exercise.
        flags.extend(facts["dg_options"])

        asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1"
        ubsan_opts = "print_stacktrace=1:halt_on_error=1"
        if self.sanitized:
            # ASan's own cap, since ulimit -v cannot be used here. Hitting
            # it prints "hard rss limit exhausted", which analyzer.classify
            # files as resource exhaustion rather than as a finding.
            asan_opts += (f":hard_rss_limit_mb={self.mem_limit_kb // 1024}"
                          ":allocator_may_return_null=1")
            mem_cap = ""
        else:
            mem_cap = f"ulimit -v {self.mem_limit_kb}; "
        return (
            # ulimit -c 0: a core dump per crash would fill the disk within
            # minutes at this hit rate, and we keep the reproducer instead.
            f"{mem_cap}ulimit -c 0; "
            f"LD_LIBRARY_PATH={self.lib_path}:$LD_LIBRARY_PATH "
            f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
            f"{binary} {' '.join(flags)} {seed_file}"
        )

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            ext = seed.metadata.get("extension") or ".c"
            seed_file = os.path.join(workdir, f"{seed.id}{ext}")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            cmd = self._build_command(seed_file, seed.content, ext)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        verdict = classify((stderr or "") + "\n" + (stdout or ""))
        res = ExecutionResult(rc, stdout, stderr, time.time() - start,
                              verdict["is_bug"], verdict["signature"])
        res.command = cmd
        # _save_crash_bundle rewrites this exact path to "$SCRIPT_DIR/test.c"
        # when it writes test.sh. Without it the fallback substitutes the bare
        # seed id instead, leaving the temp directory and a doubled extension
        # in the path — every saved reproducer comes out unrunnable.
        res.seed_file = seed_file
        return res

    # ── crash oracle ──────────────────────────────────────────────────
    #
    # execute() already classifies, but the orchestrator calls these two
    # directly in some paths (--execute replay, re-verification of a saved
    # bundle), so they must agree with it. Both delegate to the same
    # analyzer, which is the point of keeping the logic there.

    def _check_crash(self, stdout, stderr, return_code):
        return classify((stderr or "") + "\n" + (stdout or ""))["is_bug"]

    def extract_crash_signature(self, stdout, stderr, return_code):
        sig = classify((stderr or "") + "\n" + (stdout or ""))["signature"]
        return sig or super().extract_crash_signature(stdout, stderr, return_code)
