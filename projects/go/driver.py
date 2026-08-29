"""
projects/go/driver.py — compile a fused Go program and report whether what
came back is a bug.

This file owns execution: building the module context Go needs, choosing
flags, containing the compile, cleaning up. Judgement lives in
projects/go/analyzer.py.

The flags that matter
---------------------
`-d=panic` is mandatory, not tuning. base.FatalfAt only prints "internal
compiler error" when `Debug.Panic != 0 || numErrors == 0`, so on any file
that also has ordinary errors — which is most fused output — an ICE is
silently swallowed without it. See the note in analyzer.py.

`-d=ssa/check/on` verifies SSA form after every optimisation pass
("enables checking after each phase",
src/cmd/compile/internal/ssacompile/compile.go). It is Go's counterpart
to GCC's --enable-checking=rtl and the main reason a malformed-SSA bug
becomes a crash instead of silently wrong code. It costs compile time, so
it is on for most but not all executions — a pass that only misbehaves
without the checker running is still worth reaching.

Cross-compilation is nearly free in Go and is used deliberately: one
toolchain targets every GOOS/GOARCH with no extra sysroot, so varying the
target costs a environment variable and buys the whole backend surface.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.go.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_go_analyzer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class GoDriver(BaseDriver):
    """Drives the toolchain built by projects/go/setup.py."""

    # Targets this toolchain can build for without any external tooling.
    # Weighted towards the host: an amd64 miscompile is the one a user is
    # most likely to hit, and the others share most of the compiler anyway.
    TARGETS = [
        ("linux", "amd64"), ("linux", "arm64"), ("linux", "386"),
        ("linux", "arm"), ("linux", "riscv64"), ("linux", "ppc64le"),
        ("linux", "s390x"), ("linux", "mips64"), ("linux", "loong64"),
        ("darwin", "arm64"), ("windows", "amd64"), ("js", "wasm"),
        ("freebsd", "amd64"),
    ]
    TARGET_WEIGHTS = [40, 12, 6, 5, 5, 4, 3, 3, 3, 6, 6, 4, 3]

    # Compiler debug flags worth varying. -d=panic is added separately and
    # unconditionally; these are the optional ones.
    DEBUG_FLAGS = [
        "-d=ssa/check/on",          # verify SSA after every pass
        "-d=checkptr=2",            # instrument unsafe.Pointer conversions
        "-d=gccheck=1",             # check the compiler's own heap/GC use
        "-d=escapealiascheck=1",    # extra validation in alias analysis
        "-d=nil=1",                 # report nil-check elimination decisions
    ]

    # Ordinary compiler knobs. -N -l (no optimisation, no inlining) and
    # full optimisation exercise very different code, so both are drawn.
    OPT_FLAGS = ["", "-N", "-l", "-N -l", "-m", "-m -m", "-B", "-S"]

    # A single fused file can drive the compiler into unbounded type
    # instantiation. Without a cap the OOM killer fires and can take the
    # orchestrator with it; with one, Go reports "out of memory" and exits,
    # which analyzer.classify files as resource exhaustion rather than a bug.
    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        mem_mb = config.get("execution", {}).get("mem_limit_mb",
                                                 self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_kb = int(mem_mb) * 1024
        self.goroot = os.path.join(self.ffl_root, "projects", "go", "go-src")
        self.go_bin = os.path.join(self.goroot, "bin", "go")
        # One shared build cache across executions. Without it every
        # compilation rebuilds the standard library for its target, which
        # dominates the run: the first linux/arm64 build takes seconds, the
        # rest take milliseconds.
        self.gocache = os.path.join(self.ffl_root, ".fused", "go-cache")
        os.makedirs(self.gocache, exist_ok=True)

    # ── command construction ──────────────────────────────────────────

    def _gcflags(self):
        # Mandatory: without it an ICE on a file that also has ordinary
        # errors prints nothing at all.
        flags = ["-d=panic"]
        # SSA checking most of the time, but not always — a pass that only
        # misbehaves with the checker off is still a pass worth reaching.
        if random.random() < 0.75:
            flags.append("-d=ssa/check/on")
        for f in random.sample(self.DEBUG_FLAGS[1:], random.randint(0, 2)):
            flags.append(f)
        opt = random.choice(self.OPT_FLAGS)
        if opt:
            flags.append(opt)
        return " ".join(flags)

    def _build_command(self, workdir, facts):
        goos, goarch = random.choices(self.TARGETS,
                                      weights=self.TARGET_WEIGHTS, k=1)[0]
        # cgo cannot cross-compile without a target C toolchain, and a seed
        # importing "C" is not fusable anyway — keep those on the host.
        if facts["has_cgo"]:
            goos, goarch = "linux", "amd64"

        gcflags = self._gcflags()
        # `go build -o /dev/null` is only legal for a main package; for a
        # library package `go build` compiles and discards on its own.
        out = "-o /dev/null " if facts["is_main"] else ""
        env = (
            f"GOROOT={self.goroot} GOCACHE={self.gocache} "
            f"GOOS={goos} GOARCH={goarch} "
            f"GOPROXY=off GOFLAGS=-mod=mod "
            # CGO off unless the seed asks for it: it needs a target C
            # toolchain that does not exist for the cross targets.
            f"CGO_ENABLED={'1' if facts['has_cgo'] else '0'} "
            # Keep a runaway compilation from taking the whole box down.
            f"GOMEMLIMIT={self.mem_limit_kb // 1024}MiB "
        )
        # `-gcflags=` without an `all=` prefix, deliberately: the prefixed
        # form applies the flags to every dependency including the standard
        # library, and since these flags are drawn at random the cache key
        # changes on every execution — which made each one rebuild the whole
        # stdlib for its target and pinned throughput at 1.7 tests/s. The
        # stdlib is fixed input here, not the thing under test, so building
        # it once per target with default flags is both faster and more
        # correct.
        return (
            f"ulimit -v {self.mem_limit_kb}; ulimit -c 0; "
            f"{env}{self.go_bin} build {out}-gcflags='{gcflags}' ."
        )

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            # Go resolves imports through a module, so the seed needs one
            # around it. Writing a go.mod per execution is cheaper than it
            # looks — GOPROXY=off means nothing is fetched, and the build
            # cache is shared.
            with open(os.path.join(workdir, "go.mod"), "w") as f:
                f.write("module fflfuzz\n\ngo 1.21\n")
            seed_file = os.path.join(workdir, "main.go")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            cmd = self._build_command(workdir, facts)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        verdict = classify((stderr or "") + "\n" + (stdout or ""))
        res = ExecutionResult(rc, stdout, stderr, time.time() - start,
                              verdict["is_bug"], verdict["signature"])
        res.command = cmd
        # _save_crash_bundle rewrites this exact path to "$SCRIPT_DIR/test.go"
        # when it writes test.sh; without it the fallback substitutes the
        # bare seed id and the saved reproducer is unrunnable.
        res.seed_file = seed_file
        return res

    # ── crash oracle ──────────────────────────────────────────────────
    #
    # execute() already classifies, but the orchestrator calls these
    # directly on some paths (--execute replay, re-verifying a bundle), so
    # they must agree with it. Both delegate to the same analyzer.

    def _check_crash(self, stdout, stderr, return_code):
        return classify((stderr or "") + "\n" + (stdout or ""))["is_bug"]

    def extract_crash_signature(self, stdout, stderr, return_code):
        sig = classify((stderr or "") + "\n" + (stdout or ""))["signature"]
        return sig or super().extract_crash_signature(stdout, stderr, return_code)
