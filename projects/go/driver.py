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
import subprocess
import threading
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

    # GOEXPERIMENT gates whole code paths in the compiler rather than
    # tuning one that always runs: a new inliner, a different map
    # implementation, a different loop-variable scope. Each is a large body
    # of compiler code that the default build never executes, and each has
    # historically shipped with its own ICEs. Which names a given toolchain
    # accepts depends on its version — an unknown one makes `go` refuse to
    # run at all — so the set is probed once at startup and the survivors
    # are what gets drawn from. The empty string is the default build and
    # stays the common case.
    GOEXPERIMENT_CANDIDATES = [
        "newinliner",       # the rewritten inliner
        "loopvar",          # per-iteration loop variable scoping
        "swissmap",         # the Swiss-table map implementation
        "cgocheck2",        # stricter cgo pointer checking
        "aliastypeparams",  # generic alias types
        "regabiargs",       # register-based calling convention
        "arenas",           # arena allocation
    ]
    GOEXPERIMENT_RATE = 0.35

    # A single fused file can drive the compiler into unbounded type
    # instantiation. Without a cap the OOM killer fires and can take the
    # orchestrator with it; with one, Go reports "out of memory" and exits,
    # which analyzer.classify files as resource exhaustion rather than a bug.
    DEFAULT_MEM_LIMIT_MB = 4096

    # Ceiling on .fused/go-cache. See _trim_cache for why Go's own trimming
    # cannot be relied on here.
    DEFAULT_CACHE_LIMIT_MB = 8192
    # Executions between size checks. The check walks the cache's 256 shards,
    # so it is not free; it just has to run far more often than the cache
    # takes to grow by its headroom.
    DEFAULT_CACHE_CHECK_EVERY = 500
    # Never evict an entry used more recently than this. cmd/go refreshes an
    # entry's mtime on use but at most once per hour (mtimeInterval in
    # src/cmd/go/internal/cache/cache.go), so an entry in active use can
    # still carry an mtime up to an hour old. Evicting inside that window
    # risks deleting a file a running build is about to open.
    CACHE_GRACE_SECONDS = 3600

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        mem_mb = exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_kb = int(mem_mb) * 1024
        self.goroot = os.path.join(self.ffl_root, "projects", "go", "go-src")
        self.go_bin = os.path.join(self.goroot, "bin", "go")
        # One shared build cache across executions. Without it every
        # compilation rebuilds the standard library for its target, which
        # dominates the run: the first linux/arm64 build takes seconds, the
        # rest take milliseconds.
        self.gocache = os.path.join(self.ffl_root, ".fused", "go-cache")
        os.makedirs(self.gocache, exist_ok=True)
        # Scratch space for the toolchain's own temporaries (see _build_command).
        # Shared and persistent rather than per-execution: the command recorded
        # in a crash bundle's test.sh names this directory, and a reproducer
        # pointing at a directory deleted the moment the run ended is not a
        # reproducer -- `go build` fails with "creating work dir: no such file
        # or directory" before it ever reads the program.
        self.gotmp = os.path.join(self.ffl_root, ".fused", "go-tmp")
        os.makedirs(self.gotmp, exist_ok=True)

        # Disk containment. 0 (or negative) disables the cap entirely.
        self.cache_limit_bytes = int(
            exec_cfg.get("cache_limit_mb", self.DEFAULT_CACHE_LIMIT_MB)
        ) * 1024 * 1024
        self.cache_check_every = max(
            1, int(exec_cfg.get("cache_check_every",
                                self.DEFAULT_CACHE_CHECK_EVERY)))
        # execute() runs on `concurrency` threads. The counter needs a lock of
        # its own so that incrementing it never waits behind a trim in
        # progress, and the trim needs one that non-participating threads can
        # decline rather than queue on.
        self._goexperiments = self._probe_goexperiments()
        self._cache_counter_lock = threading.Lock()
        self._cache_trim_lock = threading.Lock()
        self._execs_since_cache_check = 0

    # ── command construction ──────────────────────────────────────────

    def _probe_goexperiments(self):
        """Keep only the experiments this toolchain actually knows.

        `go env` rejects an unknown GOEXPERIMENT outright, so an unprobed
        name would not produce an interesting compile failure — it would
        stop `go` before it ever read the seed, and every execution drawing
        that name would be wasted."""
        ok = []
        for name in self.GOEXPERIMENT_CANDIDATES:
            env = dict(os.environ, GOEXPERIMENT=name, GOROOT=self.goroot)
            try:
                r = subprocess.run([self.go_bin, "env", "GOEXPERIMENT"],
                                   capture_output=True, env=env, timeout=30)
                if r.returncode == 0:
                    ok.append(name)
            except (OSError, subprocess.SubprocessError):
                break
        return ok

    def _goexperiment(self):
        """The GOEXPERIMENT for one execution, or "" for the default build."""
        if self._goexperiments and random.random() < self.GOEXPERIMENT_RATE:
            return random.choice(self._goexperiments)
        return ""

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
        # GOMEMLIMIT is a *soft* limit on the Go heap; `ulimit -v` below is a
        # hard limit on total address space, which is strictly larger than the
        # heap. Setting the two equal means the process hits the hard wall
        # before the soft limit ever engages, so the runtime never gets the
        # chance to collect harder and back off. Leaving headroom is what
        # makes GOMEMLIMIT do anything at all.
        soft_mem_mib = max(64, (self.mem_limit_kb // 1024) * 3 // 4)
        env = (
            f"GOROOT={self.goroot} GOCACHE={self.gocache} "
            # Contain the toolchain's scratch space. Without this the link and
            # assembly steps write go-build* directories under /tmp, and any
            # execution alive when the orchestrator is SIGKILLed (which
            # cleanup_stale_processes does by pattern, with the watchdog
            # restarting it) leaves them there permanently. go removes its own
            # work dir on a clean exit, so what collects here is only what a
            # killed run left behind -- swept by _sweep_stale_workdirs at
            # startup, same as the execution workdirs.
            f"GOTMPDIR={self.gotmp} "
            f"GOOS={goos} GOARCH={goarch} "
            f"GOPROXY=off GOFLAGS=-mod=mod "
            # See GOEXPERIMENT_CANDIDATES. Empty means the default build,
            # and `GOEXPERIMENT=` is how you spell that.
            f"GOEXPERIMENT={self._goexperiment()} "
            # CGO off unless the seed asks for it: it needs a target C
            # toolchain that does not exist for the cross targets.
            f"CGO_ENABLED={'1' if facts['has_cgo'] else '0'} "
            # Keep a runaway compilation from taking the whole box down.
            f"GOMEMLIMIT={soft_mem_mib}MiB "
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
        # Every distinct fused program leaves new entries in the shared build
        # cache. Nothing else ever removes them; see _trim_cache.
        self._trim_cache()
        return res

    # ── disk containment ──────────────────────────────────────────────

    def prepare_environment(self):
        """Called once by the orchestrator before the fuzzing loop starts."""
        self._sweep_stale_workdirs()
        self._trim_cache(force=True)

    def _sweep_stale_workdirs(self):
        """Remove per-execution directories left by an earlier run.

        execute() wipes its workdir in a `finally`, but that only runs if the
        process lives long enough to reach it. cleanup_stale_processes kills
        by pattern with SIGKILL and the watchdog restarts the orchestrator, so
        whichever workdirs were live at that moment are orphaned — and nothing
        else collects them. Safe to do here and only here: prepare_environment
        runs before any execution thread starts, so no live workdir exists yet.
        """
        removed = 0
        for base in (os.path.join(self.fused_base, self.project_name), self.gotmp):
            if not os.path.isdir(base):
                continue
            try:
                with os.scandir(base) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            shutil.rmtree(entry.path, ignore_errors=True)
                            removed += 1
            except OSError:
                continue
        if removed:
            print(f"[go] swept {removed} stale scratch dir(s)")

    def _cache_shard_entries(self):
        """(mtime, size, path) for every file in the cache's 256 shards.

        Only the shard directories are walked, so the cache's own bookkeeping
        at the top level (trim.txt, README, lock) is never a candidate for
        eviction.
        """
        entries = []
        total = 0
        for i in range(256):
            shard = os.path.join(self.gocache, f"{i:02x}")
            try:
                with os.scandir(shard) as it:
                    for e in it:
                        try:
                            if not e.is_file(follow_symlinks=False):
                                continue
                            st = e.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        # st_blocks, not st_size: the cap is about disk, and
                        # the cache is overwhelmingly tiny files (measured at
                        # 87% of 712k entries under 4 KB) that each still take a
                        # whole filesystem block. Summing st_size undercounted
                        # real consumption by 37% -- 6635 MiB apparent against
                        # 9121 MiB actually used -- so an 8 GiB cap silently
                        # cost ~11 GB of disk.
                        size = st.st_blocks * 512
                        entries.append((st.st_mtime, size, e.path))
                        total += size
            except OSError:
                continue
        return entries, total

    def _trim_cache(self, force=False):
        """Keep .fused/go-cache under execution.cache_limit_mb.

        Why this is needed at all: cmd/go trims its own cache, but on a
        schedule built for a developer's laptop rather than a fuzzer.
        src/cmd/go/internal/cache/cache.go:

            trimInterval  = 24 * time.Hour   // scan at most once a day
            trimLimit     = 5 * 24 * time.Hour   // drop entries unused 5 days

        So for the first five days of a continuous run Go deletes *nothing*,
        and every distinct fused program has meanwhile added a fresh set of
        entries — measured at ~36 KB for a small program. At this fuzzer's
        throughput that is multiple GB per hour of monotonic growth, which is
        what fills the disk. The per-target standard libraries are not the
        problem: they are ~30 MB each and there are thirteen targets.

        Eviction order is Go's own policy, oldest mtime first. cmd/go
        refreshes an entry's mtime when it is used, so the stdlib entries —
        the ones worth keeping and the expensive ones to rebuild — stay at the
        young end while single-use fused-program entries age out. Entries
        inside CACHE_GRACE_SECONDS are never touched, which is what keeps this
        safe to run while builds are in flight.
        """
        if self.cache_limit_bytes <= 0:
            return

        if force:
            self._cache_trim_lock.acquire()
        else:
            with self._cache_counter_lock:
                self._execs_since_cache_check += 1
                if self._execs_since_cache_check < self.cache_check_every:
                    return
                self._execs_since_cache_check = 0
            # One trimmer is enough. A thread that arrives while a trim is
            # already running skips its turn rather than serialising behind it.
            if not self._cache_trim_lock.acquire(blocking=False):
                return

        try:
            entries, total = self._cache_shard_entries()
            if total <= self.cache_limit_bytes:
                return
            # Free down to 80% of the cap, so the next execution to check does
            # not immediately trigger another full walk.
            target = int(self.cache_limit_bytes * 0.8)
            cutoff = time.time() - self.CACHE_GRACE_SECONDS
            entries.sort()          # oldest mtime first
            freed = removed = 0
            for mtime, size, path in entries:
                if total - freed <= target:
                    break
                if mtime > cutoff:
                    # Sorted by mtime, so everything from here on is hot.
                    break
                try:
                    os.remove(path)
                except OSError:
                    continue
                freed += size
                removed += 1
            mib = 1024 * 1024
            print(f"[go] build cache {total // mib} MiB over the "
                  f"{self.cache_limit_bytes // mib} MiB cap: removed "
                  f"{removed} entries, freed {freed // mib} MiB")
            if total - freed > self.cache_limit_bytes:
                print("[go] warning: build cache still over its cap - every "
                      "remaining entry was used within the last hour. Raise "
                      "execution.cache_limit_mb or lower concurrency.")
        finally:
            self._cache_trim_lock.release()

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
