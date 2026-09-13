"""
projects/julia/driver.py — run a fused Julia program under the assert
build (projects/julia/setup.py) and report whether what came back is a
bug.

Julia compiles as it runs, so a single script exercises the parser, the
lowering, type inference, the optimiser and LLVM before anything of the
program itself happens. Most failures are ordinary Julia exceptions and
say nothing about the implementation. What does:

  * `Internal error: encountered unexpected error in runtime` — the
    runtime's own words for a broken invariant, printed with a Julia and
    a C backtrace.
  * `Assertion failed` from the assert build (`FORCE_ASSERTIONS=1`), and
    `LLVM ERROR` from the code generator.
  * A fatal signal: Julia installs handlers and prints `signal (11)` with
    a backtrace before dying.

`StackOverflowError` is *not* a bug: Julia detects deep recursion and
throws it, and a fused program recurses without bound all the time.
Neither is an allocation failure.

Axes drawn per execution
------------------------
`--check-bounds=yes|no` decides whether `@inbounds` is honoured — "no"
is where a wrong index reaches memory. `-O0..-O3` and
`--compile=yes|min` select how much of the optimiser and the compiler
each call goes through; `--inline=no` keeps whole categories of
optimisation from firing. Two paths that must agree on every program
are a bug source in themselves.
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult


class JuliaDriver(BaseDriver):

    FUZZ_FLAGS = [
        "--check-bounds=yes", "--check-bounds=no",
        "-O0", "-O1", "-O2", "-O3",
        "--inline=no", "--compile=min", "--math-mode=fast",
        "--depwarn=error", "--warn-overwrite=yes",
    ]

    DEFAULT_MEM_LIMIT_MB = 4096

    _INTERNAL_RE = re.compile(
        r'(Internal error: [^\n]*|Assertion failed: [^\n]*|LLVM ERROR: [^\n]*|'
        r'Unreachable reached[^\n]*|GC error[^\n]*)')
    # The C assertion form glibc prints, which is what Julia's own
    # `assert()` produces in an assert build:
    #   julia: /src/genericmemory.c:412: jl_memoryrefunset: Assertion `e' failed.
    # It has no "Assertion failed:" in it, so _INTERNAL_RE misses it and
    # every such finding was signed "unknown" — which means two different
    # assertion failures land in the same bundle directory and the second
    # overwrites the first. The first Julia finding of the campaign was
    # saved that way.
    _C_ASSERT_RE = re.compile(
        r'([\w./+-]*?([\w+-]+\.[ch])):(\d+):\s*(\w+):\s*Assertion\s+`([^\n]*?)\'\s+failed')
    _SIGNAL_RE = re.compile(r'signal \((\d+)\): ([^\n]*)')
    # Julia's own backtrace frames: `foo at /path/file.jl:12`.
    _FRAME_RE = re.compile(r'^([A-Za-z_][\w!]*) at [^\n]*\.(?:jl|c|cpp):\d+', re.M)
    _ASAN_RE = re.compile(r'SUMMARY: (\w+Sanitizer): ([\w-]+)')

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.memory_limit_mb = int(exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB) or 0)
        self.julia = os.path.join(self.ffl_root, "projects", "julia", "julia-src", "julia")

    def _random_flags(self):
        flags = random.sample(self.FUZZ_FLAGS, random.randint(0, 3))
        # -O and --check-bounds each appear once or the last wins silently.
        seen = set()
        out = []
        for f in flags:
            key = f.split("=")[0] if "=" in f else ("-O" if f.startswith("-O") else f)
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return " ".join(out)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.jl")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            env = ("JULIA_DEPOT_PATH=/tmp/ffl-julia-depot JULIA_PKG_OFFLINE=true "
                   "JULIA_NUM_THREADS=1 JULIA_LOAD_PATH=@stdlib "
                   f"TMPDIR={workdir} HOME=/tmp ")
            if self.memory_limit_mb:
                env += f"JULIA_HEAP_SIZE_HINT={self.memory_limit_mb}M "
            cmd = (f"ulimit -c 0; {env}{self.julia} --startup-file=no --color=no "
                   f"{self._random_flags()} {seed_file} < /dev/null").strip()
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        crashed = self._check_crash(stdout, stderr, rc)
        signature = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
        res = ExecutionResult(rc, stdout, stderr, time.time() - start, crashed, signature)
        res.command = cmd
        res.seed_file = seed_file
        return res

    # ── crash oracle ──────────────────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        # Deep recursion, which a fused program reaches constantly: Julia
        # detects it and throws, and the same guard page under a
        # sanitizer would be reported as a stack overflow.
        if re.search(r'StackOverflowError|stack overflow in type inference|'
                     r'AddressSanitizer: stack-overflow', text):
            return False
        if re.search(r'OutOfMemoryError|cannot allocate|out of memory|'
                     r'Unable to allocate', text):
            return False
        # A test that *prints* the words (Julia's own test suite asserts
        # on error text) is not a crash: a real report carries a
        # backtrace.
        if "Internal error" in text and "Stacktrace" not in text and "backtrace" not in text \
                and "signal (" not in text:
            return False
        return super()._check_crash(stdout, stderr, return_code)

    @staticmethod
    def _mask(s):
        s = re.sub(r'0x[0-9a-fA-F]+', '0x', s)
        s = re.sub(r'/[\w./\-]+', '<path>', s)
        s = re.sub(r'\b\d+\b', 'N', s)
        return re.sub(r'\s+', ' ', s).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        m = self._ASAN_RE.search(text)
        if m:
            return f"{m.group(1)}: {m.group(2)}"
        m = self._C_ASSERT_RE.search(text)
        if m:
            # file:line and the function, which identify the check far
            # better than the expression text does.
            return f"Assertion {m.group(2)}:{m.group(3)} {m.group(4)}: " \
                   f"{self._mask(m.group(5))[:100]}"
        m = self._INTERNAL_RE.search(text)
        if m:
            frame = self._FRAME_RE.search(text)
            sig = self._mask(m.group(1))[:160]
            return sig + (f" @ {frame.group(1)}" if frame else "")
        m = self._SIGNAL_RE.search(text)
        if m:
            frame = self._FRAME_RE.search(text)
            return f"signal {m.group(1)} ({m.group(2).strip()})" + \
                (f" @ {frame.group(1)}" if frame else "")
        return super().extract_crash_signature(stdout, stderr, return_code) or "unknown"
