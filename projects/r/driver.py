"""
projects/r/driver.py — run a fused R program under the instrumented
interpreter (projects/r/setup.py) and report whether what came back is a
bug.

Like CPython and Ruby, seeds here are *executed*: most failures are
ordinary R errors and say nothing about the interpreter. What does:

  * `*** caught segfault ***` and friends — R installs its own handler
    and prints a traceback before dying.
  * AddressSanitizer / UndefinedBehaviorSanitizer reports.
  * R's own internal-consistency errors, which arrive through the same
    channel as ordinary errors: "unimplemented type 'x' in 'y'" (an
    object reached code with no case for its SEXPTYPE), the write-barrier
    complaints that `--enable-strict-barrier` turns on, and "Internal
    error".

Not bugs, and filtered here: "C stack usage is too close to the limit"
and "node stack overflow" (R detects deep recursion and says so — a
fused program recurses without bound all the time), the sanitizer's own
stack-overflow report for the same reason, and allocation failures.

Interpreter axes drawn per execution
------------------------------------
`R_ENABLE_JIT` 0-3 selects how much of the program the byte-code
compiler compiles: 0 is the AST interpreter, 3 compiles loops and
closures on first use. Two engines that must agree on every program are
a bug source in their own right, and nothing else reaches the compiler.
`--min-vsize`/`--min-nsize` move the garbage collector's first
collection, and `R_GC_MAX_GROW_FRAC`-style tuning changes when it runs
again; a use-after-free in R shows up as a value that a collection at
the wrong moment reclaimed.
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult


class RDriver(BaseDriver):

    FUZZ_FLAGS = [
        "--no-echo", "--no-save", "--no-restore-data", "--no-environ",
        "--max-ppsize=100000", "--max-ppsize=500000",
        "--min-vsize=1M", "--min-vsize=32M", "--min-nsize=350k", "--min-nsize=2M",
    ]
    JIT_LEVELS = ["0", "1", "2", "3"]

    DEFAULT_MEM_LIMIT_MB = 3072

    _SEGV_RE = re.compile(r'\*\*\* caught (\w[\w ]*) \*\*\*')
    _ASAN_RE = re.compile(r'SUMMARY: (\w+Sanitizer): ([\w-]+)(?:[^\n]*? in ([\w:.]+))?')
    _UBSAN_RE = re.compile(r'runtime error: ([^\n]+)')
    _INTERNAL_RE = re.compile(
        r'(Internal error[^\n]*|unprotected object[^\n]*|'
        r'not safe to modify[^\n]*|attempt to set index[^\n]*)')
    # R's own frame line in a segfault traceback: `1: fun(args)`.
    _FRAME_RE = re.compile(r'^\s*\d+:\s+([A-Za-z._][\w._]*)\s*\(', re.M)

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.memory_limit_mb = int(exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB) or 0)
        self.project_dir = os.path.join(self.ffl_root, "projects", "r")
        self.rscript = os.path.join(self.project_dir, "install", "bin", "Rscript")

    def _random_flags(self):
        return " ".join(random.sample(self.FUZZ_FLAGS, random.randint(0, 2)))

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.R")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            asan = ("abort_on_error=1:detect_leaks=0:allocator_may_return_null=1:"
                    "symbolize=1:use_sigaltstack=0:detect_stack_use_after_return=0")
            if self.memory_limit_mb:
                asan += f":hard_rss_limit_mb={self.memory_limit_mb}"
            env = (f"ASAN_OPTIONS='{asan}' "
                   "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' "
                   f"R_ENABLE_JIT={random.choice(self.JIT_LEVELS)} "
                   "R_LIBS_USER=/tmp/ffl-rlibs R_PROFILE_USER=/dev/null "
                   "R_ENVIRON_USER=/dev/null R_HOME_USER=/tmp "
                   "TMPDIR=" + workdir + " HOME=/tmp ")
            cmd = (f"ulimit -c 0; {env}{self.rscript} --vanilla {self._random_flags()} "
                   f"{seed_file} < /dev/null").strip()
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
        # Deep recursion, which a fused program reaches constantly. R
        # detects it and says so; under ASan the guard page is hit first
        # and the sanitizer reports it instead. Neither is a defect.
        if re.search(r'C stack usage [^\n]* too close to the limit|'
                     r'node stack overflow|evaluation nested too deeply|'
                     r'protect\(\): protection stack overflow|'
                     r'AddressSanitizer: stack-overflow', text):
            return False
        # Allocation failure is the fused program's, not the interpreter's.
        if re.search(r'cannot allocate (?:vector|memory)|hard rss limit exhausted|'
                     r'out of memory|allocation-size-too-big|std::bad_alloc', text):
            return False
        return super()._check_crash(stdout, stderr, return_code)

    @staticmethod
    def _mask(s):
        s = re.sub(r'0x[0-9a-fA-F]+', '0x', s)
        s = re.sub(r"'[^'\n]{0,60}'", "'.'", s)
        s = re.sub(r'\b\d+\b', 'N', s)
        return re.sub(r'\s+', ' ', s).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        m = self._ASAN_RE.search(text)
        if m:
            return f"{m.group(1)}: {m.group(2)}" + (f" in {m.group(3)}" if m.group(3) else "")
        m = self._SEGV_RE.search(text)
        if m:
            frame = self._FRAME_RE.search(text)
            return f"caught {m.group(1)}" + (f" @ {frame.group(1)}" if frame else "")
        m = self._INTERNAL_RE.search(text)
        if m:
            return f"internal: {self._mask(m.group(1))}"[:200]
        m = self._UBSAN_RE.search(text)
        if m:
            return f"UBSan: {self._mask(m.group(1))}"[:200]
        return super().extract_crash_signature(stdout, stderr, return_code) or "unknown"
