"""
projects/ruby/driver.py — run a fused Ruby program under the instrumented
interpreter (projects/ruby/setup.py) and report whether what came back is
a bug.

Like CPython, seeds here are *executed*: most failures are ordinary Ruby
exceptions and say nothing about the interpreter. What does:

  * `[BUG]` — rb_bug(): the VM found its own invariants broken and dumps
    a control-frame and C-level backtrace. The message after `[BUG]` and
    the first C frame make the signature.
  * `Assertion Failed: file:line:func:cond` — RUBY_ASSERT, compiled in by
    -DRUBY_DEBUG=1.
  * ASan / UBSan reports.

ASan's `stack-overflow` is *not* a finding: Ruby detects deep recursion by
catching the guard-page fault and raising SystemStackError, and under
ASan the sanitizer sees the fault first. Every fused program that recurses
without bound would otherwise be reported.

Interpreter flags
-----------------
Drawn per execution: the two parsers (`--parser=parse.y` / `prism`),
frozen-string-literal on/off, warning levels, `--disable-gems`. The
parser switch is the important one — two front ends that must agree on
every program are a bug source in their own right.
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult


class RubyDriver(BaseDriver):

    FUZZ_FLAGS = [
        "--disable-gems", "--disable=did_you_mean", "--disable=error_highlight",
        "-W0", "-W1", "-W2", "-W:deprecated", "-W:performance",
        "--enable-frozen-string-literal", "--disable-frozen-string-literal",
        "--debug-frozen-string-literal", "--backtrace-limit=2",
    ]
    PARSER_FLAGS = ["--parser=parse.y", "--parser=prism"]

    DEFAULT_MEM_LIMIT_MB = 2048

    _BUG_RE = re.compile(r'\[BUG\]\s*(.+)')
    _CFRAME_RE = re.compile(r'^\S+\(([A-Za-z_]\w*)\+0x[0-9a-f]+\)', re.M)
    _ASSERT_RE = re.compile(r'Assertion Failed:\s*([^\n]+)')
    _ASAN_RE = re.compile(r'SUMMARY: (\w+Sanitizer): ([\w-]+)(?: [^\n]*? in ([A-Za-z_]\w*))?')
    _UBSAN_RE = re.compile(r'runtime error: ([^\n]+)')

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.memory_limit_mb = int(exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB) or 0)
        self.project_dir = os.path.join(self.ffl_root, "projects", "ruby")
        self.ruby_bin = os.path.join(self.project_dir, "install", "bin", "ruby")

    def _random_flags(self):
        flags = random.sample(self.FUZZ_FLAGS, random.randint(0, 2))
        if random.random() < 0.7:
            flags.append(random.choice(self.PARSER_FLAGS))
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.rb")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            asan = ("abort_on_error=1:detect_leaks=0:allocator_may_return_null=1:"
                    "symbolize=1:use_sigaltstack=0")
            if self.memory_limit_mb:
                asan += f":hard_rss_limit_mb={self.memory_limit_mb}"
            env = (f"ASAN_OPTIONS='{asan}' "
                   "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' "
                   "RUBYOPT= GEM_HOME=/tmp/ffl-gems HOME=/tmp ")
            cmd = (f"ulimit -c 0; {env}{self.ruby_bin} {self._random_flags()} "
                   f"-I {self.project_dir} -rffl_shim {seed_file} < /dev/null").strip()
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
        if "AddressSanitizer: stack-overflow" in text:
            return False
        # Memory exhaustion is the fused program's, not the interpreter's:
        # ASan's rss cap fires, Ruby's ASan death callback then prints
        # `[BUG] ASAN error` with a full frame dump (the first bundle of
        # the first run was exactly this).
        # `TRY_WITH_GC: could not allocate: <n> bytes` is the same class
        # written by Ruby's own allocator: the program asked for more
        # memory than the machine has (34 GB in the bundle that prompted
        # this), the GC ran and could still not satisfy it, and rb_bug
        # reports the failure. A fused program is free to compute a
        # large size; the interpreter behaved correctly.
        if re.search(r'hard rss limit exhausted|out of memory|allocation-size-too-big|'
                     r'requested allocation size .* exceeds maximum|failed to allocate|'
                     r'TRY_WITH_GC: could not allocate', text):
            return False
        # A seed may *print* `[BUG]` or a sanitizer line (test_rubyoptions
        # asserts on them); only a real report carries the frame dump.
        if "[BUG]" in text and "-- Control frame information" not in text \
                and "-- C level backtrace" not in text and "Assertion Failed" not in text:
            return False
        return super()._check_crash(stdout, stderr, return_code)

    @staticmethod
    def _mask(s):
        s = re.sub(r'0x[0-9a-fA-F]+', '0x', s)
        s = re.sub(r'\b\d+\b', 'N', s)
        return re.sub(r'\s+', ' ', s).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        m = self._ASSERT_RE.search(text)
        if m:
            parts = m.group(1).strip().split(":")
            # file:line:function:condition → drop the line number
            if len(parts) >= 4:
                return f"assert {os.path.basename(parts[0])}:{parts[2]}:{self._mask(':'.join(parts[3:]))}"[:200]
            return f"assert {self._mask(m.group(1))}"[:200]
        m = self._BUG_RE.search(text)
        if m:
            msg = self._mask(m.group(1).split("ruby ")[0])
            frame = self._CFRAME_RE.search(text)
            return f"[BUG] {msg}" + (f" @ {frame.group(1)}" if frame else "")
        m = self._ASAN_RE.search(text)
        if m:
            return f"{m.group(1)}: {m.group(2)}" + (f" in {m.group(3)}" if m.group(3) else "")
        m = self._UBSAN_RE.search(text)
        if m:
            return f"UBSan: {self._mask(m.group(1))}"[:200]
        return super().extract_crash_signature(stdout, stderr, return_code) or "unknown"
