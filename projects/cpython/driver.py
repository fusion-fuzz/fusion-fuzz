import os
import random
import re
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class CPythonDriver(BaseDriver):
    """
    CPython driver: invokes the ASAN/debug-instrumented CPython binary directly.
    FFL runs inside the ffl-cpython container where the binary lives under
    {ffl_root}/projects/cpython/cpython/build/python.
    """

    FUZZ_FLAGS = [
        "-b", "-bb", "-B", "-E", "-I", "-O", "-OO",
        "-P", "-s", "-S", "-u",
        "-X showrefcount",
        "-X tracemalloc",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.python_bin = os.path.join(
            self.ffl_root, "projects", "cpython", "cpython", "build", "python"
        )
        self.memory_limit_mb = int(
            config.get("execution", {}).get("memory_limit_mb", 512) or 0
        )

    def _get_random_flags(self):
        n = random.randint(0, 3)
        if n == 0:
            return ""
        return " ".join(random.sample(self.FUZZ_FLAGS, n))

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.py")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            asan_opts = "abort_on_error=1:detect_leaks=0:allocator_may_return_null=1"
            if self.memory_limit_mb:
                asan_opts += f":hard_rss_limit_mb={self.memory_limit_mb}"
            flags = ""  # self._get_random_flags()
            cmd = f"ASAN_OPTIONS='{asan_opts}' {self.python_bin} {flags} {seed_file}".strip()
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        duration = time.time() - start
        crashed = self._check_crash(stdout, stderr, rc)
        sig = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
        res = ExecutionResult(rc, stdout, stderr, duration, crashed, sig)
        res.command = cmd
        res.seed_file = seed_file
        return res

    def extract_crash_signature(self, stdout, stderr, return_code):
        m = re.search(r"SUMMARY: AddressSanitizer:\s+(.*)", stderr)
        if m:
            return m.group(1).strip()
        m = re.search(r"Fatal Python error:\s+(.*)", stderr)
        if m:
            return m.group(1).strip()
        m = re.search(r": Assertion `(.*)` failed", stderr)
        if m:
            return f"Assertion: {m.group(1).strip()}"
        if "Bus error" in stderr:
            return "Bus error"
        if "Segmentation fault" in stderr:
            return "Segmentation fault"
        return super().extract_crash_signature(stdout, stderr, return_code)
