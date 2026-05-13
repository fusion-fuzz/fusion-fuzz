import random
import re
import shutil
import time
import os
from core.driver import BaseDriver, ExecutionResult


class SwiftDriver(BaseDriver):
    """
    Swift driver: invokes swift -frontend directly.
    FFL runs inside the ffl-swift container where 'swift' is in PATH.
    """

    MODES = ["-typecheck", "-emit-silgen", "-emit-sil", "-emit-ir", "-c"]
    MODE_WEIGHTS = [50, 10, 10, 20, 10]
    OPT_LEVELS = ["-Onone", "-O", "-Osize", "-wmo"]
    BASE_FLAGS = ["-sil-verify-all"]
    EXPERIMENTAL_FEATURES = [
        "-enable-experimental-feature VariadicGenerics",
        "-enable-experimental-feature Macros",
        "-enable-experimental-feature MoveOnly",
        "-enable-experimental-feature NonescapableTypes",
        "-enable-experimental-feature ThenStatements",
    ]
    MISC_FLAGS = [
        "-enable-library-evolution",
        "-strict-concurrency=complete",
        "-disable-availability-checking",
        "-enforce-exclusivity=checked",
        "-debug-info-format=dwarf",
    ]

    def _get_random_flags(self):
        flags = []
        flags.append(random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0])
        flags.append(random.choice(self.OPT_LEVELS))
        flags.extend(self.BASE_FLAGS)
        spice = self.MISC_FLAGS + self.EXPERIMENTAL_FEATURES
        flags.extend(random.sample(spice, random.randint(0, 2)))
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.swift")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            flags = "-typecheck -Onone" if getattr(self, "dryrun_mode", False) else self._get_random_flags()
            asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1:detect_stack_use_after_return=1"
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            cmd = (
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"swift -frontend {flags} {seed_file}"
            )
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
        sig = super().extract_crash_signature(stdout, stderr, return_code)
        if sig:
            return sig
        for text in (stderr, stdout):
            m = re.search(r"(Assertion failed: .*)", text)
            if m:
                return m.group(1).strip()
        for text in (stderr, stdout):
            m = re.search(r"\d+\.\s+(While evaluating request [^(]*)", text)
            if m:
                return m.group(1).strip()
        return None
