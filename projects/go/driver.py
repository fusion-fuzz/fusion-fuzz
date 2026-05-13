import os
import random
import re
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class GoDriver(BaseDriver):
    """
    Go driver: compiles seeds with the Go toolchain directly.
    FFL runs inside the ffl-go container where the binary lives under
    {ffl_root}/projects/go/go/bin/go.
    Targets compiler bugs (panics, ICEs) — never executes the compiled binary.
    """

    GCFLAGS = [
        "",
        "",
        "-gcflags=all=-e",
        "-gcflags=all=-N",
        "-gcflags=all=-l",
        "-gcflags=all=-N -gcflags=all=-l",
        "-gcflags=all=-d=checkptr",
        "-gcflags=all=-d=ssa/check_bce/debug=1",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.go_bin = os.path.join(
            self.ffl_root, "projects", "go", "go", "bin", "go"
        )
        self.go_cache = os.path.join(self.fused_base, "go_cache")
        self.go_tmp = os.path.join(self.fused_base, "go_tmp")

    def _get_random_flags(self):
        return random.choice(self.GCFLAGS)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.go")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            flags = "" if seed.metadata.get("type") == "llm_translated" else self._get_random_flags()
            env = (
                f"GOCACHE={self.go_cache} GOTMPDIR={self.go_tmp}"
                f" GOMEMLIMIT=1073741824 GOGC=50"
            )
            cmd = (
                f"mkdir -p {self.go_cache} {self.go_tmp} && "
                f"{env} {self.go_bin} build -o /dev/null {flags} {seed_file}"
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
        combined = stderr + stdout
        m = re.search(r"(internal compiler error:[^\n]*)", combined)
        if m:
            return m.group(1).strip()
        m = re.search(r"panic:\s+([^\n]+)", combined)
        if m:
            return f"compiler panic: {m.group(1).strip()}"
        m = re.search(r"fatal error:\s+([^\n]+)", combined)
        if m:
            return f"compiler fatal: {m.group(1).strip()}"
        if "Segmentation fault" in combined:
            return "compiler: Segmentation fault"
        return super().extract_crash_signature(stdout, stderr, return_code)
