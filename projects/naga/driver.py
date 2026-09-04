import os
import random
import re
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class NagaDriver(BaseDriver):
    """
    Naga driver: validates WGSL through naga-cli. Randomly asks Naga to
    exercise extra writer backends when available, while plain validation
    remains the common path.
    """

    def __init__(self, config):
        super().__init__(config)
        # setup.py builds naga-cli with debug assertions and overflow
        # checks on; the copy the Dockerfile's `cargo install` leaves on
        # PATH has both compiled out. Prefer the built one, and say so
        # loudly when falling back, because the difference is the whole
        # oracle rather than a detail of where the binary lives.
        built = os.path.join(self.ffl_root, "projects", "naga",
                             "wgpu", "target", "release", "naga")
        if os.path.exists(built):
            self.naga_bin = built
        else:
            self.naga_bin = shutil.which("naga") or "naga"
            print(f"[naga] built binary not found at {built}; using "
                  f"{self.naga_bin} — its debug assertions and overflow "
                  f"checks are compiled out")

    BACKENDS = [
        ("validate", ""),
        ("spv", ".spv"),
        ("metal", ".metal"),
        ("hlsl", ".hlsl"),
        ("glsl", ".vert --profile es310"),
        ("ir", ".txt"),
    ]

    def _get_random_flags(self):
        return random.choices(self.BACKENDS, weights=[55, 12, 10, 8, 8, 7], k=1)[0]

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.wgsl")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            _, output = ("validate", "") if getattr(self, "dryrun_mode", False) else self._get_random_flags()
            output_arg = ""
            if output:
                ext, *extra = output.split(" ", 1)
                output_file = os.path.join(workdir, f"{seed.id}{ext}")
                output_arg = f"{output_file}" + (f" {extra[0]}" if extra else "")
            env = (
                "RUST_BACKTRACE=1 "
                "ASAN_OPTIONS='abort_on_error=1:detect_leaks=0:symbolize=1' "
                "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1'"
            )
            cmd = f"{env} {self.naga_bin} {seed_file} {output_arg}".strip()
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
        sig = super().extract_crash_signature(stdout, stderr, return_code)
        if sig:
            return sig
        m = re.search(r"thread '.*' panicked at\s+(.+)", combined)
        if m:
            return f"Rust panic: {m.group(1).strip()}"
        m = re.search(r"panicked at\s+(.+)", combined)
        if m:
            return f"Rust panic: {m.group(1).strip()}"
        if "Segmentation fault" in combined:
            return "Segmentation fault"
        if "Aborted" in combined:
            return "Aborted"
        return None
