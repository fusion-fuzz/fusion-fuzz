import os
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class RustDriver(BaseDriver):
    """
    Rust driver: compiles seeds with the stage1 rustc directly.
    FFL runs inside the ffl-rust container where rust-src is mounted at /rust-src.
    The rustc binary lives at /rust-src/build/x86_64-unknown-linux-gnu/stage1/bin/rustc.
    """

    RUSTC_BIN = "/usr/local/cargo/bin/rustc"

    EDITIONS = ["2015", "2018", "2021"]
    OPT_LEVELS = ["0", "1", "2", "3", "s", "z"]
    CODEGEN_UNITS = ["1", "16"]

    def _get_random_flags(self):
        flags = [
            f"--edition={random.choice(self.EDITIONS)}",
            f"-C opt-level={random.choice(self.OPT_LEVELS)}",
            f"-C codegen-units={random.choice(self.CODEGEN_UNITS)}",
        ]
        if random.random() > 0.5:
            flags.append("-g")
        if random.random() > 0.5:
            flags.append("-C debug-assertions=yes")
        if random.random() > 0.5:
            flags.append("-C overflow-checks=yes")
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.rs")
            out_file = os.path.join(workdir, f"{seed.id}.bin")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            flags = "--edition=2021" if seed.metadata.get("type") == "llm_translated" else self._get_random_flags()
            cmd = f"{self.RUSTC_BIN} {flags} {seed_file} -o {out_file}"
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
