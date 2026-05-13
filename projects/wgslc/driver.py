import os
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class WGSLC_Driver(BaseDriver):
    """
    WGSLC driver: invokes the wgslc binary directly.
    FFL runs inside the ffl-wgslc container where the binary lives under
    {ffl_root}/projects/wgslc/wgslc.
    """

    def __init__(self, config):
        super().__init__(config)
        self.wgslc_bin = os.path.join(self.ffl_root, "projects", "wgslc", "wgslc")
        self.lib_path = os.path.join(
            self.ffl_root, "projects", "wgslc", "WebKit", "build", "lib"
        )

    def _get_random_flags(self):
        flags = []
        if random.random() > 0.5:
            flags.append("--enable-shader-validation")
        if random.random() > 0.8:
            flags.append("--dump-ast-after-checking")
        return " ".join(flags)

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
            flags = self._get_random_flags()
            cmd = f"LD_LIBRARY_PATH={self.lib_path} {self.wgslc_bin} {flags} {seed_file}"
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
