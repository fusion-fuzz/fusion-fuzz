import os
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class NagaDriver(BaseDriver):
    """
    Naga driver: invokes the naga binary directly.
    FFL runs inside the ffl-naga container where the binary lives under
    {ffl_root}/projects/naga/naga.
    """

    POLICIES = ["Restrict", "ReadZeroSkipWrite", "Unchecked"]
    SHADER_MODELS = ["50", "51", "60"]
    PROFILES = ["es", "core", "es330"]

    def __init__(self, config):
        super().__init__(config)
        self.naga_bin = os.path.join(self.ffl_root, "projects", "naga", "naga")

    def _get_random_flags(self):
        flags = []
        if random.random() > 0.5:
            flags.append("--validate 0")
        if random.random() > 0.3:
            flags.append(f"--index-bounds-check-policy {random.choice(self.POLICIES)}")
        if random.random() > 0.3:
            flags.append(f"--image-load-bounds-check-policy {random.choice(self.POLICIES)}")
        if random.random() > 0.5:
            flags.append(f"--shader-model {random.choice(self.SHADER_MODELS)}")
        if random.random() > 0.5:
            flags.append(f"--profile {random.choice(self.PROFILES)}")
        if random.random() > 0.5:
            flags.append("--compact")
        if random.random() > 0.5:
            flags.append("-g")
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
            cmd = f"{self.naga_bin} {flags} {seed_file}"
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
