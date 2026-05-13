import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class MLIRDriver(BaseDriver):
    """
    MLIR driver: invokes mlir-opt directly.
    FFL runs inside the ffl-mlir container where the binary lives under
    {ffl_root}/projects/mlir/llvm-mlir-install/bin/mlir-opt.
    """

    IR2VEC_KINDS = ["symbolic", "flow-aware"]
    MIR2VEC_KINDS = ["symbolic"]
    PASSES = ["--canonicalize", "--cse", "--inline", "--symbol-dce",
              "--loop-invariant-code-motion", "--sccp"]

    # Always-on flags:
    # --split-input-file   : parse each // ----- section independently (required
    #                        for test files that encode multiple test cases).
    # --allow-unregistered-dialect : tolerate ops from dialects not compiled in.
    BASE_FLAGS = ["--split-input-file", "--allow-unregistered-dialect"]

    # Pattern matching bare `func @` (old MLIR syntax, invalid in LLVM 23+).
    # Negative lookbehind on both word chars and `.` avoids double-patching
    # already-correct `func.func @` occurrences.
    _FUNC_RE = re.compile(r'(?<![\w.])func\s+@')

    def __init__(self, config):
        super().__init__(config)
        self.mlir_opt = os.path.join(
            self.ffl_root, "projects", "mlir", "llvm-mlir-install", "bin", "mlir-opt"
        )

    def _preprocess(self, content: str) -> str:
        """Upgrade old-style `func @name` to `func.func @name` for LLVM 23+."""
        return self._FUNC_RE.sub('func.func @', content)

    def _get_random_flags(self):
        flags = []
        if random.random() > 0.3:
            num_passes = random.randint(1, len(self.PASSES))
            flags.extend(random.sample(self.PASSES, num_passes))
        if random.random() > 0.7:
            flags.append("--verify-roundtrip")
        if random.random() > 0.8:
            flags.append("--verify-each")
        if random.random() > 0.7:
            flags.append(f"--ir2vec-kind={random.choice(self.IR2VEC_KINDS)}")
        if random.random() > 0.8:
            flags.append(f"--mir2vec-kind={random.choice(self.MIR2VEC_KINDS)}")
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.mlir")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(self._preprocess(seed.content))
            asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1"
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            base = " ".join(self.BASE_FLAGS)
            flags = self._get_random_flags()
            cmd = (
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"{self.mlir_opt} {base} {flags} {seed_file}"
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
