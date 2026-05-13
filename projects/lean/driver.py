import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class LeanDriver(BaseDriver):
    """
    Lean 4 driver: elaborates/type-checks .lean files directly.
    FFL runs inside the ffl-lean container where lean is installed via elan at
    /home/ffluser/.elan/bin/lean (or a versioned toolchain path).
    """

    _ELAN_LEAN = "/home/ffluser/.elan/bin/lean"

    LEAN_FLAGS = [
        [],
        ["--trust=0"],
        ["--threads=1"],
        ["--threads=4"],
        ["--profile"],
        ["-D", "maxRecDepth=512"],
        ["-D", "maxRecDepth=4096"],
        ["-D", "maxRecDepth=2048", "--threads=2"],
        ["--trust=0", "-D", "maxRecDepth=1024"],
        ["--threads=1", "--profile"],
    ]

    def __init__(self, config):
        super().__init__(config)
        self._lean_bin = self._resolve_lean_bin()

    def _resolve_lean_bin(self) -> str:
        import glob as _glob
        matches = _glob.glob("/home/ffluser/.elan/toolchains/*/bin/lean")
        if matches:
            path = sorted(matches)[-1]
            print(f"[LeanDriver] Using direct lean binary: {path}")
            return path
        print(f"[LeanDriver] Falling back to elan proxy: {self._ELAN_LEAN}")
        return self._ELAN_LEAN

    _IMPORT_LINE_RE = re.compile(
        r'^\s*(?:public\s+|meta\s+)?import\s+([\w][\w.]*)\s*$'
    )

    def _hoist_imports(self, content: str) -> str:
        imports: list = []
        body_lines = []
        for line in content.splitlines():
            m = self._IMPORT_LINE_RE.match(line)
            if m:
                imports.append(m.group(1))
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if not imports:
            return body
        import_block = "\n".join(f"import {mod}" for mod in sorted(set(imports)))
        return f"{import_block}\n\n{body}" if body else import_block

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.lean")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(self._hoist_imports(seed.content))
            if seed.metadata.get("type") == "llm_translated":
                flags_str = ""
            else:
                flags_str = " ".join(random.choice(self.LEAN_FLAGS))
            cmd = f"{self._lean_bin} {flags_str} {seed_file}".strip()
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
        for text in (stderr, stdout):
            m = re.search(r"INTERNAL PANIC:\s+(.*)", text)
            if m:
                return m.group(1).strip()
        for text in (stderr, stdout):
            m = re.search(r"thread '.*?' panicked at '?(.*?)'?(?:\n|$)", text)
            if m:
                return m.group(1).strip()
        if "Segmentation fault" in stderr or "Segmentation fault" in stdout:
            return "Segmentation fault"
        return super().extract_crash_signature(stdout, stderr, return_code)
