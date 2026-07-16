import glob
import os
import random
import re
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class HaskellDriver(BaseDriver):
    """
    Haskell driver: type-checks and optimizes seeds via `ghc -fno-code` —
    never generates code, links, or executes the seed. FFL runs inside the
    ffl-haskell container where GHC ships pre-installed (via the
    `haskell:latest` base image) under /opt/ghc/<ver>/bin.

    `-fno-code` runs the parser, renamer, typechecker, desugarer, and the
    full Core-to-Core optimizer (strictness/demand analysis, simplifier,
    specialiser, etc.) before stopping — it genuinely exercises `-O`
    flags, unlike `runghc`'s bytecode interpreter, which explicitly warns
    "Ignoring optimization flags since they are experimental for the
    byte-code interpreter". It also doesn't require `main` to be defined
    (only a full link does), so it works uniformly on both fused programs
    (always have a synthesized `main`) and raw non-Main library-style
    seeds harvested from ghc/ghc's should_compile tests.
    Targets GHC front-end/optimizer bugs (panics/ICEs) exclusively — never
    runs the compiled program, so no RTS/runtime bug surface is covered.
    """

    # Diversifies optimization level / strictness across runs, mirroring
    # GoDriver's GCFLAGS pool.
    GHC_FLAG_SETS = [
        [],
        ["-O0"],
        ["-O1"],
        ["-O2"],
        ["-XStrict"],
        ["-fno-full-laziness"],
        ["-fno-state-hack"],
        ["-fno-omit-yields"],
        ["-feager-blackholing"],
        ["-XBangPatterns"],
    ]

    def __init__(self, config):
        super().__init__(config)
        self.ghc_bin = self._resolve_ghc()

    def _resolve_ghc(self) -> str:
        found = shutil.which("ghc")
        if found:
            return found
        matches = glob.glob("/opt/ghc/*/bin/ghc")
        if matches:
            path = sorted(matches)[-1]
            print(f"[HaskellDriver] Using direct ghc binary: {path}")
            return path
        print("[HaskellDriver] Falling back to bare 'ghc' (relying on PATH).")
        return "ghc"

    def _get_random_flags(self):
        return " ".join(random.choice(self.GHC_FLAG_SETS))

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.hs")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            flags = "" if seed.metadata.get("type") == "llm_translated" else self._get_random_flags()
            cmd = f"{self.ghc_bin} -fno-code -v0 {flags} {seed_file}".strip()
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
        combined = stderr + "\n" + stdout

        m = re.search(r"ghc: panic!\s*\(the 'impossible' happened\)([^\n]*(?:\n[^\n]*){0,3})", combined)
        if m:
            return f"ghc panic: {m.group(1).strip()[:200]}"

        m = re.search(r"GHC internal error:\s*([^\n]+)", combined)
        if m:
            return f"GHC internal error: {m.group(1).strip()}"

        m = re.search(r"internal error:\s*([^\n]+)", combined, re.IGNORECASE)
        if m:
            return f"GHC internal error: {m.group(1).strip()}"

        m = re.search(r"(RTS invariant[^\n]*)", combined)
        if m:
            return m.group(1).strip()

        if "internal inconsistency" in combined:
            return "GHC: internal inconsistency"

        m = re.search(r"(ASSERT failed![^\n]*)", combined)
        if m:
            return m.group(1).strip()

        if "Segmentation fault" in combined:
            return "ghc: Segmentation fault"

        if "out of memory" in combined.lower():
            return "ghc: out of memory"

        return super().extract_crash_signature(stdout, stderr, return_code)
