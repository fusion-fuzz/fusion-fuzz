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
    #
    # These only mean anything when GHC actually generates code. Every
    # execution used to pass `-fno-code`, which stops after typechecking:
    # measured on ghc 9.14, `-fno-code -O2 -ddump-simpl` dumps no Core at
    # all where `-O2` alone dumps 23 lines, so the simplifier — and with it
    # every flag in this list — never ran. The adapter was fuzzing the
    # parser and typechecker only. See CODEGEN_MODES.
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

    # GHC's internal IR validators — the counterpart of Go's
    # `-d=ssa/check/on` and LLVM's assertions. -dcore-lint typechecks Core
    # after each pass, -dstg-lint checks STG, -dcmm-lint checks the Cmm the
    # backend emits. A pass that builds ill-typed Core is a real GHC bug and
    # otherwise shows up, if at all, as wrong runtime behaviour much later.
    # Measured cost on a single module: 78ms without, 80ms with both.
    LINT_FLAGS = ["-dcore-lint", "-dstg-lint", "-dcmm-lint"]
    LINT_RATE = 0.75

    # `-c` compiles through the whole pipeline and writes a .o into the
    # per-execution workdir; `-fno-code` stops after typechecking. The
    # second is kept as a minority draw because it is the only way to reach
    # the frontend on a seed whose backend errors out early, but it must not
    # be the common case again.
    CODEGEN_MODES = ["-c", "-fno-code"]
    CODEGEN_WEIGHTS = [85, 15]

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
        flags = list(random.choice(self.GHC_FLAG_SETS))
        mode = random.choices(self.CODEGEN_MODES,
                              weights=self.CODEGEN_WEIGHTS, k=1)[0]
        flags.append(mode)
        # Core lint is meaningless without Core to lint.
        if mode != "-fno-code" and random.random() < self.LINT_RATE:
            flags += self.LINT_FLAGS
        return " ".join(flags)

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
            # The codegen mode now comes from _get_random_flags; -fno-code
            # is no longer forced on every execution.
            cmd = f"{self.ghc_bin} -v0 {flags} {seed_file}".strip()
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
