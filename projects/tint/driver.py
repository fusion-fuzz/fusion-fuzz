"""
projects/tint/driver.py — compile a fused WGSL program with tint and report
whether what came back is a bug.

This file owns execution: writing the shader, choosing the output backend
and flags, containing the run, cleaning up. Judgement lives in
projects/tint/analyzer.py.

Reaching the compiler's interesting code
----------------------------------------
tint is a translator: it reads WGSL and writes SPIR-V, MSL, HLSL or GLSL.
Front-end validation is the same for every run, but each *output* backend
is a separate body of code — the SPIR-V writer and the MSL writer share
almost nothing — and a translation bug lives in one of them. So the driver
varies the backend as its main lever, the way the naga driver does and the
way the JS drivers vary optimisation tier.

Two flags reach further than a plain translation:

  --validate
      Runs the backend's own validator on tint's output (spirv-val for
      SPIR-V, and so on). A miscompile that produces structurally invalid
      output is caught here rather than being written out silently.

  --ir-roundtrip
      Serialises tint's intermediate representation and reads it back
      before continuing. This exercises the IR binary format and, combined
      with the IR-validation asserts the build enables, is a dense source
      of internal-compiler-errors — it is the tint counterpart of Go's
      `-d=ssa/check/on`.

Every flag below was checked against src/tint/cmd/tint/main.cc in the
pinned checkout: tint rejects an unknown option, so an unverified name
would fail every run that draws it.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.tint.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_tint_analyzer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class TintDriver(BaseDriver):
    """Drives the tint executable built by projects/tint/setup.py."""

    # The output backends. Each `--format` selects a different code writer;
    # weighted towards SPIR-V (the reference backend and the one most
    # exercised) but reaching all of them, because a writer bug is specific
    # to its writer. `wgsl` round-trips WGSL through the IR and back, which
    # catches front-end/printer disagreements.
    BACKENDS = ["spirv", "msl", "hlsl", "glsl", "wgsl"]
    BACKEND_WEIGHTS = [40, 18, 18, 12, 12]

    # Verification and IR-exercising flags. Sampled, not always on: they
    # cost time, and a bug that only appears without them is still worth
    # reaching. --validate needs the backend's external validator present;
    # it is skipped silently by tint when it is not.
    EXTRA_FLAGS = [
        "--validate",
        "--ir-roundtrip",
        "--validate --ir-roundtrip",
        "--emit-ir",
    ]

    # Deliberately NOT drawn: --spirv-version 1.5. Combined with
    # --validate it reaches `TINT_UNREACHABLE() << "SPIR-V 1.5 validation
    # not yet supported"` in src/tint/cmd/tint/main.cc — a declared
    # not-implemented path, so every such run would report an ICE that is
    # not a bug. The same caution applies to any flag added later: check
    # whether it leads to a TINT_UNIMPLEMENTED before drawing it.

    # Front-end toggles that change what the compiler accepts or how it
    # lowers a program, drawn occasionally.
    FRONTEND_FLAGS = [
        "--disable-robustness",
        "--allow-non-uniform-derivatives",
        "--disable-workgroup-init",
    ]

    # A fused shader can drive tint into unbounded work. Enforced through
    # ASan's hard_rss_limit_mb, NOT `ulimit -v`: an ASan build reserves
    # ~20 TB of virtual address space for its shadow memory at startup, so
    # any `ulimit -v` low enough to be a useful cap kills tint before it
    # runs and every execution fails identically.
    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.mem_limit_mb = int(exec_cfg.get("mem_limit_mb",
                                             self.DEFAULT_MEM_LIMIT_MB))
        self.tint = os.path.join(self.ffl_root, "projects", "tint", "dawn-src",
                                 "dawn", "out", "fuzz", "tint")

    # -- flag selection ----------------------------------------------------

    def _choose_args(self, facts):
        """Assemble one execution's backend and flags."""
        backend = random.choices(self.BACKENDS, weights=self.BACKEND_WEIGHTS,
                                 k=1)[0]
        args = [f"--format {backend}"]

        if random.random() < 0.55:
            args.append(random.choice(self.EXTRA_FLAGS))
        if random.random() < 0.20:
            args.append(random.choice(self.FRONTEND_FLAGS))

        # A shader with no entry point never reaches a code writer at all —
        # there is nothing to emit — so those runs only exercise the front
        # end. Steer them to the WGSL writer, which round-trips the whole
        # module through the IR and is the one backend that still does
        # meaningful work without an entry point.
        if not facts.get("entry_stages"):
            args[0] = "--format wgsl"
            backend = "wgsl"
        return backend, args

    def _sanitizer_env(self):
        """ASan/UBSan settings, as a shell prefix.

        allocator_may_return_null=1
            a huge allocation returns null instead of aborting with an ASan
            report; fused shaders request absurd array sizes, and without
            this each is a "crash" unrelated to a tint bug.
        """
        asan = ":".join(filter(None, [
            f"hard_rss_limit_mb={self.mem_limit_mb}" if self.mem_limit_mb > 0 else "",
            "allocator_may_return_null=1",
            "symbolize=1",
            "detect_leaks=0",
            "handle_abort=1",
            "handle_segv=1",
            "handle_sigill=1",
            "print_summary=1",
            "exitcode=1",
        ]))
        return (f"ASAN_OPTIONS={asan} "
                f"UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0")

    def _build_command(self, shader, facts):
        _, args = self._choose_args(facts)
        return f"{self._sanitizer_env()} {self.tint} {shader} {' '.join(args)}"

    # -- execution ---------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            shader = os.path.join(workdir, f"{seed.id}.wgsl")
            with open(shader, "w", encoding="utf-8") as f:
                f.write(seed.content)
            cmd = self._build_command(shader, facts)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        output = f"{stdout}\n{stderr}"
        verdict = classify(output)
        result = ExecutionResult(
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            time=time.time() - start,
            crashed=verdict["is_bug"],
            signature=verdict["signature"],
        )
        result.command = cmd
        return result
