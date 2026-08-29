"""
projects/rust/driver.py — compile (and sometimes run) a fused Rust program
and report whether what came back is a bug.

This file owns execution: flag selection, containment, cleanup. Judgement
lives in projects/rust/analyzer.py.

Flag selection is most of the bug-finding story here, more so than for the
C compilers, because rustc's interesting checks are opt-in per invocation:

    -Z validate-mir       re-validates MIR after every transform. The
                          counterpart of GCC's --enable-checking=rtl and
                          Go's -d=ssa/check/on. A MIR pass that produces
                          malformed MIR otherwise yields wrong codegen
                          rather than a reportable crash.
    -Z verify-llvm-ir=yes runs LLVM's verifier on what rustc emitted, which
                          is what separates "rustc built bad IR" from "LLVM
                          mishandled good IR".
    -Z mir-opt-level=4    above the default 2; the higher levels are
                          markedly less exercised.

Unsafe Rust
-----------
The unsafe surface is where a compiler bug turns into unsoundness, so
seeds that use it get a different treatment (see `_unsafe_flags`). The
three that matter:

    -Z randomize-layout    randomises field order for `repr(Rust)` types.
                           Code that assumed a layout — which is most
                           incorrect unsafe code — breaks, and so does a
                           compiler that assumed one.
    -Z strict-init-checks  makes `mem::uninitialized` on a type that cannot
                           hold uninit bytes an error rather than UB.
    -Z sanitizer=address   instruments the *generated* program, so running
                           it turns a miscompile into a diagnosable report
                           instead of a wrong answer.

`-C debug-assertions=yes` belongs in the same list: it switches on the
standard library's own precondition checks (slice::from_raw_parts
alignment, `unreachable_unchecked`, NonNull validity), which are exactly
the invariants unsafe code violates.

Running the program
-------------------
Compiling alone cannot see a miscompile. When a seed has a `fn main`, no
`known-bug` marker and enough unsafe constructs to be worth it, the child
is built and run under a short timeout. A plain panic from the program is
*not* a finding — the program is doing what it was written to do — so
analyzer.classify only treats sanitizer and Miri output as bugs there.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.rust.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_rust_analyzer",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class RustDriver(BaseDriver):
    """Drives the stage1 rustc built by projects/rust/setup.py."""

    EDITIONS = ["2015", "2018", "2021", "2024"]
    OPT_LEVELS = ["0", "1", "2", "3", "s", "z"]
    CODEGEN_UNITS = ["1", "4", "16", "256"]
    LTO = ["off", "thin", "fat"]
    PANIC = ["unwind", "abort"]

    # What to ask rustc to produce. Stopping at metadata only type-checks;
    # each later stage brings in more of the compiler, and `link` is the
    # only one that can produce a runnable binary.
    EMITS = ["metadata", "mir", "llvm-ir", "asm", "obj", "link"]
    EMIT_WEIGHTS = [30, 8, 12, 10, 15, 25]

    # -Z flags that check rustc's own output. Weighted so the two verifiers
    # are common and the rest sample the space.
    VERIFIERS = ["-Zvalidate-mir", "-Zverify-llvm-ir=yes"]

    # Every entry is checked against the built rustc, because an unknown
    # -Z flag makes rustc reject the whole invocation — the execution is
    # spent producing "error: unknown unstable option" instead of
    # compiling anything. Three plausible-looking flags were removed after
    # exactly that: -Zpolymorphize (gone), -Zmir-emit-retag (gone), and
    # -Zno-parallel-backend (renamed to --jobs-backend).
    NIGHTLY_FLAGS = [
        "-Zmir-opt-level=0", "-Zmir-opt-level=1", "-Zmir-opt-level=3",
        "-Zmir-opt-level=4",
        "-Zinline-mir=yes", "-Zinline-mir=no",
        "-Zdylib-lto",
        "-Zshare-generics=yes", "-Zshare-generics=no",
        "-Zprint-type-sizes",
        "-Zthreads=2", "-Zthreads=4",
    ]

    # Applied to seeds that actually use unsafe. See the module docstring.
    UNSAFE_FLAGS = [
        "-Zrandomize-layout",
        "-Zstrict-init-checks",
        "-Zsanitizer=address",
        "-Zsanitizer=leak",
        "-Cdebug-assertions=yes",
    ]

    # Cross targets rustc can emit for with only the stage1 std we built.
    # Anything needing a foreign linker is kept out: those fail in the
    # linker, which teaches nothing about the compiler. `--emit=link` is
    # forced off the host target for the same reason.
    CROSS_TARGETS = [
        "x86_64-unknown-linux-gnu",
        "i686-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "wasm32-unknown-unknown",
        "thumbv7em-none-eabihf",
    ]

    DEFAULT_MEM_LIMIT_MB = 6144      # rustc is hungrier than gcc or go
    RUN_TIMEOUT_S = 5

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.mem_limit_kb = int(exec_cfg.get("mem_limit_mb",
                                             self.DEFAULT_MEM_LIMIT_MB)) * 1024
        # Ask setup.py where the compiler is rather than duplicating the
        # triple-dependent path here.
        import importlib.util as ilu
        proj = os.path.join(self.ffl_root, "projects", "rust")
        spec = ilu.spec_from_file_location("ffl_rust_setup",
                                           os.path.join(proj, "setup.py"))
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rustc = mod.rustc_path(proj)
        self.host_triple = mod.host_triple()
        self.sysroot = os.path.join(proj, "rust-src", "build",
                                    self.host_triple, "stage1")

    # ── command construction ──────────────────────────────────────────

    def _unsafe_flags(self, facts):
        """Extra flags for seeds that use unsafe Rust.

        Gated on the seed actually containing unsafe constructs: the
        sanitizers roughly triple compile time, and spending that on a seed
        with no raw pointers buys nothing. `unsafe_score` counts unsafe
        blocks, raw pointer types, transmute/MaybeUninit/union uses.
        """
        if facts["unsafe_score"] <= 0:
            return []
        # More unsafe means more of these are worth paying for.
        k = 1 if facts["unsafe_score"] < 3 else 2
        return random.sample(self.UNSAFE_FLAGS, min(k, len(self.UNSAFE_FLAGS)))

    def _flags(self, facts, emit, target):
        flags = [f"--edition={facts['edition'] or random.choice(self.EDITIONS)}",
                 f"-Copt-level={random.choice(self.OPT_LEVELS)}",
                 f"-Ccodegen-units={random.choice(self.CODEGEN_UNITS)}"]

        # The verifiers are the point of the exercise; keep them common.
        for v in self.VERIFIERS:
            if random.random() < 0.6:
                flags.append(v)
        flags.extend(random.sample(self.NIGHTLY_FLAGS, random.randint(0, 3)))

        if random.random() < 0.4:
            flags.append(f"-Clto={random.choice(self.LTO)}")
        if random.random() < 0.3:
            flags.append(f"-Cpanic={random.choice(self.PANIC)}")
        if random.random() < 0.5:
            flags.append(f"-Coverflow-checks={random.choice(['yes', 'no'])}")
        if random.random() < 0.4:
            flags.append(f"-Cdebug-assertions={random.choice(['yes', 'no'])}")
        if random.random() < 0.2:
            flags.append("-g")

        flags.extend(self._unsafe_flags(facts))
        # The seed's own compile-flags last so they win any conflict: they
        # are what the test was written to exercise.
        flags.extend(facts["compile_flags"])

        flags.append(f"--emit={emit}")
        if target != self.host_triple:
            flags.append(f"--target={target}")
        return flags

    def _pick_target(self, emit, facts):
        # Linking for a cross target needs that target's linker, which is
        # not installed; those failures happen outside the compiler.
        if emit == "link":
            return self.host_triple
        if random.random() < 0.75:
            return self.host_triple
        return random.choice(self.CROSS_TARGETS)

    def _should_run(self, facts, emit):
        """Whether to execute the built binary.

        Only for programs that can actually be built and run, and only when
        the seed uses unsafe: a run costs a process and a timeout, and for
        safe Rust a miscompile that a plain run would notice is rare enough
        that the compile budget is better spent elsewhere. `known-bug`
        seeds are skipped — they are supposed to fail.
        """
        return (emit == "link" and facts["has_main"] and not facts["no_main"]
                and not facts["is_known_bug"] and facts["unsafe_score"] > 0)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            seed_file = os.path.join(workdir, f"{seed.id}.rs")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)

            emit = random.choices(self.EMITS, weights=self.EMIT_WEIGHTS, k=1)[0]
            target = self._pick_target(emit, facts)
            flags = self._flags(facts, emit, target)
            out_bin = os.path.join(workdir, "a.out")
            out = f"-o {out_bin}" if emit == "link" else f"--out-dir {workdir}"

            env = (
                # A stage1 rustc is not a released one, so nightly features
                # and -Z flags need this to be accepted.
                "RUSTC_BOOTSTRAP=1 "
                # Turns an ICE backtrace from addresses into symbol names,
                # which is most of a crash signature.
                "RUST_BACKTRACE=1 "
                f"LD_LIBRARY_PATH={self.sysroot}/lib:$LD_LIBRARY_PATH "
            )
            build = (f"ulimit -v {self.mem_limit_kb}; ulimit -c 0; "
                     f"{env}{self.rustc} {' '.join(flags)} {out} {seed_file}")

            if self._should_run(facts, emit):
                asan = "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1"
                # `|| true` on the run: a non-zero exit from the *program*
                # is the program's business, and analyzer.classify only
                # treats sanitizer/Miri output as a finding here.
                cmd = (f"{build} && (ulimit -v {self.mem_limit_kb}; "
                       f"{asan} timeout {self.RUN_TIMEOUT_S} {out_bin} || true)")
            else:
                cmd = build

            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        verdict = classify((stderr or "") + "\n" + (stdout or ""))
        res = ExecutionResult(rc, stdout, stderr, time.time() - start,
                              verdict["is_bug"], verdict["signature"])
        res.command = cmd
        # _save_crash_bundle rewrites this exact path to "$SCRIPT_DIR/test.rs"
        # when it writes test.sh; without it the fallback substitutes the
        # bare seed id and the saved reproducer is unrunnable.
        res.seed_file = seed_file
        return res

    # ── crash oracle ──────────────────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        return classify((stderr or "") + "\n" + (stdout or ""))["is_bug"]

    def extract_crash_signature(self, stdout, stderr, return_code):
        sig = classify((stderr or "") + "\n" + (stdout or ""))["signature"]
        return sig or super().extract_crash_signature(stdout, stderr, return_code)
