import json
import logging
import os
import re
import random
import shutil
import subprocess
import tempfile
import time
from core.driver import BaseDriver, ExecutionResult

logger = logging.getLogger("FFL.Driver.flang")


class FlangDriver(BaseDriver):
    """
    Flang driver: invokes the flang built from llvm-project trunk by
    projects/flang/setup.py (assertions, ABI-breaking checks, MLIR pattern
    checks and — unless disabled — an ASan+UBSan instrumented compiler).
    It falls back to whatever `flang` is on PATH so an unbuilt container
    still runs.

    Flang shares clang's LLVM-based driver and crash-reporting
    infrastructure, so the crash signatures look the same (Stack dump,
    LLVM ERROR, Assertion, ...).
    """

    FLANG_BIN = "flang"

    # flang only accepts -O0..-O3 (no -Os/-Oz, unlike clang).
    OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3"]

    # Compilation "depth" to exercise through the *driver*: syntax-only is
    # cheap and hits the parser/semantics most often; the others push
    # further into lowering/CodeGen.
    MODES = ["-fsyntax-only", "-emit-llvm -S -o /dev/null",
             "-S -o /dev/null", "-c -o /dev/null"]
    MODE_WEIGHTS = [40, 20, 20, 20]

    # Frontend (-fc1) actions, which the driver cannot reach. Each stops the
    # pipeline at a different stage and prints that stage's own
    # representation, so a bug in one of those printers — or in the
    # HLFIR/FIR the middle stages build — surfaces here and nowhere else.
    # flang's own test corpus is written almost entirely against these
    # (`%flang_fc1 -fdebug-unparse-with-symbols %s`), which is how the
    # seeds exercise them upstream.
    FC1_ACTIONS = [
        "-fsyntax-only",
        "-fdebug-unparse",
        "-fdebug-unparse-with-symbols",
        "-fdebug-dump-symbols",
        "-fdebug-dump-parse-tree",
        "-fdebug-dump-provenance",
        "-fdebug-dump-all",
        "-fdebug-pre-fir-tree",
        "-fdebug-dump-pft",
        "-emit-hlfir -o /dev/null",
        "-emit-fir -o /dev/null",
        "-emit-mlir -o /dev/null",
        "-emit-llvm -o /dev/null",
    ]
    FC1_WEIGHTS = [10, 8, 8, 8, 8, 5, 5, 6, 6, 12, 12, 6, 6]

    #: Share of executions that go through `flang -fc1` instead of the
    #: driver. Kept a minority: the driver path is what real users hit and
    #: it is the only one that runs the full backend.
    FC1_RATE = 0.25

    # Only f2018 is currently accepted by flang's -std=.
    STD_VALUES = ["f2018"]

    # Triples whose backends projects/flang/setup.py builds. Each brings its
    # own data layout, kind mapping (REAL(16), vector types) and lowering
    # path, and the seed corpus carries target-specific tests for all of
    # them. armv7 is deliberately absent: flang answers it with "not yet
    # implemented: target not implemented" for every input.
    TARGETS = [
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "powerpc64le-unknown-linux-gnu",
        "riscv64-unknown-linux-gnu",
    ]
    TARGET_RATE = 0.35

    # Feature flags. Grouped only for readability; one flat sample is drawn
    # from the whole list. DRIVER_ONLY_FLAGS holds the ones the *driver*
    # accepts but `-fc1` does not — the driver translates them into
    # frontend options (`-ffast-math` becomes `-menable-no-nans` and
    # friends), so passing the user-facing spelling straight to -fc1 is an
    # "unknown argument" and burns the execution.
    MISC_FLAGS = [
        # Default kinds and storage
        "-fdefault-real-8", "-fdefault-integer-8",
        "-fno-automatic", "-finit-global-zero", "-fsave-main-program",
        "-flarge-sizes", "-fstack-arrays",
        # Source dialect
        "-fbackslash", "-fimplicit-none", "-falternative-parameter-statement",
        "-flogical-abbreviations", "-fxor-operator", "-funderscoring",
        "-fno-underscoring", "-fhermetic-module-files",
        # Floating point semantics — each changes what lowering is allowed
        # to fold, and the fast-math family is where flang's constant
        # folder and the LLVM backend disagree most often.
        "-ffast-math", "-freciprocal-math", "-fno-signed-zeros",
        "-fapprox-func", "-ffp-contract=fast", "-ffp-contract=off",
        "-fcomplex-arithmetic=basic", "-fcomplex-arithmetic=improved",
        "-fcomplex-arithmetic=full",
        # Optimisation shape
        "-funroll-loops", "-fno-unroll-loops", "-floop-interchange",
        "-fversion-loops-for-stride",
        # Array semantics
        "-frealloc-lhs", "-fno-realloc-lhs",
        # Diagnostics: -pedantic turns on flang's standard-conformance
        # portability warnings, which run extra semantic checks.
        "-pedantic",
    ]

    DRIVER_ONLY_FLAGS = [
        "-fdefault-double-8", "-fassociative-math", "-fno-honor-nans",
        "-fno-honor-infinities", "-fvectorize", "-fno-vectorize",
        "-g", "-gline-tables-only",
    ]

    # Directive-language support. Held apart from MISC_FLAGS and drawn
    # together with a version because a large slice of the corpus is
    # OpenMP/OpenACC tests whose directives are inert comments without
    # these — the semantic checks for them are simply never reached.
    OPENMP_VERSIONS = ["45", "50", "51", "52"]
    DIRECTIVE_RATE = 0.2

    # Lowering path selection. HLFIR is the default and the deprecated
    # direct-to-FIR path is still reachable; running both means a bug in
    # either is found, and a crash in only one of them localises itself.
    LOWERING_FLAGS = ["-flang-experimental-hlfir", "-flang-deprecated-no-hlfir"]
    LOWERING_RATE = 0.15

    # Extra verifiers, off by default in a release pipeline. These are pure
    # oracles: they cost time and can only ever turn a silently-corrupt
    # data structure into a loud failure.
    VERIFIER_FLAGS = [
        "-mllvm -verify-dom-info",
        "-mllvm -verify-loop-info",
        "-mllvm -verify-scev",
        "-mllvm -verify-region-info",
        "-mllvm -verify-machineinstrs",
    ]
    VERIFIER_RATE = 0.25

    _FIXED_FORM_EXTS = (".f", ".F")

    #: Address-space cap for a non-sanitized build. Fuzzer inputs routinely
    #: hit runaway allocation in the constant folder; without a cap one seed
    #: can take the whole container down through the system OOM killer.
    DEFAULT_MEM_LIMIT_MB = 3072

    #: Applied to execution.timeout when the compiler is sanitizer-built and
    #: the config did not set a timeout of its own.
    SANITIZED_TIMEOUT_FACTOR = 3

    def __init__(self, config):
        super().__init__(config)
        mem_limit_mb = config.get('execution', {}).get(
            'mem_limit_mb', self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_mb = int(mem_limit_mb)
        self.build_info = self._load_build_info()
        self.flang_bin = self.build_info.get("binary") or self.FLANG_BIN
        self.sanitized = bool(self.build_info.get("sanitizers"))
        if self.sanitized and 'timeout' not in config.get('execution', {}):
            # An ASan+UBSan compiler runs 2-4x slower. Left at the default,
            # the -O3 codegen modes would mostly time out, which reads as a
            # hang rather than as the cost of the instrumentation.
            self.timeout *= self.SANITIZED_TIMEOUT_FACTOR
        self._apply_probe()


    #: Tiny well-formed program used to ask the compiler which flags it
    #: still accepts.
    _PROBE_SOURCE = "program p\n  implicit none\n  integer :: i\n  i = 1\nend program p\n"

    #: Flags that are only legal alongside another one. Probed together, or
    #: the probe would conclude the flag is unsupported and drop it.
    _PROBE_COMPANIONS = {"-fdefault-double-8": "-fdefault-real-8"}

    def _probe_flags(self):
        """Keep only the flags this particular flang accepts.

        flang's option set moves with trunk — `-flang-experimental-hlfir`
        and `-flang-deprecated-no-hlfir` existed in LLVM 22 and are gone in
        24, and a driver that keeps drawing them spends every such run on
        "unknown argument" instead of on the test case (measured: 14% of
        executions on the first trunk build). Probing once per binary and
        caching the answer means the pools degrade quietly instead of
        silently taxing the campaign.
        """
        cache_path = os.path.join(self.ffl_root, "projects", "flang",
                                  "flag-support.json")
        try:
            stamp = str(os.path.getmtime(self.flang_bin))
        except OSError:
            stamp = ""
        key = f"{self.flang_bin}@{stamp}"
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("key") == key:
                return cached["accepted"]
        except (OSError, ValueError, KeyError):
            pass

        try:
            usable = subprocess.run([self.flang_bin, "--version"],
                                    capture_output=True, timeout=120).returncode == 0
        except Exception:
            usable = False
        if not usable:
            # No compiler to ask (an unbuilt checkout, or a unit test on a
            # machine with no flang). Leave every pool as declared rather
            # than concluding that nothing is supported.
            logger.warning("cannot run %s — keeping the declared flag pools",
                           self.flang_bin)
            return {}

        workdir = tempfile.mkdtemp(prefix="ffl-flang-probe-")
        probe = os.path.join(workdir, "probe.f90")
        with open(probe, "w") as fh:
            fh.write(self._PROBE_SOURCE)

        def accepts(prefix_args, flag):
            try:
                companion = self._PROBE_COMPANIONS.get(flag, "")
                done = subprocess.run(
                    f"{self._env_prefix()}{self.flang_bin} {prefix_args} "
                    f"{flag} {companion} {probe} >/dev/null",
                    shell=True, capture_output=True, text=True, timeout=120,
                    cwd=workdir)
            except Exception:
                return False
            return done.returncode == 0

        accepted = {}
        try:
            for name, flags, prefix_args in (
                    ("MISC_FLAGS", self.MISC_FLAGS, "-fsyntax-only"),
                    ("DRIVER_ONLY_FLAGS", self.DRIVER_ONLY_FLAGS, "-fsyntax-only"),
                    ("LOWERING_FLAGS", self.LOWERING_FLAGS, "-fsyntax-only"),
                    ("VERIFIER_FLAGS", self.VERIFIER_FLAGS, "-S -o /dev/null"),
                    ("TARGETS", [f"--target={t}" for t in self.TARGETS], "-c -o /dev/null"),
                    ("FC1_ACTIONS", self.FC1_ACTIONS, "-fc1"),
            ):
                kept = [f for f in flags if accepts(prefix_args, f)]
                dropped = [f for f in flags if f not in kept]
                if dropped:
                    logger.warning("flang does not accept %s: %s",
                                   name, " ".join(dropped))
                accepted[name] = kept
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        # MISC_FLAGS is drawn for -fc1 too, so anything the frontend
        # rejects has to move to the driver-only pool rather than vanish.
        try:
            with open(cache_path, "w") as fh:
                json.dump({"key": key, "accepted": accepted}, fh, indent=1)
        except OSError:
            pass
        return accepted

    def _apply_probe(self):
        accepted = self._probe_flags()
        self.MISC_FLAGS = accepted.get("MISC_FLAGS") or self.MISC_FLAGS
        self.DRIVER_ONLY_FLAGS = accepted.get("DRIVER_ONLY_FLAGS", self.DRIVER_ONLY_FLAGS)
        self.LOWERING_FLAGS = accepted.get("LOWERING_FLAGS", self.LOWERING_FLAGS)
        self.VERIFIER_FLAGS = accepted.get("VERIFIER_FLAGS", self.VERIFIER_FLAGS)
        self.FC1_ACTIONS = accepted.get("FC1_ACTIONS") or self.FC1_ACTIONS
        targets = accepted.get("TARGETS")
        if targets:
            self.TARGETS = [t.split("=", 1)[1] for t in targets]
        # FC1_WEIGHTS is positional; rebuild it against whatever survived.
        if len(self.FC1_ACTIONS) != len(type(self).FC1_ACTIONS):
            weights = dict(zip(type(self).FC1_ACTIONS, type(self).FC1_WEIGHTS))
            self.FC1_WEIGHTS = [weights.get(a, 5) for a in self.FC1_ACTIONS]

    def _load_build_info(self):
        """What projects/flang/setup.py recorded about the build it made."""
        path = os.path.join(self.ffl_root, "projects", "flang", "build-info.json")
        try:
            with open(path) as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            return {}
        binary = info.get("binary")
        if binary and not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            # Stale record (build tree wiped) — fall back to PATH rather
            # than failing every execution with "no such file".
            info.pop("binary", None)
        return info

    def _env_prefix(self):
        """Memory cap plus sanitizer options, as a shell prefix.

        `ulimit -v` and ASan are mutually exclusive: ASan maps terabytes of
        virtual address space for its shadow memory, so a virtual-size cap
        kills an instrumented flang at startup, before it has read a single
        line of the test case. Against a sanitized build the cap has to be
        ASan's own RSS limit instead."""
        asan = ["abort_on_error=1", "detect_leaks=0", "symbolize=1"]
        ubsan = ["print_stacktrace=1", "halt_on_error=1"]
        prefix = ""
        if self.sanitized:
            asan.append(f"hard_rss_limit_mb={self.mem_limit_mb}")
            # Make an over-cap allocation fail the way it does without a
            # sanitizer — malloc returns null and LLVM reports "out of
            # memory" — instead of ASan aborting, so the two builds agree
            # on what resource exhaustion looks like.
            asan.append("allocator_may_return_null=1")
        else:
            prefix = f"ulimit -v {self.mem_limit_mb * 1024}; "
        return (f"{prefix}ASAN_OPTIONS='{':'.join(asan)}' "
                f"UBSAN_OPTIONS='{':'.join(ubsan)}' ")

    def _lang_flags(self, ext):
        return ["-ffixed-form"] if ext in self._FIXED_FORM_EXTS else ["-ffree-form"]

    def _shared_flags(self, ext, driver_mode):
        """Flags meaningful to both entry points; `driver_mode` adds the ones
        only the driver understands."""
        flags = list(self._lang_flags(ext))
        if random.random() > 0.5:
            flags.append(f"-std={random.choice(self.STD_VALUES)}")
        if random.random() < self.DIRECTIVE_RATE:
            if random.random() < 0.6:
                flags.append("-fopenmp")
                if random.random() < 0.5:
                    flags.append(f"-fopenmp-version={random.choice(self.OPENMP_VERSIONS)}")
            else:
                flags.append("-fopenacc")
        if self.LOWERING_FLAGS and random.random() < self.LOWERING_RATE:
            flags.append(random.choice(self.LOWERING_FLAGS))

        pool = self.MISC_FLAGS + (self.DRIVER_ONLY_FLAGS if driver_mode else [])
        misc = random.sample(pool, random.randint(0, 4))
        # flang rejects -fdefault-double-8 on its own ("requires
        # -fdefault-real-8"), so drawing it alone makes the run fail for a
        # reason that has nothing to do with the test case.
        if "-fdefault-double-8" in misc and "-fdefault-real-8" not in misc:
            misc.append("-fdefault-real-8")
        flags.extend(misc)
        return flags

    def _get_random_flags(self, ext):
        """Driver-mode command line."""
        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0]]
        flags.append(random.choice(self.OPT_LEVELS))
        if self.TARGETS and random.random() < self.TARGET_RATE:
            flags.append(f"--target={random.choice(self.TARGETS)}")
        flags.extend(self._shared_flags(ext, driver_mode=True))
        if self.VERIFIER_FLAGS and random.random() < self.VERIFIER_RATE:
            flags.append(random.choice(self.VERIFIER_FLAGS))
        return " ".join(flags)

    def _get_random_fc1_flags(self, ext):
        """`-fc1` command line: one frontend action plus the shared flags.

        -fc1 takes `-triple`, not `--target=`, and rejects the driver's
        `-mllvm` passthrough, so neither is drawn here."""
        flags = ["-fc1", random.choices(self.FC1_ACTIONS, weights=self.FC1_WEIGHTS, k=1)[0]]
        flags.append(random.choice(self.OPT_LEVELS))
        if self.TARGETS and random.random() < self.TARGET_RATE:
            flags.append(f"-triple {random.choice(self.TARGETS)}")
        flags.extend(self._shared_flags(ext, driver_mode=False))
        return " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            ext = seed.metadata.get("extension") or ".f90"
            seed_file = os.path.join(workdir, f"{seed.id}{ext}")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)

            if random.random() < self.FC1_RATE:
                flags = self._get_random_fc1_flags(ext)
            else:
                flags = self._get_random_flags(ext)
            # stdout carries only what was asked for — assembly, or a
            # -fdebug-dump-* rendering of the whole program, which for a
            # fused seed runs to megabytes. Every diagnostic and every
            # crash report (the `fatal internal error:` line, the stack
            # dump, the sanitizer report) goes to stderr, so discarding
            # stdout costs no oracle and keeps one runaway dump from
            # dominating the run's memory.
            cmd = (
                f"{self._env_prefix()}"
                f"{self.flang_bin} {flags} {seed_file} >/dev/null"
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

    # flang's TODO() bail-out for a feature it hasn't implemented yet. It is
    # routed through the same `LLVM ERROR: aborting` channel as a real
    # internal error, but it is a deliberate, diagnosed refusal — the
    # message names the construct — not a bug worth reporting. (Same
    # reasoning as the tint adapter's TINT_UNIMPLEMENTED filter.)
    _NOT_YET_IMPLEMENTED_RE = re.compile(r"not yet implemented:\s*([^\n]+)")

    # The memory cap in _env_prefix() is ours: a program that needs more
    # than we allow aborts with LLVM's OOM handler, or — on a sanitized
    # build — with ASan's RSS-limit message. Either way that is our
    # resource cap talking, not a flang defect.
    # Deliberately narrow: ASan's "requested allocation size ... exceeds
    # maximum" is NOT filtered, because in a compiler an absurd allocation
    # request usually means a size was computed wrong — that is a finding,
    # not a cap.
    _OOM_RE = re.compile(
        r"LLVM ERROR: out of memory"
        r"|AddressSanitizer: hard rss limit exhausted")

    def _check_crash(self, stdout, stderr, return_code):
        combined = stderr + stdout
        if self._NOT_YET_IMPLEMENTED_RE.search(combined):
            return False
        if self._OOM_RE.search(combined):
            return False
        return super()._check_crash(stdout, stderr, return_code)

    # flang states its own internal-error identity on one line, and that
    # line — not the stack — is the stable name for the bug: it is what
    # upstream titles the issue with, and it is identical across the
    # different flag sets and call paths that reach the same CHECK.
    _FATAL_RE = re.compile(r"fatal internal error:\s*([^\n]+)")
    # The message names its source file relative to the llvm-project root
    # ("flang/lib/Evaluate/intrinsics.cpp"), which is stable and is how
    # upstream refers to it — but some builds print the absolute build
    # directory in front of it, which is not. Strip only that prefix.
    _FATAL_PATH_RE = re.compile(r"/\S*?(?=flang/(?:lib|include|runtime)/)")

    #: How often one function may appear in a backtrace before the stack is
    #: judged to have run away. An ordinary flang backtrace is ~20-60
    #: frames with no function repeated more than a couple of times; a
    #: runaway recursion repeats its cycle dozens of times.
    _RECURSION_REPEATS = 8

    #: libstdc++'s own assertions ("<header>:479: <signature>: Assertion
    #: 'this->_M_is_engaged()' failed."), which use straight quotes rather
    #: than LLVM's backtick form and would otherwise fall through to the
    #: bare "Aborted" signature.
    _LIBSTDCXX_ASSERT_RE = re.compile(
        r"[^\s:]+:\d+: (?P<fn>[^\n]*?): Assertion '(?P<expr>[^']+)' failed")

    # ── Signature normalisation ────────────────────────────────────
    #
    # Two of flang's messages name the thing that went wrong in a way that
    # differs on every hit, so one bug arrives under a fresh signature each
    # time. Measured on an 18-hour trunk run: 10 bundles for one PowerPC
    # bf16 ISel failure and 10 for one "setting error on" invariant, out of
    # 42 findings.

    #: "No error was reported but setting error on: a size=4 offset=16:
    #: ObjectEntity type: INTEGER(4)" — the symbol's *name*, size and offset
    #: identify the test case, not the bug. The symbol's kind and type do
    #: carry information, so they stay.
    _SYMBOL_NAME_RE = re.compile(
        r"(setting error on:\s*)[A-Za-z_]\w*\s*(?=[(:])")
    _SYMBOL_LAYOUT_RE = re.compile(r"\s*\bsize=\d+\s+offset=\d+")

    #: "Cannot select: t378: f32 = bf16_to_fp t381" — SelectionDAG node
    #: numbers and the fast-math flags carried on the node both vary with
    #: the input; the type and the opcode are the bug.
    _ISEL_NODE_RE = re.compile(r"\bt\d+(?::\d+)?\b")
    _ISEL_FASTMATH_RE = re.compile(
        r"\s+(?:nnan|ninf|nsz|arcp|contract|afn|reassoc|fast|exact|nuw|nsw)\b")

    #: A source location naming the fused seed's temp file, and LLVM IR
    #: metadata ids — both change every run.
    _ISEL_LOC_RE = re.compile(r",?\s*[\w./-]+\.[fF](?:90|95)?:\d+:\d+")
    _ISEL_METADATA_RE = re.compile(r",?\s*![\w.]+\s*!\d+")

    @classmethod
    def _normalise_symbol(cls, detail: str) -> str:
        # Layout first: it sits between the name and the colon the name
        # match looks ahead for, so stripping it last leaves the name.
        detail = cls._SYMBOL_LAYOUT_RE.sub("", detail)
        return cls._SYMBOL_NAME_RE.sub(r"\1", detail)

    @classmethod
    def _normalise_isel(cls, detail: str) -> str:
        if not detail.startswith("Cannot select:"):
            return detail
        detail = cls._ISEL_LOC_RE.sub("", detail)
        detail = cls._ISEL_METADATA_RE.sub("", detail)
        detail = cls._ISEL_NODE_RE.sub("t*", detail)
        return cls._ISEL_FASTMATH_RE.sub("", detail).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        combined = stderr + stdout

        m = self._FATAL_RE.search(combined)
        if m:
            detail = self._FATAL_PATH_RE.sub("", m.group(1).strip())
            return f"fatal internal error: {self._normalise_symbol(detail)}"

        fp = self._stack_overflow_fingerprint(combined)
        if fp:
            return f"Stack overflow: {fp}"

        m = re.search(r"SUMMARY: AddressSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"ASAN: {self._FATAL_PATH_RE.sub('', m.group(1).strip())}"

        m = re.search(r"SUMMARY: UndefinedBehaviorSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"UBSAN: {self._FATAL_PATH_RE.sub('', m.group(1).strip())}"

        m = re.search(r"LLVM ERROR:\s+([^\n]+)", combined)
        if m:
            return f"LLVM ERROR: {self._normalise_isel(m.group(1).strip())}"

        m = re.search(r"Assertion `([^']+)' failed", combined)
        if m:
            return f"Assertion: {m.group(1).strip()}"

        m = self._LIBSTDCXX_ASSERT_RE.search(combined)
        if m:
            # A hardened-libstdc++ assertion (LLVM_ENABLE_ASSERTIONS turns
            # _GLIBCXX_ASSERTIONS on). The expression alone is useless as an
            # identity — every empty-optional dereference in flang reports
            # the same `this->_M_is_engaged()` — so the instantiated type is
            # what names the bug.
            expr = m.group("expr").strip()
            tparam = re.search(r"_Tp = ([^,\]]+)", m.group("fn"))
            return (f"Assertion: {expr} [{tparam.group(1).strip()}]"
                    if tparam else f"Assertion: {expr}")

        fp = self._stack_dump_fingerprint(combined)
        if fp:
            return f"Stack dump: {fp}"

        if "Aborted" in combined:
            return "Aborted"
        if "Segmentation fault" in combined:
            return "Segmentation fault"

        return super().extract_crash_signature(stdout, stderr, return_code)

    _STACK_DUMP_BODY_RE = re.compile(r'Stack dump:\n((?:.*\n?){1,60})')
    _STACK_MSG_LINE_RE = re.compile(r'^\d+\.\t(?:\S+:\d+:\d+:\s*)?(.+)$', re.MULTILINE)
    _STACK_FRAME_RE = re.compile(r'^\s*#\d+\s+0x[0-9a-f]+\s+([A-Za-z_][\w:<>,~ &*]*?)\s*\(', re.MULTILINE)
    _STACK_OFFSET_RE = re.compile(r'\(([^()\s]+?)\+(0x[0-9a-f]+)\)', re.MULTILINE)
    _NOISE_FRAME_RE = re.compile(
        r'^(?:llvm::sys::PrintStackTrace|llvm::sys::RunSignalHandlers|'
        r'.*SignalHandler.*|abort|raise|gsignal|pthread_kill|'
        r'__assert_fail|__cxa_throw)$'
    )

    def _stack_overflow_fingerprint(self, text):
        """Name a runaway recursion by the function it repeats, not by where
        it happened to die.

        Six of the campaign's first 29 flang findings were one bug — a
        self-referential CHARACTER function result length — re-reported
        under six signatures, because each run blew the stack at a
        different frame (`__libc_malloc`, `operator new`, `Scope::FindSymbol`,
        ...). The repeating frame is the same every time."""
        # Scanned over the whole output, not _STACK_DUMP_BODY_RE's 60-line
        # window: what matters here is how often a frame repeats over the
        # entire backtrace.
        counts = {}
        for name in self._STACK_FRAME_RE.findall(text):
            if self._NOISE_FRAME_RE.match(name):
                continue
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        repeats = max(counts.values())
        if repeats < self._RECURSION_REPEATS:
            return None
        # Report the whole cycle, not just one member: mutual recursion
        # between several functions is as common as self-recursion, and
        # which of them the stack died in varies run to run.
        cycle = sorted(name for name, n in counts.items() if n >= repeats / 2)
        return " <-> ".join(cycle)

    def _stack_dump_fingerprint(self, text):
        """Same rationale as ClangDriver's: fingerprint the crash *location*
        (top real stack frames), not the invocation, so two runs of the same
        underlying bug hit with different random flags/temp paths collapse
        to the same signature."""
        m = self._STACK_DUMP_BODY_RE.search(text)
        if not m:
            return None
        body = m.group(1)

        msg_lines = self._STACK_MSG_LINE_RE.findall(body)
        message = msg_lines[1].strip() if len(msg_lines) > 1 else ""

        frames = []
        for fm in self._STACK_FRAME_RE.finditer(body):
            name = fm.group(1).strip()
            if not name or self._NOISE_FRAME_RE.match(name):
                continue
            frames.append(name)
            if len(frames) >= 3:
                break

        if not frames:
            offsets = self._STACK_OFFSET_RE.findall(body)
            if offsets:
                lib, off = offsets[-1]
                frames.append(f"{os.path.basename(lib)}+{off}")

        frame_part = " > ".join(frames)
        if message and frame_part:
            return f"{message} [{frame_part}]"
        return message or frame_part or None
