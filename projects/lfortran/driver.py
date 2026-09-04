import os
import subprocess
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class LfortranDriver(BaseDriver):
    """
    LFortran driver: invokes the `lfortran` binary built by setup.py
    (installed to /opt/lfortran/bin, already on PATH — see Dockerfile).

    LFortran's crash reporting is its own (LCompilers) machinery, not
    LLVM's/clang's, so the modes and signature extraction below are
    LFortran-specific: see src/bin/lfortran.cpp's main()/main_app() and
    src/libasr/stacktrace.cpp in the upstream source for the exact
    wording matched here.
    """

    LFORTRAN_BIN = "lfortran"

    # Different compilation depths to exercise. --show-asr is weighted
    # heaviest: it forces full semantic analysis (parser + ASR build +
    # passes) without needing the LLVM backend to succeed, mirroring why
    # flang's driver weights -fsyntax-only heaviest — most frontend bugs
    # live there. The rest push further into codegen (LLVM IR/ASM/object)
    # or the alternate C/C++ backends.
    MODES = [
        "--show-ast", "--show-asr", "--show-llvm", "--show-c",
        "--show-cpp", "--show-asm", "-S -o /dev/null", "-c -o /dev/null",
    ]
    MODE_WEIGHTS = [15, 30, 15, 10, 5, 5, 10, 10]

    STD_VALUES = ["lf", "f23", "legacy"]

    MISC_FLAGS = [
        "--implicit-typing", "--implicit-interface",
        "--implicit-argument-casting", "--logical-casting",
        "--use-loop-variable-after-loop", "--legacy-array-sections",
        "--cpp", "-g",
    ]

    _FIXED_FORM_EXTS = (".f", ".F")

    def _lang_flags(self, ext):
        return ["--fixed-form"] if ext in self._FIXED_FORM_EXTS else []

    def _get_random_flags(self, ext):
        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0]]
        flags.extend(self._lang_flags(ext))
        if random.random() > 0.5:
            flags.append(f"--std={random.choice(self.STD_VALUES)}")
        flags.extend(random.sample(self.MISC_FLAGS, random.randint(0, 3)))
        return " ".join(flags)

    def _resolved_bin(self):
        """LFORTRAN_BIN as an absolute path.

        LFORTRAN_BIN is a bare name resolved through PATH at exec time. That
        is fine for running it and wrong for inspecting it: `nm -D lfortran`
        looks for a file called "lfortran" in the working directory and
        fails, rather than searching PATH."""
        cached = getattr(self, "_bin_path_cache", None)
        if cached is None:
            cached = shutil.which(self.LFORTRAN_BIN) or self.LFORTRAN_BIN
            self._bin_path_cache = cached
        return cached

    def _is_sanitized(self):
        """Whether the lfortran under test was built with sanitizers.

        This decides whether the execution gets `ulimit -v`, and getting it
        wrong is not a small error: a sanitized lfortran under `ulimit -v`
        fails its shadow-memory reservation and aborts on every seed, which
        the "Abort: Signal SIGABRT" crash pattern then reports as a finding.
        Measured on 250 seeds with the probe misresolving the binary: 250
        aborts, 250 false alarms."""
        cached = getattr(self, "_sanitized_cache", None)
        if cached is not None:
            return cached
        result = False
        try:
            out = subprocess.run(["nm", "-D", self._resolved_bin()],
                                 capture_output=True, text=True, timeout=120)
            result = "__asan_init" in (out.stdout or "")
        except (OSError, subprocess.SubprocessError):
            result = False
        self._sanitized_cache = result
        return result

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

            flags = self._get_random_flags(ext)
            # Memory limiting, and the sanitizer options that go with it.
            #
            # `ulimit -v` cannot be used against a sanitized lfortran: ASan
            # reserves ~16 TB of shadow address space at startup, the
            # reservation fails under a virtual-memory cap, and the process
            # dies before it reads the seed — every execution, not some.
            # hard_rss_limit_mb caps resident memory instead, which is the
            # thing worth bounding.
            #
            # detect_leaks=0 because a compiler that frees nothing on the
            # way out is normal; leaving it on reports a leak for every
            # single seed. UBSan keeps halt_on_error=0 so one finding does
            # not mask the rest of the run — the oracle matches on the
            # "SUMMARY: UndefinedBehaviorSanitizer" text (see config.yaml),
            # not on the exit status.
            if self._is_sanitized():
                rss_mb = 3072
                asan = (f"ASAN_OPTIONS='detect_leaks=0:symbolize=1"
                        f":hard_rss_limit_mb={rss_mb}' ")
                ubsan = "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=0' "
                limit = ""
            else:
                asan = ubsan = ""
                limit = "ulimit -v 3145728; "
            cmd = (f"{limit}{asan}{ubsan}"
                   f"{self.LFORTRAN_BIN} {flags} {seed_file}")
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

    _TRACEBACK_RE = re.compile(r'Traceback \(most recent call last\):\n((?:.*\n?){1,80})')
    _FRAME_IN_RE = re.compile(r', in ([A-Za-z_][\w:<>,~ &*0-9]*)')
    _NOISE_FRAME_RE = re.compile(
        r'^(?:main|main_app|loc_segfault_callback_print_stack|'
        r'loc_abort_callback_print_stack)$'
    )

    def _stacktrace_fingerprint(self, text):
        """Fingerprint the crash *location* (top real ASR/codegen frames
        from LFortran's own Traceback block), not the invocation, so two
        runs of the same underlying bug with different random flags/temp
        paths collapse to the same signature. Mirrors FlangDriver's
        _stack_dump_fingerprint, adapted to LCompilers' stacktrace2str
        format ('File "...", line N, in <function>')."""
        m = self._TRACEBACK_RE.search(text)
        if not m:
            return None
        body = m.group(1)
        frames = []
        for fm in self._FRAME_IN_RE.finditer(body):
            name = fm.group(1).strip()
            if not name or self._NOISE_FRAME_RE.match(name):
                continue
            frames.append(name)
            if len(frames) >= 3:
                break
        return " > ".join(frames) if frames else None

    # lfortran reports an unimplemented language feature the same way it
    # reports a genuine internal error: "Internal Compiler Error:
    # LCompilersException: visit_DoLoop() not implemented". The config's
    # "Internal Compiler Error" crash pattern therefore fires on every seed
    # that uses a construct lfortran has not got to yet, which is not a bug
    # in lfortran and not something to report. Measured on 250 integration
    # tests: 10 of 41 reported crashes were this.
    _NOT_IMPLEMENTED_RE = re.compile(r"not implemented", re.IGNORECASE)

    def _check_crash(self, stdout, stderr, return_code):
        combined = (stderr or "") + (stdout or "")
        # A sanitizer finding is a real finding even if the same run also
        # hit an unimplemented feature, so check for those first.
        if ("SUMMARY: AddressSanitizer" in combined
                or "SUMMARY: UndefinedBehaviorSanitizer" in combined):
            return True
        if ("Internal Compiler Error" in combined
                and self._NOT_IMPLEMENTED_RE.search(combined)):
            return False
        return super()._check_crash(stdout, stderr, return_code)

    def extract_crash_signature(self, stdout, stderr, return_code):
        combined = stderr + stdout

        m = re.search(r"SUMMARY: AddressSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"ASAN: {m.group(1).strip()}"

        m = re.search(r"SUMMARY: UndefinedBehaviorSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"UBSAN: {m.group(1).strip()}"

        fp = self._stacktrace_fingerprint(combined)

        if "Internal Compiler Error" in combined:
            tail = combined[combined.find("Internal Compiler Error"):]
            m = re.search(r'\n(\w[\w:]*): (.+)', tail)
            detail = f": {m.group(1)}: {m.group(2).strip()}" if m else ""
            label = f"ICE{detail}"
            return f"{label} [{fp}]" if fp else label

        if "Segfault: Signal SIGSEGV" in combined:
            return f"SIGSEGV [{fp}]" if fp else "SIGSEGV"

        if "Abort: Signal SIGABRT" in combined:
            return f"SIGABRT [{fp}]" if fp else "SIGABRT"

        m = re.search(r"terminate called after throwing an instance of '([^']+)'", combined)
        if m:
            return f"terminate: {m.group(1)}"

        m = re.search(r"runtime_error:\s*([^\n]+)", combined)
        if m:
            return f"runtime_error: {m.group(1).strip()}"

        m = re.search(r"std::exception:\s*([^\n]+)", combined)
        if m:
            return f"std::exception: {m.group(1).strip()}"

        if "Unknown Exception" in combined:
            return "Unknown Exception"

        if "Aborted" in combined:
            return "Aborted"
        if "Segmentation fault" in combined:
            return "Segmentation fault"

        return super().extract_crash_signature(stdout, stderr, return_code)
