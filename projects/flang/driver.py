import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class FlangDriver(BaseDriver):
    """
    Flang driver: invokes the system flang frontend directly.
    FFL runs inside the ffl-flang container where 'flang' is already in
    PATH (see projects/flang/Dockerfile — no from-source LLVM build
    required). Flang shares clang's LLVM-based driver and crash-reporting
    infrastructure, so the crash signatures look the same (Stack dump,
    LLVM ERROR, Assertion, ...).
    """

    FLANG_BIN = "flang"

    # flang only accepts -O0..-O3 (no -Os/-Oz, unlike clang).
    OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3"]

    # Compilation "depth" to exercise: syntax-only is cheap and hits the
    # parser/semantics most often; the others push further into
    # lowering/CodeGen.
    MODES = ["-fsyntax-only", "-emit-llvm -S -o /dev/null",
             "-S -o /dev/null", "-c -o /dev/null"]
    MODE_WEIGHTS = [45, 20, 20, 15]

    # Only f2018 is currently accepted by flang's -std=.
    STD_VALUES = ["f2018"]

    MISC_FLAGS = [
        "-ffast-math", "-fdefault-real-8", "-fdefault-integer-8",
        "-fdefault-double-8", "-fbackslash", "-fimplicit-none",
        "-falternative-parameter-statement", "-finit-global-zero",
        "-g", "-fno-automatic",
    ]

    _FIXED_FORM_EXTS = (".f", ".F")

    def _lang_flags(self, ext):
        return ["-ffixed-form"] if ext in self._FIXED_FORM_EXTS else ["-ffree-form"]

    def _get_random_flags(self, ext):
        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0]]
        flags.append(random.choice(self.OPT_LEVELS))
        flags.extend(self._lang_flags(ext))
        if random.random() > 0.5:
            flags.append(f"-std={random.choice(self.STD_VALUES)}")
        flags.extend(random.sample(self.MISC_FLAGS, random.randint(0, 3)))
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

            flags = self._get_random_flags(ext)
            asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1"
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            cmd = (
                f"ulimit -v 3145728; "
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"{self.FLANG_BIN} {flags} {seed_file}"
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

    def extract_crash_signature(self, stdout, stderr, return_code):
        combined = stderr + stdout

        m = re.search(r"SUMMARY: AddressSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"ASAN: {m.group(1).strip()}"

        m = re.search(r"SUMMARY: UndefinedBehaviorSanitizer:\s+([^\n]+)", combined)
        if m:
            return f"UBSAN: {m.group(1).strip()}"

        m = re.search(r"LLVM ERROR:\s+([^\n]+)", combined)
        if m:
            return f"LLVM ERROR: {m.group(1).strip()}"

        m = re.search(r"Assertion `([^']+)' failed", combined)
        if m:
            return f"Assertion: {m.group(1).strip()}"

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
