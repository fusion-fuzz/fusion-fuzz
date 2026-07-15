import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class ClangDriver(BaseDriver):
    """
    Clang driver: invokes clang/clang++ built from llvm-project source.
    FFL runs inside the fuzz-clang container; projects/clang/setup.py clones
    llvm-project's main branch and builds it, installing to
    {ffl_root}/projects/clang/llvm-clang-install/bin/{clang,clang++}.
    """

    STD_C = ["c89", "c99", "c11", "c17", "c23", "gnu99", "gnu11", "gnu17"]
    STD_CXX = ["c++03", "c++11", "c++14", "c++17", "c++20", "c++23", "gnu++17", "gnu++20"]
    OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"]

    # Compilation "depth" to exercise: syntax-only is cheap and hits the
    # parser/sema most often; the others push further into CodeGen/opt.
    MODES = ["-fsyntax-only", "-emit-llvm -S -o /dev/null",
             "-S -o /dev/null", "-c -o /dev/null"]
    MODE_WEIGHTS = [45, 20, 20, 15]

    MISC_FLAGS = [
        "-Wall", "-Wextra", "-ffast-math", "-fno-strict-aliasing",
        "-fsanitize=address", "-fsanitize=undefined", "-g",
        "-funroll-loops", "-fno-inline", "-ffp-contract=fast",
        "-fstrict-enums", "-fno-elide-constructors",
    ]

    _CXX_EXTS = (".cpp", ".cc", ".cxx", ".mm")

    # Fuzzer-generated inputs routinely hit clang's classic memory-blowup
    # bug classes (exponential template instantiation, runaway constexpr
    # evaluation, absurd array sizes...). Without a cap a single seed can
    # balloon to many GB and trigger the *system* OOM killer, which under
    # cgroup v2's default oom_group behavior can take down every process in
    # the container at once — including the orchestrator and any watchdog
    # wrapping it, so nothing survives to restart. Capping each compiler
    # invocation's address space makes that failure local and clean: malloc
    # fails, LLVM's allocator calls report_bad_alloc_error() and aborts with
    # "LLVM ERROR: out of memory" — one of our existing crash_patterns —
    # instead of taking the whole run down.
    DEFAULT_MEM_LIMIT_MB = 3072

    def __init__(self, config):
        super().__init__(config)
        mem_limit_mb = config.get('execution', {}).get('mem_limit_mb', self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_kb = int(mem_limit_mb) * 1024
        install_bin = os.path.join(self.ffl_root, "projects", "clang", "llvm-clang-install", "bin")
        self.clang_bin = os.path.join(install_bin, "clang")
        self.clangxx_bin = os.path.join(install_bin, "clang++")

    def _lang_for(self, ext):
        if ext in self._CXX_EXTS:
            return self.clangxx_bin, self.STD_CXX
        if ext == ".m":
            return self.clang_bin, None  # Objective-C
        return self.clang_bin, self.STD_C

    def _get_random_flags(self, ext):
        binname, stds = self._lang_for(ext)
        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0]]
        flags.append(random.choice(self.OPT_LEVELS))
        if stds and random.random() > 0.3:
            flags.append(f"-std={random.choice(stds)}")
        if ext == ".m":
            flags.append("-fobjc-arc" if random.random() > 0.5 else "-fno-objc-arc")
        flags.extend(random.sample(self.MISC_FLAGS, random.randint(0, 3)))
        return binname, " ".join(flags)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            ext = seed.metadata.get("extension") or ".c"
            seed_file = os.path.join(workdir, f"{seed.id}{ext}")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)

            binname, flags = self._get_random_flags(ext)
            asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1"
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            cmd = (
                f"ulimit -v {self.mem_limit_kb}; "
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"{binname} {flags} {seed_file}"
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
    # LLVM's PrettyStackTrace/signal-handler dump prints frames as
    # "<n>  <binary>  0x<addr> [<symbol>(<args>) + <offset>]" — no leading
    # '#' and no parens around the binary+address, unlike the gdb-style
    # format this regex was originally written for. Matching '#\d+' here
    # silently matched zero frames on every real crash, degrading every
    # Stack dump signature down to just the generic crash-site message.
    _STACK_FRAME_RE = re.compile(r'^\s*\d+\s+\S+\s+0x[0-9a-f]+\s+([A-Za-z_][\w:<>,~ &*]*?)\s*\(', re.MULTILINE)
    _STACK_OFFSET_RE = re.compile(r'^\s*\d+\s+(\S+)\s+(0x[0-9a-f]+)\s*$', re.MULTILINE)
    _NOISE_FRAME_RE = re.compile(
        r'^(?:llvm::sys::PrintStackTrace|llvm::sys::RunSignalHandlers|'
        r'llvm::sys::CleanupOnSignal|'
        r'.*SignalHandler.*|abort|raise|gsignal|pthread_kill|'
        r'__assert_fail|__cxa_throw)$'
    )

    def _stack_dump_fingerprint(self, text):
        """Build a signature from the *crash location*, not the invocation:
        the diagnostic line right after 'Program arguments' (with the
        file:line:col prefix stripped, since that's unique per temp file),
        plus the first few real stack frames (skipping signal-handler noise
        and address-only frames). Two runs of the *same* underlying bug hit
        with different random flags/temp paths must still collapse to the
        same signature, otherwise every fuzzing run reports the same crash
        as a "new" bug."""
        m = self._STACK_DUMP_BODY_RE.search(text)
        if not m:
            return None
        body = m.group(1)

        msg_lines = self._STACK_MSG_LINE_RE.findall(body)
        # msg_lines[0] is "Program arguments: ..." — the crash-site message
        # (if any) is the next numbered line.
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
            # Every frame in the captured window was unsymbolized (stripped
            # library / inlined deep in shared lib) — fall back to the last
            # "lib+offset" so distinct crash sites still disambiguate
            # instead of all collapsing into one generic message-only bucket.
            offsets = self._STACK_OFFSET_RE.findall(body)
            if offsets:
                lib, off = offsets[-1]
                frames.append(f"{os.path.basename(lib)}+{off}")

        frame_part = " > ".join(frames)
        if message and frame_part:
            return f"{message} [{frame_part}]"
        return message or frame_part or None
