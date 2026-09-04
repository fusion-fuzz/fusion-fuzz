import os
import subprocess
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

    # c89 measured markedly worse than the rest on this corpus (58.8% vs
    # ~47-53% error) because most seeds are C99-or-later; keep it in the mix
    # for coverage of the old parser paths, just not at equal weight.
    STD_C = ["c99", "c11", "c17", "c23", "gnu99", "gnu11", "gnu17", "c89"]
    STD_C_WEIGHTS = [15, 15, 15, 15, 15, 15, 15, 5]
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

    # Objective-C on Linux defaults to the GCC legacy runtime, which does not
    # support ARC: passing -fobjc-arc alone makes clang bail out with
    # "error: -fobjc-arc is not supported on platforms using the legacy
    # runtime" before parsing a single line of the seed. Any of these
    # runtimes enables the non-fragile ABI that ARC requires.
    OBJC_ARC_RUNTIMES = ["gnustep-2.0", "ios-7", "macosx-10.14"]

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

    # Since clang 16, implicit function declarations and implicit int are
    # errors by default in C mode. A large share of the C seed corpus is
    # pre-C99 style (`foo(x) { ... }`, calls to undeclared functions), so
    # without these two the seed is rejected on K&R syntax before any of
    # the frontend work we are fuzzing happens. They only downgrade those
    # two diagnostics back to warnings; nothing else changes. Not passed
    # for C++, which has no such constructs.
    C_LEGACY_FLAGS = ["-Wno-implicit-function-declaration", "-Wno-implicit-int"]

    # Triple the seed's own RUN line asks for. Most of clang/test is written
    # against a specific backend, and compiling it for the host instead
    # leaves every target builtin and type undeclared. Honouring the triple
    # took standalone validity on that part of the corpus from 46.8% to
    # 50.8% and is the only way these seeds reach the AArch64/RISCV/PowerPC
    # frontends at all.
    _TRIPLE_RE = re.compile(
        r'(?m)^\s*(?://|/\*|#)\s*RUN:.*?(?:-triple[= ]\s*|--target=)([\w.]+(?:-[\w.]+)*)')
    # Architectures this build actually registers (setup.py:
    # LLVM_TARGETS_TO_BUILD). An unregistered arch would just error out.
    _SUPPORTED_ARCH = (
        "x86_64", "i386", "i686", "i586", "x86",
        "aarch64", "aarch64_be", "aarch64_32", "arm64", "arm64_32",
        "arm", "armeb", "armv7", "armv7a", "armv6", "armv8", "thumb", "thumbv7", "thumbv8",
        "thumbv8.1m", "riscv32", "riscv64",
        "powerpc", "powerpc64", "powerpc64le", "ppc32", "ppc64", "ppc64le", "ppc32le",
    )
    # Intrinsic headers that need their extension switched on explicitly.
    _FEATURE_BY_HEADER = [
        (re.compile(r'#\s*include\s*[<"](?:riscv_vector|sifive_vector|andes_vector)\.h'),
         ("riscv", "-march=rv64gcv")),
        (re.compile(r'#\s*include\s*[<"]arm_sme\.h'), ("aarch64", "-march=armv9-a+sve2+sme")),
        (re.compile(r'#\s*include\s*[<"]arm_sve\.h'), ("aarch64", "-march=armv8-a+sve")),
        (re.compile(r'#\s*include\s*[<"]arm_mve\.h'), ("thumb", "-march=armv8.1-m.main+mve")),
    ]

    def _target_flags(self, content):
        """(-target/-march flags, is_cross) implied by the seed's RUN line."""
        triples = [t for t in self._TRIPLE_RE.findall(content or "")
                   if t.split("-")[0].lower() in self._SUPPORTED_ARCH]
        if not triples:
            return "", False
        triple = random.choice(triples)
        arch = triple.split("-")[0].lower()
        # -ffreestanding: a cross target still resolves <stdint.h> to the
        # host's glibc headers, which then fail on bits/libc-header-start.h.
        flags = [f"-target {triple}", "-ffreestanding"]
        for rx, (arch_prefix, march) in self._FEATURE_BY_HEADER:
            if arch.startswith(arch_prefix) and rx.search(content or ""):
                flags.append(march)
                break
        return " ".join(flags), not arch.startswith(("x86", "i3", "i5", "i6"))

    def _get_random_flags(self, ext, content=""):
        binname, stds = self._lang_for(ext)
        target_flags, cross = self._target_flags(content)
        flags = [random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0]]
        flags.append(random.choice(self.OPT_LEVELS))
        if stds and random.random() > 0.3:
            weights = self.STD_C_WEIGHTS if stds is self.STD_C else None
            std = random.choices(stds, weights=weights, k=1)[0] if weights else random.choice(stds)
            flags.append(f"-std={std}")
        if ext not in self._CXX_EXTS:
            flags.extend(self.C_LEGACY_FLAGS)
        # -fblocks for every language: blocks (^{ ... }) are a clang extension
        # used across the C, C++ and Obj-C seeds alike and are off by default
        # on Linux, so without it those seeds die on "blocks support disabled"
        # before reaching the frontend paths being fuzzed.
        flags.append("-fblocks")
        if ext in (".m", ".mm"):  # Objective-C and Objective-C++
            # Always select a modern runtime, not just for ARC: the Linux
            # default (GCC legacy, fragile ABI) rejects weak references and
            # property synthesis without a matching ivar — both pervasive in
            # the Obj-C corpus — and refuses -fobjc-arc outright.
            flags.append(f"-fobjc-runtime={random.choice(self.OBJC_ARC_RUNTIMES)}")
            flags.append("-fobjc-arc" if random.random() > 0.5 else "-fno-objc-arc")
        misc = random.sample(self.MISC_FLAGS, random.randint(0, 3))
        if cross:
            # No sanitizer runtime is built for the cross targets, and the
            # driver rejects the flag outright rather than just warning.
            misc = [f for f in misc if not f.startswith("-fsanitize")]
        flags.extend(misc)
        if target_flags:
            flags.append(target_flags)
        return binname, " ".join(flags)

    def _is_sanitized(self):
        """Whether the clang under test was built with sanitizers.

        Cached: it is one nm/strings pass, and the answer decides how every
        subsequent execution is memory-limited. Probing the binary rather
        than reading a config keeps the two in step when someone rebuilds
        with FFL_CLANG_SANITIZERS=none."""
        cached = getattr(self, "_sanitized_cache", None)
        if cached is not None:
            return cached
        result = False
        try:
            out = subprocess.run(["nm", "-D", self.clang_bin],
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
            ext = seed.metadata.get("extension") or ".c"
            seed_file = os.path.join(workdir, f"{seed.id}{ext}")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)

            binname, flags = self._get_random_flags(ext, seed.content)
            # Memory limiting, and why it is not `ulimit -v` when the
            # compiler is sanitized.
            #
            # ASan reserves ~16 TB of shadow address space at startup. Under
            # `ulimit -v` that reservation fails and the process dies before
            # it reads the seed:
            #     ReserveShadowMemoryRange failed while trying to map
            #     0xdfff0001000 bytes. Perhaps you're using ulimit -v
            # That is every execution, not an occasional one, and the abort
            # matches the "Aborted" crash pattern — so an assertions-only
            # build silently turns into a 100% false-alarm rate the moment
            # LLVM_USE_SANITIZER is switched on.
            #
            # ASan's own hard_rss_limit_mb caps *resident* memory instead,
            # which is the thing actually worth bounding, and leaves the
            # address-space reservation alone. `ulimit -v` stays for the
            # uninstrumented build, where it costs nothing.
            rss_mb = max(64, self.mem_limit_kb // 1024)
            asan_opts = ("abort_on_error=1:detect_leaks=0:symbolize=1"
                         f":hard_rss_limit_mb={rss_mb}")
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            limit = ("ulimit -c 0; " if self._is_sanitized()
                     else f"ulimit -v {self.mem_limit_kb}; ulimit -c 0; ")
            cmd = (
                f"{limit}"
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
