import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult


class MLIRDriver(BaseDriver):
    """
    MLIR driver: invokes mlir-opt directly.
    FFL runs inside the ffl-mlir container where the binary lives under
    {ffl_root}/projects/mlir/llvm-mlir-install/bin/mlir-opt.
    """

    IR2VEC_KINDS = ["symbolic", "flow-aware"]
    MIR2VEC_KINDS = ["symbolic"]
    PASSES = ["--canonicalize", "--cse", "--inline", "--symbol-dce",
              "--loop-invariant-code-motion", "--sccp"]

    # Always-on flags:
    # --split-input-file   : parse each // ----- section independently (required
    #                        for test files that encode multiple test cases).
    # --allow-unregistered-dialect : tolerate ops from dialects not compiled in.
    BASE_FLAGS = ["--split-input-file", "--allow-unregistered-dialect"]

    # Pattern matching bare `func @` (old MLIR syntax, invalid in LLVM 23+).
    # Negative lookbehind on both word chars and `.` avoids double-patching
    # already-correct `func.func @` occurrences.
    _FUNC_RE = re.compile(r'(?<![\w.])func\s+@')

    def __init__(self, config):
        super().__init__(config)
        self.mlir_opt = os.path.join(
            self.ffl_root, "projects", "mlir", "llvm-mlir-install", "bin", "mlir-opt"
        )

    def _preprocess(self, content: str) -> str:
        """Upgrade old-style `func @name` to `func.func @name` for LLVM 23+."""
        return self._FUNC_RE.sub('func.func @', content)

    def _get_random_flags(self):
        flags = []
        if random.random() > 0.3:
            num_passes = random.randint(1, len(self.PASSES))
            flags.extend(random.sample(self.PASSES, num_passes))
        if random.random() > 0.7:
            flags.append("--verify-roundtrip")
        if random.random() > 0.8:
            flags.append("--verify-each")
        if random.random() > 0.7:
            flags.append(f"--ir2vec-kind={random.choice(self.IR2VEC_KINDS)}")
        if random.random() > 0.8:
            flags.append(f"--mir2vec-kind={random.choice(self.MIR2VEC_KINDS)}")
        return " ".join(flags)

    # Signature = where the compiler died, not which seed made it: the
    # assertion's file:line and function, an LLVM ERROR / UNREACHABLE
    # message with numbers stripped, or the first MLIR frame of the stack
    # dump. Without this, every non-sanitizer crash had signature None and
    # the orchestrator's fallback grouped by little more than the return
    # code.
    _ASSERT_RE = re.compile(r'([\w.+-]+\.(?:cpp|h|cc|inc)):(\d+): (?:[\w:<>~ ,*&()]*?)\b(\w+)\(.*?: Assertion')
    _SUPPORT_HEADERS = {"Casting.h", "SmallVector.h", "DenseMap.h", "ArrayRef.h", "PointerUnion.h",
                        "STLExtras.h", "StringRef.h", "TypeSwitch.h", "PointerIntPair.h",
                        "Optional.h", "APInt.h", "APFloat.h", "ErrorHandling.h", "Value.h",
                        "Operation.h", "OpDefinition.h", "Types.h", "Attributes.h", "Builders.h",
                        "BuiltinTypes.h", "Region.h", "Block.h", "UseDefLists.h"}
    _FRAME_RE = re.compile(r'^#\d+ 0x[0-9a-f]+ (?:in )?((?:\(anonymous namespace\)::|mlir::|llvm::)[^ (]+)', re.M)

    def extract_crash_signature(self, stdout, stderr, return_code):
        base = super().extract_crash_signature(stdout, stderr, return_code)
        if base:
            return base
        text = (stderr or "") + "\n" + (stdout or "")
        m = self._ASSERT_RE.search(text)
        if m:
            # An assertion inside an LLVM support header (`cast<>` on the
            # wrong type, SmallVector bounds) says nothing about *where*
            # the compiler went wrong: use the first MLIR frame instead.
            if m.group(1) in self._SUPPORT_HEADERS:
                fm = self._first_real_frame(text)
                if fm:
                    return f"Assertion: {m.group(1)} in {fm}"
            return f"Assertion: {m.group(1)}:{m.group(2)} {m.group(3)}"
        for key in ("LLVM ERROR:", "UNREACHABLE executed", "report_fatal_error"):
            i = text.find(key)
            if i != -1:
                line = text[i:].splitlines()[0]
                return re.sub(r'\b\d+\b', 'N', line)[:160]
        fm = self._first_real_frame(text)
        if fm:
            return f"Crash: {fm}"
        return None

    _ADT_FRAME_RE = re.compile(r'^llvm::(?:ArrayRef|ilist|detail|function_ref|SmallVector|DenseMap|'
                               r'PointerUnion|cast|dyn_cast|isa|iterator|indexed_accessor|zippy|'
                               r'SmallDenseMap|StringRef|TypeSwitch|Optional|unique_function|'
                               r'CrashRecoveryContext|sys::)')

    def _first_real_frame(self, text):
        """The first stack frame that names compiler code: an `mlir::` or
        anonymous-namespace frame, else the first `llvm::` frame that is not
        an ADT/support template (ArrayRef, ilist_iterator, function_ref...),
        which only say *through* what the crash was reached."""
        frames = self._FRAME_RE.findall(text)
        for f in frames:
            if f.startswith(("mlir::", "(anonymous namespace)::")):
                return f
        for f in frames:
            if not self._ADT_FRAME_RE.match(f):
                return f
        return frames[0] if frames else None

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.mlir")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(self._preprocess(seed.content))
            # hard_rss_limit_mb, not `ulimit -v`: ASan reserves ~16 TB of
            # shadow address space at startup and `ulimit -v` makes that
            # reservation fail, killing every execution before it reads the
            # seed. Capping resident memory bounds the thing worth bounding
            # and leaves the reservation alone. Without any cap a runaway
            # pass under ASan — which needs roughly 3x the memory — can take
            # the whole machine down.
            rss_mb = getattr(self, "mem_limit_kb", 0) // 1024 or 4096
            asan_opts = ("abort_on_error=1:detect_leaks=0:symbolize=1"
                         f":hard_rss_limit_mb={max(64, rss_mb)}")
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            base = " ".join(self.BASE_FLAGS)
            # --dry-run/--pre-analysis asks whether mlir-opt accepts the
            # seed at all: parse + verify, no random pass pipeline, so the
            # recorded rc is a property of the seed and not of the draw.
            flags = "" if getattr(self, "dryrun_mode", False) else self._get_random_flags()
            cmd = (
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"{self.mlir_opt} {base} {flags} {seed_file}"
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
