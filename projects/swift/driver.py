import random
import re
import shutil
import time
import os
from core.driver import BaseDriver, ExecutionResult


class SwiftDriver(BaseDriver):
    """
    Swift driver: invokes swift -frontend directly.
    FFL runs inside the ffl-swift container where 'swift' is in PATH.
    """

    MODES = ["-typecheck", "-emit-silgen", "-emit-sil", "-emit-ir", "-c"]
    MODE_WEIGHTS = [50, 10, 10, 20, 10]
    OPT_LEVELS = ["-Onone", "-O", "-Osize", "-wmo"]
    BASE_FLAGS = ["-sil-verify-all"]
    EXPERIMENTAL_FEATURES = [
        "-enable-experimental-feature VariadicGenerics",
        "-enable-experimental-feature Macros",
        "-enable-experimental-feature MoveOnly",
        "-enable-experimental-feature NonescapableTypes",
        "-enable-experimental-feature ThenStatements",
    ]
    MISC_FLAGS = [
        "-enable-library-evolution",
        "-strict-concurrency=complete",
        "-disable-availability-checking",
        "-enforce-exclusivity=checked",
        # `-debug-info-format` is only accepted together with `-g`:
        # alone, the frontend exits with "option '-debug-info-format=dwarf'
        # is missing a required argument (-g)" before reading the program.
        # Measured at 10-14% of every strategy's invalid children.
        "-g -debug-info-format=dwarf",
    ]

    def _get_random_flags(self):
        flags = []
        flags.append(random.choices(self.MODES, weights=self.MODE_WEIGHTS, k=1)[0])
        flags.append(random.choice(self.OPT_LEVELS))
        flags.extend(self.BASE_FLAGS)
        spice = self.MISC_FLAGS + self.EXPERIMENTAL_FEATURES
        flags.extend(random.sample(spice, random.randint(0, 2)))
        return " ".join(flags)

    # A seed's own text can contain "Please submit a bug report" (FileCheck
    # `CHECK-NOT:` lines), and a normal diagnostic echoes the source line
    # in its gutter (`74 | // CHECK-NOT: Please submit ...`) — one false
    # bundle per run before this. The marker counts only when the crash
    # apparatus is there too.
    # A diagnostic echoes the offending source line (`59 | // Assertion
    # failed: ...`); a seed that quotes an old crash in a comment then
    # matched the crash patterns with rc 0. Every echoed line is dropped
    # before matching, and a text match needs a non-zero exit.
    _GUTTER_ECHO_RE = re.compile(r'^\s*\d+\s*\|.*$', re.M)

    def _check_crash(self, stdout, stderr, return_code):
        if not super()._check_crash(stdout, stderr, return_code):
            return False
        text = (stderr or "") + "\n" + (stdout or "")
        if return_code >= 128 or return_code in (134, 139):
            return True
        stripped = self._GUTTER_ECHO_RE.sub('', text)
        if return_code == 0:
            return False
        return ("Stack dump:" in stripped or "Assertion failed:" in stripped
                or "Please submit a bug report" in stripped)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.swift")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            flags = "-typecheck -Onone" if getattr(self, "dryrun_mode", False) else self._get_random_flags()
            asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1:detect_stack_use_after_return=1"
            ubsan_opts = "print_stacktrace=1:halt_on_error=1"
            cmd = (
                f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "
                f"swift -frontend {flags} {seed_file}"
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

    # The numbered context lines swift-frontend prints before the stack:
    #   3. While evaluating request IRGenRequest(IR Generation for module x)
    #   5. While running pass #63 SILFunctionTransform "PackMetadata..." on SILFunction "@$s..."
    #   6. While verifying SIL function "@$s...".
    _CONTEXT_LINE_RE = re.compile(r'^\d+\.\s+(While .*)$', re.M)

    @classmethod
    def _normalize_context(cls, line: str) -> str:
        line = re.sub(r'"@\$s[^"]*"', '"<fn>"', line)          # mangled symbols
        line = re.sub(r'\bfor module \w+', 'for module <m>', line)
        line = re.sub(r'\bon SIL for \w+', 'on SIL for <m>', line)
        line = re.sub(r'pass #\d+', 'pass', line)
        line = re.sub(r'\([^()]*\.swift:\d+:\d+[^()]*\)', '(<loc>)', line)
        # Bare locations, ranges and source excerpts, fusion name suffixes
        # (`Q_d15419a`), and quoted identifiers: none of them says *where
        # the compiler crashed*, and each one split one crash site into
        # many signatures (31 bundles for ~12 sites in one 10-minute run).
        line = re.sub(r'\S*\.swift:\d+:\d+', '<loc>', line)
        line = re.sub(r'\[?<loc> - line:\d+:\d+\]', '<range>', line)
        line = re.sub(r'\b\w+\.\(file\)', '(file)', line)          # module prefix
        line = re.sub(r'RangeText="[^"]*"?', 'RangeText', line)
        line = re.sub(r'_[0-9a-f]{7,8}\b', '', line)
        # A request's argument names the declaration it was evaluating
        # (`CompareDeclSpecializationRequest(<hex> AbstractFunctionDecl
        # name=init() : ...)`), which differs per pair: nine bundles for
        # one site. Keep the request name only.
        line = re.sub(r'(While evaluating request \w+)\([^\n]*', r'\1(<args>)', line)
        # `While canonicalizing ... SIL node %3 = differentiable_function
        # [...] %2 : $@callee_guaranteed ...`: the node's operands and
        # type are per pair; the canonicalization is the site.
        line = re.sub(r'(SIL node\s+)%[^\n]*', r'\1<node>', line)
        # `While type-checking extension of MyClass at <loc>`: the type is
        # the pair's, the context kind is the site.
        line = re.sub(r'\b(extension|declaration|conformance|body) of \S+', r'\1 of <T>', line)
        line = re.sub(r"'[^']*'", "'<id>'", line)
        line = re.sub(r'(τ_\d+_\d+)\s*:\s*[\w.]+', r'\1 : <id>', line)   # generic-signature constraints
        line = re.sub(r'\b0x[0-9a-fA-F]+\b', '<hex>', line)
        return re.sub(r'\s+', ' ', line).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        sig = super().extract_crash_signature(stdout, stderr, return_code)
        if sig:
            return sig
        for text in (stderr, stdout):
            m = re.search(r"(Assertion failed: .*)", self._GUTTER_ECHO_RE.sub('', text or ""))
            if m:
                return m.group(1).strip()
        # Previously only the *outermost* request ("While evaluating
        # request IRGenRequest") was used, which put every IRGen crash in
        # one bucket. The pass and verification context underneath is
        # what tells two crashes apart, so the whole chain is the key
        # (with module names, symbols and pass numbers normalised).
        for text in (stderr, stdout):
            ctx = [self._normalize_context(l) for l in self._CONTEXT_LINE_RE.findall(text)]
            if ctx:
                # Deepest two: the pass and what it was doing. The outer
                # request lines are the same for every crash of a kind.
                return " | ".join(ctx[-2:])[:200]
        return None
