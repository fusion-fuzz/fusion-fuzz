import os
import re
import random
import shutil
import time
from core.driver import BaseDriver, ExecutionResult

# Matches any Cangjie top-level declaration
_TOPLEVEL_DECL = re.compile(
    r'^\s*(?:(?:public|private|protected|open|abstract|override)\s+)*'
    r'(?:func|class|struct|enum|interface|extend)\s+[A-Za-z_]'
    r'|^\s*main\s*\(\s*\)',
    re.MULTILINE,
)

# Python/shell inline or full-line # comment (not a raw-string prefix like #")
_HASH_COMMENT = re.compile(r'^([ \t]*)#(?!["#])(.*)', re.MULTILINE)

# Em-dash and en-dash — invalid tokens in Cangjie source
_EM_DASH = re.compile(r'[—–]')


class CangjieDriver(BaseDriver):
    """
    Cangjie compiler driver: invokes `cjc` to compile .cj seed files.
    FFL runs inside the ffl-cangjie container where the SDK is installed at /opt/cangjie.

    We fuzz the compiler front-end (parser, type-checker, IR lowering) by varying
    optimization levels and debug flags. Runtime execution of the compiled binary is
    skipped to keep iteration latency low; crashes in the compiler itself are the goal.
    """

    CJC_BIN = "/opt/cangjie/bin/cjc"

    # Vary optimization level to exercise different compiler paths
    OPT_LEVELS = ["-O0", "-O1", "-O2"]

    # Additional flags to exercise less-common code paths
    EXTRA_FLAG_GROUPS = [
        [],
        ["-g"],
        ["--output-type", "staticlib"],
        ["-g", "--output-type", "staticlib"],
    ]

    def __init__(self, config):
        super().__init__(config)
        self.memory_limit_mb = int(
            config.get("execution", {}).get("memory_limit_mb", 512) or 0
        )

    def _sanitize(self, content: str) -> str:
        """Best-effort cleanup of fused content to reduce trivial parse errors."""
        # 1. Convert Python/shell # comments → Cangjie // comments
        content = _HASH_COMMENT.sub(lambda m: f"{m.group(1)}//{m.group(2)}", content)
        # 2. Remove em-dash / en-dash (not valid Cangjie tokens)
        content = _EM_DASH.sub('-', content)
        # 3. If there are zero top-level declarations, wrap the whole body in main()
        #    so the compiler at least has a valid entry point to attempt parsing.
        if not _TOPLEVEL_DECL.search(content):
            indented = '\n'.join('    ' + ln for ln in content.splitlines())
            content = f'main() {{\n{indented}\n}}'
        return content

    def _get_random_flags(self) -> str:
        parts = [random.choice(self.OPT_LEVELS)]
        parts += random.choice(self.EXTRA_FLAG_GROUPS)
        return " ".join(parts)

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.cj")
            out_file = os.path.join(workdir, f"{seed.id}.out")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(self._sanitize(seed.content))

            # LLM-translated seeds are already well-formed; keep flags minimal
            if seed.metadata.get("type") == "llm_translated":
                flags = "-O0"
            else:
                flags = self._get_random_flags()

            # Build base command: compile only, discard output binary
            cmd = f"{self.CJC_BIN} {flags} {seed_file} -o {out_file}"

            # Apply RSS cap via ASAN options if limit is set
            if self.memory_limit_mb:
                asan_opts = f"hard_rss_limit_mb={self.memory_limit_mb}"
                cmd = f"ASAN_OPTIONS='{asan_opts}' {cmd}"

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

        # ASAN / UBSAN summary
        m = re.search(r"SUMMARY:\s+\w+Sanitizer:\s+(.*)", combined)
        if m:
            return m.group(1).strip()

        # Compiler ICE
        m = re.search(r"internal compiler error[:\s]+(.*)", combined, re.IGNORECASE)
        if m:
            return f"ICE: {m.group(1).strip()[:120]}"

        # Rust-style panic (in case compiler is Rust-based)
        m = re.search(r"panicked at '?(.*?)'?(?:,\s*[\w/]+\.rs:\d+)?$", combined, re.MULTILINE)
        if m:
            return f"panic: {m.group(1).strip()[:120]}"

        # LLVM backend error
        m = re.search(r"LLVM ERROR:\s+(.*)", combined)
        if m:
            return f"LLVM: {m.group(1).strip()[:120]}"

        if "Segmentation fault" in combined:
            return "Segmentation fault"
        if "stack overflow" in combined.lower():
            return "stack overflow"

        return super().extract_crash_signature(stdout, stderr, return_code)
