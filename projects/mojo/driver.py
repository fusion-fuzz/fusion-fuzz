"""
projects/mojo/driver.py — compile (and sometimes run) a fused Mojo module
and say whether what came back is a bug.

Toolchain (recorded by setup.py in projects/mojo/toolchain.json): the
`mojo` driver built from source under projects/mojo/bin (assertions on)
when the Bazel build has finished, otherwise the released wheel in
projects/mojo/venv. `kgen-translate` (parse + elaborate only) is drawn
when the from-source build provides it.

What varies per run:

  mode       build to an object / LLVM IR / assembly / executable
             (compile only, most of the weight), or build-and-run: the
             executable runs under a timeout and its own assertions
             (-D ASSERT=all, the stdlib's debug_assert) are on, so a
             miscompile surfaces as a program crash the analyzer keeps in
             its own class.
  flags      -O0..-O3, --debug-level none|line-tables|full, -D ASSERT=
             all|warn|none, --fp-mode contract=off, --sanitize address,
             --target-cpu / --mcpu, --Werror or --disable-warnings, and
             the test suite's own combinations (lit's %mojo is
             `-Werror -D ASSERT=all --debug-level full`).

Includes: -I for Mojo/stdlib/test (the tests' `test_utils` package) and
Mojo/test/test-packages (the parser tests' stub packages). A module
without `def main` (parser tests) gets a trivial one appended so that it
compiles as a program. Compilation is capped at two threads per run so
eight workers do not oversubscribe the box; the crash-reporter and
telemetry are disabled by environment.
"""

import json
import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ffl_mojo_analyzer", os.path.join(_here, "analyzer.py"))
_an = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_an)
classify = _an.classify

_MAIN_RE = re.compile(r'^def\s+main\s*\(', re.M)


class MojoDriver(BaseDriver):
    RUN_SHARE = 0.35
    PARSE_SHARE = 0.15                 # only when kgen-translate exists
    EMITS = [("object", 40), ("llvm", 15), ("asm", 15), ("exe", 30)]
    OPT = ["-O0", "-O1", "-O2", "-O3", "-O3"]
    DEBUG = ["--debug-level none", "--debug-level line-tables", "--debug-level full", "--debug-level full"]
    ASSERT = ["-DASSERT=all", "-DASSERT=all", "-DASSERT=warn", "-DASSERT=none"]
    MISC = [
        "--fp-mode contract=off", "--Werror", "--disable-warnings", "--warn-on-unstable-apis",
        "--target-cpu x86-64", "--mcpu=x86-64-v3", "--mcpu=skylake", "--mcpu=znver3",
        "--target-features -avx2", "--elaboration-error-include-prelude",
        "--max-notes-per-diagnostic 1", "--diagnose-missing-doc-strings",
    ]
    SANITIZE = ["--sanitize address"]
    SANITIZE_SHARE = 0.08

    def __init__(self, config):
        super().__init__(config)
        root = os.path.join(self.ffl_root, "projects", "mojo")
        info = {}
        try:
            with open(os.path.join(root, "toolchain.json")) as f:
                info = json.load(f)
        except Exception:
            pass
        self.mojo = info.get("mojo") or shutil.which("mojo")
        self.kgen_translate = info.get("kgen_translate")
        self.include_dirs = [d for d in info.get("include_dirs", []) if os.path.isdir(d)]
        self.search_paths = info.get("search_paths", [])
        self.env = dict(info.get("env", {}))
        self.env.setdefault("MODULAR_CRASH_REPORTING_ENABLED", "false")
        self.env.setdefault("MODULAR_TELEMETRY_ENABLED", "false")
        self.mem_limit_mb = int(config.get("execution", {}).get("mem_limit_mb", 0) or 0)
        self.timeout = int(config.get("execution", {}).get("timeout", 120) or 120)

    # ── command construction ──────────────────────────────────────────

    def _prefix(self, workdir):
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        cap = f"ulimit -v {self.mem_limit_mb * 1024}; " if self.mem_limit_mb else ""
        return f'export {env} HOME={workdir} TMPDIR={workdir}; {cap}'

    def _includes(self):
        return " ".join(f"-I {d}" for d in self.include_dirs)

    def _flags(self, run):
        flags = [random.choice(self.OPT), random.choice(self.DEBUG), random.choice(self.ASSERT), "-j 1"]
        k = random.choice([0, 1, 1, 2])
        flags += random.sample(self.MISC, k)
        if "--Werror" in flags and "--disable-warnings" in flags:
            flags.remove("--Werror")
        if random.random() < self.SANITIZE_SHARE:
            flags.append(random.choice(self.SANITIZE))
        return flags

    def _build_command(self, seed_file, workdir, dryrun=False):
        pre = self._prefix(workdir)
        inc = self._includes()
        if dryrun:
            # pre-analysis: parse + elaborate only when kgen-translate is
            # there (1-2 s a seed against 5-60 s for a full build)
            if self.kgen_translate:
                sp = ",".join(self.search_paths + self.include_dirs)
                return (f"{pre}{self.kgen_translate} -import-mojo -mojo-enable-prebuilt-packages "
                        f"-mojo-search-paths={sp} {seed_file} -o /dev/null"), "parse"
            return f"{pre}{self.mojo} build --emit object -j 1 -DASSERT=all {inc} -o {workdir}/out.o {seed_file}", "build"
        if self.kgen_translate and random.random() < self.PARSE_SHARE:
            sp = ",".join(self.search_paths + self.include_dirs)
            return (f"{pre}{self.kgen_translate} -import-mojo -mojo-enable-prebuilt-packages "
                    f"-mojo-search-paths={sp} {seed_file} -o /dev/null"), "parse"
        flags = self._flags(run=False)
        if random.random() < self.RUN_SHARE:
            exe = os.path.join(workdir, "prog")
            cmd = (f"{pre}{self.mojo} build {' '.join(flags)} {inc} -o {exe} {seed_file} && "
                   f"echo FFL_COMPILED_OK && timeout {max(10, self.timeout // 3)} {exe}")
            return cmd, "run"
        emits, weights = zip(*self.EMITS)
        emit = random.choices(emits, weights=weights, k=1)[0]
        out = os.path.join(workdir, "out." + {"object": "o", "llvm": "ll", "asm": "s", "exe": "exe"}[emit])
        return f"{pre}{self.mojo} build --emit {emit} {' '.join(flags)} {inc} -o {out} {seed_file}", "build"

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd, tool = "unknown", "build"
        rc, stdout, stderr = 1, "", ""
        seed_file = None
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.mojo")
            content = seed.content
            if not _MAIN_RE.search(content):
                content = content.rstrip("\n") + "\n\n\ndef main():\n    pass\n"
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(content)
            cmd, tool = self._build_command(seed_file, workdir, dryrun=getattr(self, "dryrun_mode", False))
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        duration = time.time() - start
        compiled_ok = "FFL_COMPILED_OK" in (stdout or "") if tool == "run" else None
        may_trap = bool(re.search(r'\babort\s*\(|os\.abort|__builtin_trap|\bunreachable\b|debug_assert|\btrap\(|'
                                  r'\.critical\(|\bexit\(|\bargv\(\)', seed.content))
        verdict = classify((stdout or "") + "\n" + (stderr or ""), tool, rc, compiled_ok, may_trap)
        crashed = bool(verdict["is_bug"])
        if verdict["kind"] in ("resource", "timeout", "unsupported", "rejected") and \
                not re.search(r"error: ", stderr or ""):
            stderr = f"error: module {verdict['kind']}\n" + (stderr or "")
        res = ExecutionResult(rc, stdout, stderr, duration, crashed,
                              verdict["signature"] if crashed else None)
        res.command = cmd
        res.seed_file = seed_file
        res.tool = tool
        res.verdict = verdict["kind"]
        return res

    # ── oracle hooks (BaseDriver) ─────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        return bool(classify((stdout or "") + "\n" + (stderr or ""),
                             return_code=return_code)["is_bug"])

    def extract_crash_signature(self, stdout, stderr, return_code):
        return classify((stdout or "") + "\n" + (stderr or ""),
                        return_code=return_code)["signature"]
