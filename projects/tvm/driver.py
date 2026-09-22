"""
projects/tvm/driver.py — run a fused TVMScript module through
projects/tvm/runner.py and say whether what came back is a bug.

The runner (Python, imports the built TVM) does the compiler-facing work;
this file draws the configuration and contains the run:

  mode        parse (TVMScript + verifier only), build (passes + codegen),
              run (build + execute on random inputs), diff (run, and run
              again at `llvm -opt-level=0`; disagreement is a miscompile
              candidate).
  target      llvm with CPU/feature variants (the host is x86-64: AVX2 and
              AVX-512 code paths both compile; only the host's own ISA is
              executed), an aarch64 cross target for build-only modes, and
              the C backend.
  opt level   PassContext opt_level 0..3 and the target's own -opt-level.
  passes      0..4 no-argument TIR/Relax transforms drawn by the runner
              from its --seed (the pass names are printed, and replaying
              test.sh with the same --seed draws the same ones).
  pipeline    tvm.compile's default TIR pipeline or the `tirx` one.
"""

import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ffl_tvm_analyzer", os.path.join(_here, "analyzer.py"))
_an = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_an)
classify = _an.classify


class TVMDriver(BaseDriver):
    MODES = [("parse", 10), ("build", 45), ("run", 20), ("diff", 25)]
    BUILD_TARGETS = [
        ("llvm", 6), ("llvm -mcpu=core-avx2", 3), ("llvm -mcpu=skylake-avx512", 3),
        ("llvm -mcpu=znver3", 1), ("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72", 2),
        ("llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+v", 1), ("c", 2),
    ]
    RUN_TARGETS = [("llvm", 4), ("llvm -mcpu=core-avx2", 2), ("llvm -mcpu=native", 2),
                   ("llvm -mcpu=skylake-avx512", 1)]
    OPT_LEVELS = [0, 1, 2, 3, 3]
    PASS_COUNTS = [0, 0, 1, 1, 2, 3, 4]
    PIPELINES = [("default", 3), ("tirx", 1)]

    def __init__(self, config):
        super().__init__(config)
        root = os.path.join(self.ffl_root, "projects", "tvm")
        self.runner = os.path.join(root, "runner.py")
        # tvm_ffi comes from the package setup.py installs (its compiled
        # `core` extension is not part of TVM's CMake build), so the
        # submodule's source tree must not shadow it
        # tvm_ffi (its compiled `core`) is installed in the user site;
        # HOME is redirected per run, which would hide that directory, so
        # it is named on PYTHONPATH explicitly (computed once, real HOME).
        import site
        pythonpath = [os.path.join(root, "tvm", "python"), site.getusersitepackages()]
        self.pythonpath = ":".join(p for p in pythonpath if os.path.isdir(p))
        self.lib_dir = os.path.join(root, "tvm", "build", "lib")
        self.mem_limit_mb = int(config.get("execution", {}).get("mem_limit_mb", 0) or 0)

    def _prefix(self, workdir):
        cap = f"ulimit -v {self.mem_limit_mb * 1024}; " if self.mem_limit_mb else ""
        return (f'export PYTHONPATH={self.pythonpath} TVM_LIBRARY_PATH={self.lib_dir} '
                f'TVM_LOG_DEBUG=0 HOME={workdir} TMPDIR={workdir} OMP_NUM_THREADS=1 TVM_NUM_THREADS=1; {cap}')

    def _build_command(self, seed_file, workdir, dryrun=False):
        pre = self._prefix(workdir)
        if dryrun:
            return f"{pre}python3 {self.runner} {seed_file} --mode build --target llvm --opt-level 3", "build"
        modes, w = zip(*self.MODES)
        mode = random.choices(modes, weights=w, k=1)[0]
        targets = self.RUN_TARGETS if mode in ("run", "diff") else self.BUILD_TARGETS
        names, tw = zip(*targets)
        target = random.choices(names, weights=tw, k=1)[0]
        opt = random.choice(self.OPT_LEVELS)
        npass = random.choice(self.PASS_COUNTS) if mode != "parse" else 0
        pipes, pw = zip(*self.PIPELINES)
        pipe = random.choices(pipes, weights=pw, k=1)[0]
        seed = random.randrange(1 << 30)
        cmd = (f'{pre}python3 {self.runner} {seed_file} --mode {mode} --target "{target}" '
               f'--opt-level {opt} --num-passes {npass} --tir-pipeline {pipe} --seed {seed}')
        return cmd, mode

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd, tool = "unknown", "build"
        rc, stdout, stderr = 1, "", ""
        seed_file = None
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.py")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            cmd, tool = self._build_command(seed_file, workdir, dryrun=getattr(self, "dryrun_mode", False))
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        duration = time.time() - start
        if stderr:
            # TVM's LLVM instance logs an "Error: Using LLVM ... -mcpu=... is
            # not valid" line at import for CPU names the linked LLVM lacks,
            # and Python deprecation UserWarnings; neither is about the seed
            # and the orchestrator's validity rule would read "Error:" as a
            # rejection.
            stderr = "\n".join(l for l in stderr.splitlines()
                                if "llvm_instance.cc" not in l and "UserWarning" not in l
                                and not l.strip().startswith("warnings.warn"))
        verdict = classify((stdout or "") + "\n" + (stderr or ""), tool, rc)
        crashed = bool(verdict["is_bug"])
        if verdict["kind"] in ("resource", "timeout", "rejected") and "error: " not in (stderr or ""):
            stderr = f"error: module {verdict['kind']}\n" + (stderr or "")
        res = ExecutionResult(rc, stdout, stderr, duration, crashed,
                              verdict["signature"] if crashed else None)
        res.command = cmd
        res.seed_file = seed_file
        res.tool = tool
        res.verdict = verdict["kind"]
        return res

    def _check_crash(self, stdout, stderr, return_code):
        return bool(classify((stdout or "") + "\n" + (stderr or ""), return_code=return_code)["is_bug"])

    def extract_crash_signature(self, stdout, stderr, return_code):
        return classify((stdout or "") + "\n" + (stderr or ""), return_code=return_code)["signature"]
