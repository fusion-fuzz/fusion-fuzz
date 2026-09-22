"""
projects/cuda/driver.py — compile a fused CUDA translation unit with the
clang CUDA frontend or with nvcc, and say whether the result is a bug.

Compile only. Every mode stops at syntax checking, LLVM IR, PTX, cubin
or an object file — nothing is linked or launched, so no GPU and no
driver are needed. What varies per run:

  compiler   clang (weight 0.6): the frontend this adapter is built to
             stress, with assertions when projects/clang's build is used.
             nvcc (0.4): NVIDIA's EDG frontend + cicc + ptxas; closed
             source, so a crash is reported from its messages alone.
  side       device only (most of the weight: the CUDA-specific
             code paths), host only, or both.
  depth      -fsyntax-only < -emit-llvm < -S (PTX) < -c (ptxas + fatbin).
  target     --cuda-gpu-arch / -arch from Maxwell to Blackwell: the
             NVPTX backend's feature handling and ptxas differ per
             architecture, and the lit tests mostly pin sm_20-sm_70.
  dialect    C++14..C++23, -O0..-O3 plus fast-math / denormal / rdc /
             short-pointer / relaxed-constexpr / extended-lambda knobs.

Includes: -I for cuda-samples' Common/ (helper headers) is always passed;
lit tests need nothing beyond the toolkit. nvcc writes its intermediate
files to TMPDIR, which is pointed at the per-run work directory.
"""

import json
import os
import random
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ffl_cuda_analyzer", os.path.join(_here, "analyzer.py"))
_an = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_an)
classify = _an.classify


class CUDADriver(BaseDriver):
    CLANG_SHARE = 0.6
    ARCHS = ["sm_52", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a"]
    ARCH_WEIGHTS = [1, 1, 1, 2, 2, 4, 2, 2, 4, 1]
    NVCC_EXTRA_ARCHS = ["sm_100", "sm_120", "compute_80", "compute_90"]
    STD = ["c++14", "c++17", "c++17", "c++20", "c++20", "c++23"]
    OPT = ["-O0", "-O1", "-O2", "-O2", "-O3", "-O3", "-Os"]
    # (flags, tool tag) — side + depth for clang
    CLANG_MODES = [
        ("--cuda-device-only -fsyntax-only", 15),
        ("--cuda-device-only -emit-llvm -S -o /dev/null", 20),
        ("--cuda-device-only -S -o /dev/null", 25),          # NVPTX → PTX text
        ("--cuda-device-only -c -o /dev/null", 15),          # + ptxas
        ("--cuda-host-only -fsyntax-only", 5),
        ("--cuda-host-only -S -emit-llvm -o /dev/null", 5),
        ("-fsyntax-only", 5),
        ("-c -o /dev/null", 10),                              # both sides + fatbin
    ]
    CLANG_FLAGS = [
        "-ffast-math", "-fcuda-flush-denormals-to-zero", "-fno-cuda-approx-transcendentals",
        "-fgpu-rdc", "-fcuda-short-ptr", "-fgpu-defer-diag", "-fgpu-exclude-wrong-side-overloads",
        "-Wall -Wextra", "-g", "-fno-inline", "-funroll-loops", "-ffp-contract=fast",
        "-fno-strict-aliasing", "-fno-exceptions", "-fno-rtti", "-fcuda-is-device",
        "-Xclang -fcuda-allow-variadic-functions", "-fgpu-approx-transcendentals",
        "-mllvm -nvptx-sched4reg", "-fno-gpu-rdc", "-fstrict-enums", "-fvectorize",
    ]
    NVCC_MODES = [
        ("--ptx -o /dev/null", 30),
        ("--cubin -o /dev/null", 20),
        ("-c -o /dev/null", 25),
        ("--fatbin -o /dev/null", 10),
        ("-dc -o /dev/null", 15),                              # relocatable device code
    ]
    NVCC_FLAGS = [
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--extended-lambda",
        "--use_fast_math", "-Xptxas -O0", "-Xptxas -O3", "-Xptxas -v", "-lineinfo",
        "--fmad=false", "--ftz=true", "--prec-div=false", "--prec-sqrt=false",
        "-G", "-rdc=true", "-Xcicc -O0", "--restrict", "-std=c++17", "-Wno-deprecated-gpu-targets",
        "--device-debug", "-maxrregcount=32", "--extra-device-vectorization",
    ]

    def __init__(self, config):
        super().__init__(config)
        root = os.path.join(self.ffl_root, "projects", "cuda")
        info = {}
        try:
            with open(os.path.join(root, "toolchain.json")) as f:
                info = json.load(f)
        except Exception:
            pass
        self.clang = info.get("clang") or shutil.which("clang++")
        self.nvcc = info.get("nvcc") or shutil.which("nvcc")
        self.cuda_path = info.get("cuda_path") or "/usr/local/cuda"
        self.include_dirs = [d for d in info.get("include_dirs", []) if os.path.isdir(d)]
        self.mem_limit_mb = int(config.get("execution", {}).get("mem_limit_mb", 0) or 0)

    # ── command construction ──────────────────────────────────────────

    def _ulimit(self):
        return f"ulimit -v {self.mem_limit_mb * 1024}; " if self.mem_limit_mb else ""

    def _includes(self):
        return " ".join(f"-I{d}" for d in self.include_dirs)

    def _clang_command(self, seed_file, dryrun):
        arch = random.choices(self.ARCHS, weights=self.ARCH_WEIGHTS, k=1)[0]
        base = (f"{self.clang} -x cuda --cuda-path={self.cuda_path} --cuda-gpu-arch={arch} "
                f"-Wno-unknown-cuda-version -Wno-everything -ferror-limit=5 {self._includes()}")
        if dryrun:
            return f"{base} --cuda-device-only -fsyntax-only {seed_file}"
        modes, weights = zip(*self.CLANG_MODES)
        mode = random.choices(modes, weights=weights, k=1)[0]
        flags = [mode, random.choice(self.OPT), f"-std={random.choice(self.STD)}"]
        k = random.choice([0, 1, 1, 2, 3])
        flags += random.sample(self.CLANG_FLAGS, k)
        if "-fcuda-is-device" in flags:
            # a cc1 flag; only meaningful with device-only compilation
            flags.remove("-fcuda-is-device")
        return f"{base} {' '.join(flags)} {seed_file}"

    def _nvcc_command(self, seed_file, dryrun):
        arch = random.choices(self.ARCHS + self.NVCC_EXTRA_ARCHS,
                              weights=self.ARCH_WEIGHTS + [1, 1, 1, 1], k=1)[0]
        base = f"{self.nvcc} -arch={arch} -w {self._includes()}"
        if dryrun:
            return f"{base} --ptx -o /dev/null {seed_file}"
        modes, weights = zip(*self.NVCC_MODES)
        mode = random.choices(modes, weights=weights, k=1)[0]
        flags = [mode, random.choice(self.OPT)]
        std = random.choice(self.STD)
        if std != "c++23":                      # nvcc 12.8 stops at c++20
            flags.append(f"-std={std}")
        k = random.choice([0, 1, 1, 2, 3])
        flags += random.sample(self.NVCC_FLAGS, k)
        if "-G" in flags or "--device-debug" in flags:
            flags = [f for f in flags if not f.startswith("-Xptxas") and f != "-lineinfo"]
        if mode.startswith("-dc") and "-rdc=true" in flags:
            flags.remove("-rdc=true")
        return f"{base} {' '.join(flags)} {seed_file}"

    def _build_command(self, seed_file, dryrun=False):
        use_clang = bool(self.clang) and (not self.nvcc or random.random() < self.CLANG_SHARE)
        if dryrun:
            use_clang = bool(self.clang)
        if use_clang:
            return f"{self._ulimit()}{self._clang_command(seed_file, dryrun)}", "clang"
        return f"{self._ulimit()}{self._nvcc_command(seed_file, dryrun)}", "nvcc"

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd, tool = "unknown", "clang"
        rc, stdout, stderr = 1, "", ""
        seed_file = None
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.cu")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            cmd, tool = self._build_command(seed_file, dryrun=getattr(self, "dryrun_mode", False))
            # nvcc writes its intermediates to TMPDIR; $PWD is the per-run
            # work directory here and the bundle directory when test.sh is
            # replayed (a literal path would name a directory that is gone).
            cmd = f'export TMPDIR="$PWD"; {cmd}'
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        duration = time.time() - start
        verdict = classify((stdout or "") + "\n" + (stderr or ""), tool, rc)
        crashed = bool(verdict["is_bug"])
        if verdict["kind"] in ("resource", "timeout", "unsupported"):
            # the orchestrator's validity rule keys on "error:" in stderr
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
