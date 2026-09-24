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
    # nvcc 13 knows the Blackwell parts and the family/arch-specific
    # variants; `all-major` compiles one SASS per major generation and is
    # the heaviest single configuration, so it is drawn least.
    NVCC_EXTRA_ARCHS = ["sm_100", "sm_100a", "sm_103", "sm_120", "sm_120a", "sm_121"]
    NVCC_EXTRA_WEIGHTS = [2, 1, 1, 2, 1, 1]
    #: Virtual (PTX-only) architectures. nvcc refuses `--cubin`/`-c`/
    #: `--fatbin` for one of these ("not allowed when compiling for a
    #: virtual compute architecture"), so they are only drawn for `--ptx`.
    NVCC_VIRTUAL_ARCHS = ["compute_75", "compute_80", "compute_90", "compute_100", "compute_120"]
    #: How often several device architectures are compiled at once, so the
    #: fat-binary path and per-arch specialisation run. nvcc allows that
    #: only for the modes that produce an object or a fatbin; `--ptx` and
    #: `--cubin` are refused ("not allowed when compiling for multiple GPU
    #: code instances"), so the draw is conditioned on the mode.
    MULTI_ARCH_SHARE = 0.25
    NVCC_MULTI_ARCH_MODES = ("-c", "--fatbin", "-dc")
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
        # ptxas / device-side back-end knobs (device-only and full compiles)
        "-Xcuda-ptxas -O0", "-Xcuda-ptxas -O3", "-Xcuda-ptxas --allow-expensive-optimizations=true",
        "-Xcuda-ptxas --maxrregcount=32", "--cuda-noopt-device-debug", "-fno-cuda-flush-denormals-to-zero",
        "-mllvm -nvptx-prec-divf32=0", "-mllvm -nvptx-prec-sqrtf32=0", "-mllvm -nvptx-fma-level=0",
        "-fno-vectorize", "-fno-unroll-loops", "-fdenormal-fp-math=preserve-sign",
        "-Xclang -disable-llvm-passes", "-Xclang -disable-llvm-optzns", "-fno-gpu-defer-diag",
        "-fcuda-short-ptr -fgpu-rdc",
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
        # ptxas and cicc knobs, device-link optimisation, threading
        "-Xptxas -O1", "-Xptxas -O2", "-Xptxas --allow-expensive-optimizations=true",
        "-Xptxas --def-load-cache=cg", "-Xptxas --warn-on-spills", "-Xptxas --disable-optimizer-constants",
        "-Xcicc -O3", "-dopt on", "--split-compile=2", "--generate-line-info",
        "-default-stream per-thread", "--display-error-number", "-Xcompiler -fno-strict-aliasing",
        "--no-host-device-initializer-list", "--no-host-device-move-forward",
        "-Xptxas --fmad=false", "-Xnvlink -O0",
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
        archs = f"--cuda-gpu-arch={arch}"
        multi = not dryrun and random.random() < self.MULTI_ARCH_SHARE
        if multi:
            second = random.choice([a for a in self.ARCHS if a != arch])
            archs += f" --cuda-gpu-arch={second}"
        base = (f"{self.clang} -x cuda --cuda-path={self.cuda_path} {archs} "
                f"-Wno-unknown-cuda-version -Wno-everything -ferror-limit=5 {self._includes()}")
        if dryrun:
            return f"{base} --cuda-device-only -fsyntax-only {seed_file}"
        modes, weights = zip(*self.CLANG_MODES)
        mode = random.choices(modes, weights=weights, k=1)[0]
        if multi:
            # clang writes one file per device arch and refuses `-o` then
            # ("cannot specify -o when generating multiple output files");
            # -fsyntax-only has no output, the others write next to $PWD
            # (the per-run work directory, removed afterwards)
            mode = mode.replace(" -o /dev/null", "")
        flags = [mode, random.choice(self.OPT), f"-std={random.choice(self.STD)}"]
        k = random.choice([0, 1, 1, 2, 3])
        flags += random.sample(self.CLANG_FLAGS, k)
        if "-fcuda-is-device" in flags:
            # a cc1 flag; only meaningful with device-only compilation
            flags.remove("-fcuda-is-device")
        return f"{base} {' '.join(flags)} {seed_file}"

    def _nvcc_command(self, seed_file, dryrun):
        real_archs = self.ARCHS + self.NVCC_EXTRA_ARCHS
        real_weights = self.ARCH_WEIGHTS + self.NVCC_EXTRA_WEIGHTS
        if dryrun:
            arch = random.choices(real_archs, weights=real_weights, k=1)[0]
            return (f"{self.nvcc} -arch={arch} -w {self._includes()} "
                    f"--ptx -o /dev/null {seed_file}")
        modes, weights = zip(*self.NVCC_MODES)
        mode = random.choices(modes, weights=weights, k=1)[0]
        head = mode.split()[0]
        if head == "--ptx" and random.random() < 0.25:
            # PTX for a virtual architecture: the cicc side only, no SASS
            archs = f"-arch={random.choice(self.NVCC_VIRTUAL_ARCHS)}"
        elif head in self.NVCC_MULTI_ARCH_MODES and random.random() < self.MULTI_ARCH_SHARE:
            if random.random() < 0.3:
                archs = "-arch=all-major"      # one SASS per major generation
            else:
                picks = random.sample([a for a in real_archs if a[-1].isdigit()], 2)
                archs = " ".join(f"-gencode arch=compute_{a[3:]},code={a}" for a in picks)
        else:
            arch = random.choices(real_archs, weights=real_weights, k=1)[0]
            archs = f"-arch={arch}"
        base = f"{self.nvcc} {archs} -w {self._includes()}"
        flags = [mode, random.choice(self.OPT)]
        std = random.choice(self.STD)
        if std != "c++23":                      # nvcc 12.8 stops at c++20
            flags.append(f"-std={std}")
        k = random.choice([0, 1, 1, 2, 3])
        flags += random.sample(self.NVCC_FLAGS, k)
        if "-G" in flags or "--device-debug" in flags:
            flags = [f for f in flags if not f.startswith("-Xptxas") and f != "-lineinfo"
                     and f not in ("--generate-line-info", "-dopt on", "--split-compile=2")]
        if "-dopt on" in flags and mode.split()[0] not in ("-dc", "-c"):
            flags.remove("-dopt on")          # device-link optimisation needs a link step
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
