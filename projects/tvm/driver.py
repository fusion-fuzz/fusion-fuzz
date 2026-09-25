"""
projects/tvm/driver.py — run a fused TVMScript module through
projects/tvm/runner.py and say whether what came back is a bug.

The runner (Python, imports the built TVM) does the compiler-facing work;
this file draws the configuration and contains the run:

  mode        parse (TVMScript + verifier only), build (passes + codegen),
              run (build + execute on random inputs), diff (run, and run
              again at `llvm -opt-level=0`; disagreement is a miscompile
              candidate).
  target      the GPU backends are the primary target and take most of the
              build draws: `cuda` (CUDA C++ codegen, and with nvcc in the
              image the generated source is compiled to PTX), `nvptx`
              (LLVM NVPTX back end), `rocm`, `vulkan` (SPIR-V), `opencl`,
              `metal` and `webgpu`. All of them are compile-only — no GPU
              is present and nothing is launched — which is what makes
              them usable here at all. Each GPU target also draws device
              limits (`max_num_threads`, `max_shared_memory_per_block`,
              `thread_warp_size`, ...) so the split/tiling/bound checks
              that read them see device shapes no real part has.
              CPU targets remain for the run/diff oracles: llvm with
              CPU/feature variants (the host is x86-64), an aarch64 cross
              target for build-only modes, and the C backend.
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
    MODES = [("parse", 8), ("build", 57), ("run", 15), ("diff", 20)]
    #: CPU build targets. Drawn for the (1 - GPU_SHARE) part of build runs.
    BUILD_TARGETS = [
        ("llvm", 6), ("llvm -mcpu=core-avx2", 3), ("llvm -mcpu=skylake-avx512", 3),
        ("llvm -mcpu=znver3", 1), ("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72", 2),
        ("llvm -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+v", 1), ("c", 2),
    ]
    #: GPU targets, compile-only (see runner.GPU_KINDS). `cuda` and
    #: `nvptx` carry an `arch`/`mcpu`; the source-level codegens
    #: (opencl/metal/webgpu) take none. Availability is probed once in
    #: __init__ — a target whose codegen this build lacks raises
    #: "Cannot find global function target.build.<kind>" and would count
    #: every draw as a rejection.
    #: rocm is left out: TVM links AMD device bitcode at codegen time and
    #: Ubuntu's rocm-device-libs 5.0 has neither the `oclc_abi_version_*`
    #: modules TVM asks for nor an ISA module for anything past gfx90c, so
    #: every draw would be a rejection. The codegen is built and works
    #: given a real ROCm install (set ROCM_PATH).
    GPU_TARGETS = [
        ("cuda", 10), ("nvptx", 5), ("vulkan", 4), ("opencl", 3),
        ("metal", 2), ("webgpu", 2),
    ]
    #: Share of build-mode runs that go to a GPU target.
    GPU_SHARE = 0.75
    CUDA_ARCHS = ["sm_50", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a",
                  "sm_100", "sm_120"]
    CUDA_ARCH_WEIGHTS = [1, 2, 2, 4, 2, 2, 4, 1, 2, 1]
    ROCM_ARCHS = ["gfx900", "gfx90a", "gfx942", "gfx1030", "gfx1100"]
    #: nvcc arch for the generated-CUDA-source stage (runner --nvcc-arch);
    #: only real (`sm_`) architectures, and only when nvcc is in the image.
    NVCC_SHARE = 0.4
    #: Device limits the GPU codegens read. Deliberately includes values
    #: below what any shipped part has: a 64-thread block limit or 4 KB of
    #: shared memory forces the splitting and bound-check paths that a
    #: realistic limit never reaches.
    #: Which options each target kind actually declares (from each
    #: backend's target_kind.cc). Passing one a kind does not know raises
    #: "Unknown config option", which would make every such draw a harness
    #: error rather than a test.
    KIND_OPTIONS = {
        "cuda": ("max_num_threads", "max_threads_per_block",
                 "max_shared_memory_per_block", "thread_warp_size",
                 "registers_per_block", "l2_cache_size_bytes"),
        "nvptx": ("max_num_threads", "thread_warp_size"),
        "rocm": ("max_num_threads", "max_threads_per_block",
                 "max_shared_memory_per_block", "thread_warp_size"),
        "vulkan": ("max_num_threads", "max_threads_per_block", "thread_warp_size",
                   "max_shared_memory_per_block", "max_block_size_x",
                   "max_block_size_y", "max_block_size_z",
                   "max_push_constants_size", "max_storage_buffer_range"),
        "opencl": ("max_num_threads", "max_threads_per_block",
                   "max_shared_memory_per_block", "thread_warp_size",
                   "max_function_args", "texture_spatial_limit"),
        "metal": ("max_num_threads", "max_threads_per_block",
                  "max_shared_memory_per_block", "thread_warp_size",
                  "max_function_args"),
        # thread_warp_size is only legal for WebGPU together with
        # supports_subgroups ("WebGPU target with thread_warp_size > 1
        # must declare subgroups"), so it is not drawn here
        "webgpu": ("max_num_threads", "max_shared_memory_per_block"),
    }
    GPU_LIMITS = {
        "max_num_threads": ["64", "256", "512", "1024", "2048"],
        "max_threads_per_block": ["64", "256", "1024"],
        "max_shared_memory_per_block": ["4096", "16384", "49152", "232448"],
        "thread_warp_size": ["1", "16", "32", "64"],
    }
    #: Vulkan capability bits. The defaults are restrictive (TVM assumes a
    #: baseline device), and a module using a type the target does not
    #: declare is refused with "Vulkan target does not support Int8
    #: capability" — a rejection, not a codegen bug. So the permissive set
    #: is declared on every Vulkan draw and individual bits are turned off
    #: only sometimes: that way the SPIR-V codegen runs, and the "device
    #: cannot do this" path is still covered.
    VULKAN_CAPS = ["supports_float16", "supports_float64", "supports_int8",
                   "supports_int16", "supports_int64", "supports_8bit_buffer",
                   "supports_16bit_buffer", "supports_storage_buffer_storage_class",
                   "supports_integer_dot_product", "supports_cooperative_matrix"]
    VULKAN_FLAGS = {"max_spirv_version": ["66560", "67072", "67584"]}
    # executed on this host: only ISAs it has (AVX-512 code SIGILLs here)
    RUN_TARGETS = [("llvm", 4), ("llvm -mcpu=core-avx2", 2), ("llvm -mcpu=native", 2)]
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
        self.gpu_targets = self._available_gpu_targets(root)
        self.nvcc = shutil.which("nvcc")

    def _available_gpu_targets(self, root):
        """The GPU target kinds this TVM build can actually generate code
        for, asked of the build itself (`target.build.<kind>` is the
        registered codegen). A kind the build lacks would make every draw
        a rejection, so it is dropped here rather than filtered later."""
        import subprocess
        names = [n for n, _ in self.GPU_TARGETS]
        script = (
            "import tvm, tvm_ffi\n"
            "for k in %r:\n"
            "    try:\n"
            "        f = tvm_ffi.get_global_func('target.build.' + k, allow_missing=True)\n"
            "    except Exception:\n"
            "        f = None\n"
            "    if f is not None:\n"
            "        print(k)\n" % (names,))
        env = dict(os.environ, PYTHONPATH=self.pythonpath, TVM_LIBRARY_PATH=self.lib_dir)
        try:
            r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                               timeout=120, env=env, cwd=root)
            found = {l.strip() for l in r.stdout.splitlines() if l.strip() in names}
        except Exception:
            found = set()
        if not found:
            # probe failed (not a reason to stop fuzzing): keep the
            # source-level codegens, which need no toolkit at all
            found = {"cuda", "opencl", "metal", "webgpu"}
        return [(n, w) for n, w in self.GPU_TARGETS if n in found]

    def _draw_gpu_target(self):
        """(target string, nvcc arch or "")."""
        names, weights = zip(*self.gpu_targets)
        kind = random.choices(names, weights=weights, k=1)[0]
        parts = [kind]
        nvcc_arch = ""
        if kind in ("cuda", "nvptx"):
            arch = random.choices(self.CUDA_ARCHS, weights=self.CUDA_ARCH_WEIGHTS, k=1)[0]
            parts.append(f"-arch={arch}" if kind == "cuda" else f"-mcpu={arch}")
            if kind == "cuda" and self.nvcc and random.random() < self.NVCC_SHARE:
                nvcc_arch = arch
        elif kind == "rocm":
            parts.append(f"-mcpu={random.choice(self.ROCM_ARCHS)}")
        allowed = self.KIND_OPTIONS.get(kind, ())
        for name, values in self.GPU_LIMITS.items():
            if name in allowed and random.random() < 0.3:
                parts.append(f"-{name}={random.choice(values)}")
        if kind == "vulkan":
            for cap in self.VULKAN_CAPS:
                parts.append(f"-{cap}=" + ("0" if random.random() < 0.15 else "1"))
            for name, values in self.VULKAN_FLAGS.items():
                if random.random() < 0.3:
                    parts.append(f"-{name}={random.choice(values)}")
        return " ".join(parts), nvcc_arch

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
        nvcc_arch = ""
        if mode == "build" and self.gpu_targets and random.random() < self.GPU_SHARE:
            target, nvcc_arch = self._draw_gpu_target()
        else:
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
        if nvcc_arch:
            cmd += f" --nvcc-arch {nvcc_arch}"
        # The tag goes into the crash signature, so it has to say what kind
        # of path this was, not which of the six GPU targets was drawn: one
        # codegen ICHECK otherwise became a separate bundle per target
        # (build-cuda, build-metal, build-opencl, ...). The exact target is
        # in the bundle's own command.
        kind = target.split()[0]
        if kind in ("cuda", "nvptx", "rocm", "vulkan", "opencl", "metal", "webgpu"):
            return cmd, f"{mode}-gpu"
        return cmd, (f"{mode}-{kind}" if kind != "llvm" else mode)

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
