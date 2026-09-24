"""
projects/xla/driver.py — run a fused HLO module through XLA's CPU backend
and report whether what came back is a bug.

This file owns execution: writing the module, choosing the tool, the pass
list and the debug-option flags, containing the run, cleaning up.
Judgement lives in projects/xla/analyzer.py.

Two tools, two oracles
----------------------
  run_hlo_module   compile for the CPU backend (the full HLO pipeline, then
                   LLVM codegen through XLA's emitters and the thunk
                   runtime), execute on random inputs and — most of the
                   time — run the same module through the HLO evaluator
                   and compare. That is the crash oracle over the whole
                   compiler *and* a differential oracle over the generated
                   code. It is the default for modules the CPU runner can
                   execute.

  hlo-opt          the pass driver. `--stage=hlo` runs the CPU backend's
                   HLO optimisation pipeline and prints the result;
                   `--stage=llvm`/`llvm-before-optimizations` lowers to
                   LLVM IR (the emitters, without executing);
                   `--passes=a,b,c` runs an arbitrary subset of the ~170
                   registered passes in an arbitrary order, which is where
                   pass-ordering assumptions get tested. Modules that use
                   collectives or custom calls the CPU runner cannot
                   execute still go through here.

Flags
-----
XLA's DebugOptions are all reachable as command-line flags on both tools.
The driver draws a handful per run — optimisation level, fast-math and its
sub-flags, the vector width / ISA cap, xnnpack, the concurrency-optimised
scheduler, tiling propagation, the new xtile lowering, region-based copy
insertion, the parallel-codegen split — so the same module meets several
backend configurations over time. Every flag name here was taken from
xla/xla.proto in the pinned checkout; an unknown flag makes the tool exit
before it reads the module.

Fast-math is never drawn together with the interpreter comparison: a
reduction under fast-math can legitimately differ from the evaluator, and
the point of the comparison is a mismatch that cannot be legitimate.

Containment
-----------
Each run gets its own working directory and an address-space ulimit from
execution.mem_limit_mb (a fused module can declare an enormous shape and
the evaluator will try to materialise it). The subprocess runs in its own
session so a timeout kills the whole process group.
"""

import os
import random
import re
import shutil
import subprocess
import time

from core.driver import BaseDriver, ExecutionResult

try:
    from projects.xla.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_xla_analyzer",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


class XLADriver(BaseDriver):

    #: Share of runs that go to run_hlo_module when the module is runnable.
    RUN_SHARE = 0.55
    #: Of those, share that also runs the interpreter and compares.
    COMPARE_SHARE = 0.7

    #: hlo-opt modes and their draw weights.
    #: `passes` runs the named passes on the unverified module (hlo-opt has
    #: no verifier pass to prepend), so its crashes are weaker evidence;
    #: it keeps a smaller share and the analyzer tags its signatures.
    OPT_MODES = [("stage_hlo", 0.40), ("passes", 0.20), ("stage_llvm", 0.40)]

    #: Stages cpu_opt.cc registers (confirmed against `--list-stages` at
    #: init; the fallback list is for a binary that will not answer).
    DEFAULT_STAGES = ["hlo", "llvm-before-optimizations", "llvm", "buffer-assignment"]

    #: DebugOptions drawn per run. (flag, [values]) — one value is picked.
    OPT_FLAGS = [
        ("--xla_backend_optimization_level", ["0", "1", "2", "3"]),
        ("--xla_cpu_multi_thread_eigen", ["true", "false"]),
        ("--xla_llvm_disable_expensive_passes", ["true"]),
        ("--xla_cpu_use_xnnpack", ["true", "false"]),
        ("--xla_cpu_prefer_vector_width", ["128", "256", "512"]),
        ("--xla_cpu_max_isa", ["SSE4_2", "AVX", "AVX2", "AVX512"]),
        ("--xla_cpu_enable_concurrency_optimized_scheduler", ["true"]),
        ("--xla_cpu_experimental_enable_tiling_propagation", ["true"]),
        ("--xla_cpu_use_new_xtile_lowering", ["true"]),
        ("--xla_cpu_copy_insertion_use_region_analysis", ["true"]),
        ("--xla_cpu_emitter_verification_level", ["1", "2"]),
        ("--xla_cpu_parallel_codegen_split_count", ["1", "4"]),
        ("--xla_allow_excess_precision", ["true", "false"]),
        ("--xla_cpu_strict_dot_conv_math", ["true"]),
        ("--xla_llvm_enable_alias_scope_metadata", ["false"]),
        ("--xla_llvm_enable_noalias_metadata", ["false"]),
        ("--xla_llvm_enable_invariant_load_metadata", ["false"]),
    ]
    #: GPU compile-only mode (bin/hlo-opt.gpu, see setup.py): the device
    #: comes from a target-config text proto, so no GPU is needed, and the
    #: GPU backend is the primary target of this adapter (GPU_SHARE). NVIDIA
    #: specs only: the CUDA build segfaults on the AMD/Intel ones (mi200,
    #: mi350, gfx1250, pvc, bmg_g21), a mismatch between the build and the
    #: spec rather than a finding. Autotuning is off because there is no
    #: device to time kernels on.
    #:
    #: Every NVIDIA spec in xla/backends/gpu/target_config/specs is used;
    #: the ones listed here are the known set, others found on disk are
    #: added. Pre-Turing parts (p100 = CC 6.0, v100 = CC 7.0) can only run
    #: the HLO pipeline: the hermetic CUDA 13 ptxas dropped CC < 7.5
    #: ("ptxas too old. Falling back to the driver", and there is no
    #: driver), which fails every stage past `hlo`.
    GPU_SPECS = ["a100_pcie_80", "a100_sxm_40", "a100_sxm_80", "a6000", "b200", "b300",
                 "gb200", "gb300", "h100_pcie", "h100_sxm", "h200", "p100", "rtx6000pro", "v100"]
    GPU_SPEC_HLO_ONLY = {"p100", "v100"}
    GPU_STAGES = ["hlo", "hlo-backend", "llvm", "llvm-before-optimizations",
                  "llvm-after-optimizations", "ptx", "ptx", "buffer-assignment"]
    GPU_SHARE = 0.70
    #: Each base spec also gets variant device descriptions (written once
    #: by _spec_variants into bin/gpu_specs): the same compute capability
    #: with a smaller/larger core count, shared memory, register file and
    #: L2, so the tiling, fusion and scheduling heuristics that read those
    #: numbers see configurations no shipped part has. Drawn about a third
    #: of the time.
    GPU_VARIANT_SHARE = 0.35
    #: All verified accepted by this build's hlo-opt.gpu (an unknown flag
    #: makes it print its usage and exit 1, which would count every draw as
    #: a rejection). Left out on purpose: xla_gpu_enable_libnvjitlink (no
    #: nvJitLink in the hermetic SDK), xla_gpu_experimental_enable_conv_fusion
    #: (needs cuDNN deviceless mode which needs a stream executor).
    GPU_FLAGS = [
        ("--xla_gpu_enable_triton_gemm", ["true", "false"]),
        ("--xla_gpu_triton_gemm_any", ["true"]),
        ("--xla_gpu_enable_latency_hiding_scheduler", ["true", "false"]),
        ("--xla_gpu_enable_while_loop_double_buffering", ["true"]),
        ("--xla_gpu_enable_dynamic_slice_fusion", ["true"]),
        ("--xla_gpu_deterministic_ops", ["true"]),
        ("--xla_gpu_enable_fast_min_max", ["true"]),
        ("--xla_gpu_ftz", ["true"]),
        ("--xla_gpu_enable_cublaslt", ["true"]),
        ("--xla_gpu_experimental_enable_triton_heroless_priority_fusion", ["true"]),
        ("--xla_gpu_exhaustive_tiling_search", ["true"]),
        ("--xla_gpu_enable_analytical_latency_estimator", ["true"]),
        ("--xla_gpu_enable_cub_radix_sort", ["false"]),
        ("--xla_backend_optimization_level", ["0", "1", "2", "3"]),
        # fusion / emitter selection
        ("--xla_gpu_experimental_all_fusions_with_triton", ["true"]),
        ("--xla_gpu_experimental_gemm_fusion_v2", ["true"]),
        ("--xla_gpu_experimental_enable_fusion_block_level_rewriter", ["true"]),
        ("--xla_gpu_experimental_enable_triton_warp_specialization", ["true"]),
        ("--xla_gpu_experimental_enable_tiling_propagation", ["true"]),
        ("--xla_gpu_experimental_enable_same_shape_multi_output_fusion", ["true"]),
        ("--xla_gpu_unsupported_enable_triton_multi_output_fusion", ["true"]),
        ("--xla_gpu_experimental_use_ragged_dot_fusion", ["true"]),
        ("--xla_gpu_experimental_scaled_dot_with_triton", ["true"]),
        ("--xla_gpu_enable_triton_gemm_int4", ["true"]),
        ("--xla_gpu_experimental_enable_subchannel_dequantisation_fusion", ["true"]),
        ("--xla_gpu_use_runtime_fusion", ["true"]),
        ("--xla_gpu_experimental_enable_fusion_autotuner", ["true"]),
        ("--xla_gpu_experimental_enable_raft_for_stable_topk", ["true"]),
        ("--xla_gpu_gemm_rewrite_size_threshold", ["0", "1"]),
        ("--xla_gpu_dot_merger_threshold_mb", ["0", "1", "1024"]),
        ("--xla_gpu_experimental_pack_dot_operands_along_k_dimension", ["false"]),
        ("--xla_gpu_default_to_alg_dot_bf16_bf16_f32", ["true"]),
        # cuDNN / cuBLAS paths (compile-time rewrites; no library call is made)
        ("--xla_gpu_enable_cudnn_fmha", ["true"]),
        ("--xla_gpu_enable_cudnn_layer_norm", ["true"]),
        ("--xla_gpu_cudnn_gemm_fusion_level", ["1", "2", "3"]),
        ("--xla_gpu_enable_cudnn_int8x32_convolution_reordering", ["false"]),
        ("--xla_gpu_force_conv_nhwc", ["true"]),
        ("--xla_gpu_force_conv_nchw", ["true"]),
        # scheduling, streams, command buffers, memory
        ("--xla_gpu_enable_command_buffer", ["", "FUSION", "FUSION,CUBLAS,CUDNN", "FUSION,CUBLAS,CUDNN,CUSTOM_CALL"]),
        ("--xla_gpu_command_buffer_unroll_loops", ["true"]),
        ("--xla_gpu_enable_pdl", ["true"]),
        ("--xla_gpu_enable_host_memory_offloading", ["true"]),
        ("--xla_gpu_enable_allocator_spatial_partitioning", ["true"]),
        ("--xla_gpu_temp_buffer_use_separate_color", ["true"]),
        ("--xla_gpu_redzone_padding_bytes", ["0"]),
        ("--xla_gpu_enable_scatter_determinism_expander", ["false"]),
        ("--xla_gpu_enable_dus_accumulator_zero_init_elimination", ["true"]),
        ("--xla_gpu_experimental_enable_selective_memcpy_overlap", ["true"]),
        ("--xla_gpu_multi_streamed_windowed_einsum", ["true"]),
        ("--xla_gpu_experimental_enable_alltoall_windowed_einsum", ["true"]),
        ("--xla_gpu_enable_reassociation_for_converted_ar", ["true"]),
        ("--xla_gpu_analytical_latency_estimator_options", ["nccl_op_launch_us:100", "nic_speed_gbps:400"]),
        # LLVM / PTX back end
        ("--xla_gpu_experimental_max_unroll_factor", ["1", "2", "8"]),
        ("--xla_gpu_native_emitter_tune_unroll_factor_for_loops", ["true"]),
        ("--xla_gpu_llvm_verification_level", ["1"]),
        ("--xla_gpu_disable_gpuasm_optimizations", ["true"]),
        ("--xla_gpu_generate_line_info", ["true"]),
        ("--xla_gpu_ptx_compiler_extra_flags", ["-O0", "-O1", "--maxrregcount=32", "--allow-expensive-optimizations=true"]),
    ]

    #: Only drawn when no interpreter comparison is made.
    FAST_MATH_FLAGS = [
        ("--xla_cpu_enable_fast_math", ["true"]),
        ("--xla_cpu_fast_math_honor_nans", ["false"]),
        ("--xla_cpu_fast_math_honor_infs", ["false"]),
        ("--xla_cpu_fast_math_honor_division", ["false"]),
        ("--xla_cpu_fast_math_honor_functions", ["false"]),
        ("--xla_cpu_enable_fast_min_max", ["true"]),
        ("--xla_cpu_ftz", ["true"]),
    ]

    def __init__(self, config):
        super().__init__(config)
        self.bin_dir = os.path.join(self.ffl_root, "projects", "xla", "bin")
        self.run_hlo_module = os.path.join(self.bin_dir, "run_hlo_module")
        self.hlo_opt = os.path.join(self.bin_dir, "hlo-opt")
        self.hlo_opt_asan = os.path.join(self.bin_dir, "hlo-opt.asan")
        self.hlo_opt_gpu = os.path.join(self.bin_dir, "hlo-opt.gpu")
        spec_dir = os.path.join(self.ffl_root, "projects", "xla", "xla", "xla",
                                "backends", "gpu", "target_config", "specs")
        self.gpu_specs = [os.path.join(spec_dir, s + ".txtpb") for s in self.GPU_SPECS
                          if os.path.exists(os.path.join(spec_dir, s + ".txtpb"))]
        self.gpu_variants = self._spec_variants(os.path.join(self.bin_dir, "gpu_specs"))
        self.mem_limit_mb = int(config.get("execution", {}).get("mem_limit_mb", 0) or 0)
        self.passes = self._load_passes()
        self.stages = self._load_stages()

    # ── GPU device-description variants ───────────────────────────────

    _SPEC_SCALE = {
        # name: (field multipliers/overrides) — only fields every spec has
        "small": {"core_count": 0.25, "shared_memory_per_block": 16384,
                  "shared_memory_per_block_optin": 49152, "threads_per_block_limit": 512,
                  "registers_per_block_limit": 32768, "l2_cache_size": 0.25,
                  "device_memory_size": 0.25},
        "big": {"core_count": 2.0, "l2_cache_size": 2.0, "device_memory_size": 2.0,
                "memory_bandwidth": 2.0, "registers_per_core_limit": 2.0},
        "narrow": {"core_count": 1, "threads_per_core_limit": 1024,
                   "threads_per_block_limit": 256, "shared_memory_per_block": 8192,
                   "block_dim_limit_y": 1, "block_dim_limit_z": 1},
    }

    def _spec_variants(self, out_dir):
        """Write (once) a small/big/narrow variant of each NVIDIA spec and
        return their paths. Same compute capability; core count, shared
        memory, register file, thread limits and L2 scaled, so the
        cost-model, tiling and scheduling heuristics see device shapes no
        shipped part has. Skipped silently when bin/ is not writable."""
        out = []
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return out
        for spec in self.gpu_specs:
            base = os.path.splitext(os.path.basename(spec))[0]
            try:
                with open(spec) as f:
                    text = f.read()
            except OSError:
                continue
            for vname, rules in self._SPEC_SCALE.items():
                path = os.path.join(out_dir, f"{base}__{vname}.txtpb")
                if not os.path.exists(path):
                    lines = []
                    for line in text.splitlines():
                        m = re.match(r"^(\s*)([a-z_0-9]+):\s*(-?\d+)\s*$", line)
                        if m and m.group(2) in rules:
                            rule = rules[m.group(2)]
                            val = int(int(m.group(3)) * rule) if isinstance(rule, float) else int(rule)
                            line = f"{m.group(1)}{m.group(2)}: {max(val, 1)}"
                        lines.append(line)
                    try:
                        with open(path, "w") as f:
                            f.write("\n".join(lines) + "\n")
                    except OSError:
                        continue
                out.append(path)
        return out

    def _draw_gpu_spec(self):
        """(spec path, base name)."""
        if self.gpu_variants and random.random() < self.GPU_VARIANT_SHARE:
            spec = random.choice(self.gpu_variants)
            return spec, os.path.basename(spec).split("__")[0]
        spec = random.choice(self.gpu_specs)
        return spec, os.path.splitext(os.path.basename(spec))[0]

    # ── tool discovery ────────────────────────────────────────────────

    def _load_passes(self):
        path = os.path.join(self.bin_dir, "passes.txt")
        names = []
        if os.path.exists(path):
            with open(path) as f:
                # one per line, or hlo-opt's own comma-separated line
                names = [n.strip() for l in f for n in l.split(",") if n.strip()]
        if not names and os.path.exists(self.hlo_opt):
            try:
                out = subprocess.run([self.hlo_opt, "--platform=cpu", "--list-passes"],
                                     capture_output=True, text=True, timeout=120)
                names = [n.strip() for l in (out.stdout + out.stderr).splitlines()
                         if l.strip() and " " not in l.strip()
                         for n in l.split(",") if n.strip()]
            except Exception:
                names = []
        return names

    def _load_stages(self):
        if os.path.exists(self.hlo_opt):
            try:
                out = subprocess.run([self.hlo_opt, "--platform=cpu", "--list-stages"],
                                     capture_output=True, text=True, timeout=120)
                names = [l.strip() for l in (out.stdout + out.stderr).splitlines()
                         if l.strip() and " " not in l.strip()]
                if names:
                    return names
            except Exception:
                pass
        return list(self.DEFAULT_STAGES)

    # ── flag drawing ──────────────────────────────────────────────────

    def _draw_flags(self, allow_fast_math):
        pool = list(self.OPT_FLAGS) + (list(self.FAST_MATH_FLAGS) if allow_fast_math else [])
        k = random.choice([0, 0, 1, 1, 2, 3])
        flags = []
        for name, values in random.sample(pool, min(k, len(pool))):
            flags.append(f"{name}={random.choice(values)}")
        return flags

    def _draw_passes(self):
        if not self.passes:
            return None
        n = random.choice([1, 2, 3, 4, 6])
        return ",".join(random.sample(self.passes, min(n, len(self.passes))))

    # ── command building ──────────────────────────────────────────────

    def _runnable(self, facts):
        """Whether run_hlo_module can execute this module on one CPU."""
        return not facts.get("has_collectives") and not facts.get("has_custom_call")

    _RNG_OPS = ("rng", "rng-bit-generator", "rng-get-and-update-state")

    def _comparable(self, facts):
        """Whether a CPU-vs-interpreter comparison can mean anything: a
        module drawing random numbers differs between backends by design
        (first batch: an `rng_uniform` module filed as a mismatch)."""
        return not any(op in self._RNG_OPS for op in facts.get("opcodes", []))

    def _ulimit(self):
        if self.mem_limit_mb > 0:
            return f"ulimit -v {self.mem_limit_mb * 1024}; "
        return ""

    def _build_command(self, seed_file, facts, dryrun=False):
        """(command, tool, compares)"""
        # ulimit -n: absl's symbolizer wants a spare high fd ("Unable to get
        # high fd: limit=1024") or every stack trace is `(unknown)`.
        env = ("ASAN_OPTIONS='abort_on_error=1:detect_leaks=0:symbolize=1' "
               "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' ulimit -n 4096; ")
        if dryrun:
            # Does the CPU backend accept the module at all: parse, verify,
            # run the HLO pipeline, no codegen and no draw.
            cmd = f"{env}{self._ulimit()}{self.hlo_opt} --platform=cpu --stage=hlo --o=/dev/null {seed_file}"
            return cmd, "hlo-opt", False
        if self._runnable(facts) and random.random() < self.RUN_SHARE:
            compare = self._comparable(facts) and random.random() < self.COMPARE_SHARE
            flags = ["--platform=cpu", "--random_init_input_literals=true"]
            flags.append("--reference_platform=interpreter" if compare else "--reference_platform=")
            if random.random() < 0.2:
                flags.append("--use_large_float_range=true")
            if random.random() < 0.15:
                flags.append(f"--iterations={random.choice([2, 3])}")
                if random.random() < 0.5:
                    flags.append("--different_random_seeds=true")
            if random.random() < 0.1:
                flags.append("--isolate_instructions=true")
            flags += self._draw_flags(allow_fast_math=not compare)
            cmd = f"{env}{self._ulimit()}{self.run_hlo_module} {' '.join(flags)} {seed_file}"
            return cmd, "run_hlo_module", compare
        # hlo-opt, GPU backend compile-only when that binary was built
        if os.path.exists(self.hlo_opt_gpu) and self.gpu_specs and random.random() < self.GPU_SHARE:
            spec, base = self._draw_gpu_spec()
            stage = "hlo" if base in self.GPU_SPEC_HLO_ONLY else random.choice(self.GPU_STAGES)
            flags = ["--platform=gpu", f"--stage={stage}",
                     f"--xla_gpu_target_config_filename={spec}",
                     "--xla_gpu_autotune_level=0"]
            k = random.choice([0, 1, 1, 2, 2, 3, 4])
            for name, values in random.sample(self.GPU_FLAGS, k):
                flags.append(f"{name}={random.choice(values)}")
            # bin/gpu_lib holds the hermetic CUDA libraries the binary links
            # (see setup.py _install_gpu_runtime); bin/cuda_sdk holds
            # ptxas/nvlink/libdevice the NVPTX backend looks up at run time.
            # The CUDA libraries reserve well over 8 GB of address space, so
            # the GPU run gets a 16 GB cap regardless of mem_limit_mb.
            flags.append(f"--xla_gpu_cuda_data_dir={os.path.join(self.bin_dir, 'cuda_sdk')}")
            gpu_env = f"LD_LIBRARY_PATH={os.path.join(self.bin_dir, 'gpu_lib')} "
            cap = f"ulimit -v {max(self.mem_limit_mb, 16384) * 1024}; " if self.mem_limit_mb else ""
            cmd = f"{env}{cap}{gpu_env}{self.hlo_opt_gpu} {' '.join(flags)} --o=/dev/null {seed_file}"
            return cmd, "hlo-opt-gpu", False
        binary = self.hlo_opt
        if os.path.exists(self.hlo_opt_asan) and random.random() < 0.5:
            binary = self.hlo_opt_asan
        r = random.random()
        acc = 0.0
        mode = self.OPT_MODES[-1][0]
        for name, w in self.OPT_MODES:
            acc += w
            if r < acc:
                mode = name
                break
        flags = ["--platform=cpu"]
        if mode == "passes" and self.passes:
            flags.append(f"--passes={self._draw_passes()}")
        elif mode == "stage_llvm":
            stages = [s for s in self.stages if s != "hlo"] or self.stages
            flags.append(f"--stage={random.choice(stages)}")
        else:
            flags.append("--stage=hlo")
        flags += self._draw_flags(allow_fast_math=True)
        cmd = f"{env}{self._ulimit()}{binary} {' '.join(flags)} --o=/dev/null {seed_file}"
        return cmd, ("hlo-opt-passes" if mode == "passes" else "hlo-opt"), False

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        seed_file = None
        cmd, tool = "unknown", "hlo-opt"
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.hlo")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            facts = seed.metadata if isinstance(seed.metadata, dict) and "opcodes" in seed.metadata \
                else analyze_seed(seed.content)
            cmd, tool, _compare = self._build_command(
                seed_file, facts, dryrun=getattr(self, "dryrun_mode", False))
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
            if tool == "hlo-opt-gpu" and stderr:
                # Expected on a host without a GPU: the platform probes the
                # driver and logs `failed call to cuInit: INTERNAL: CUDA
                # error ... NO_DEVICE`; the compile-only path continues.
                stderr = "\n".join(l for l in stderr.splitlines()
                                    if "cuInit" not in l and "cuda_status.cc" not in l)
            verdict = classify((stdout or "") + "\n" + (stderr or ""), tool, rc)
            if verdict["is_bug"] and tool == "hlo-opt-passes":
                verdict = self._downgrade_unverified(seed_file, workdir, verdict)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        duration = time.time() - start
        crashed = bool(verdict["is_bug"])
        # The orchestrator's validity rule looks for "error:" in stderr and
        # otherwise takes rc 1 as accepted; XLA reports rejection as a
        # status line (`INVALID_ARGUMENT: ...`) and exits 1 either way, so
        # a rejected or unsupported module is marked here or it would be
        # counted as a valid fused program.
        if verdict["kind"] in ("rejected", "unsupported", "resource", "timeout"):
            stderr = f"error: module {verdict['kind']} by XLA\n" + (stderr or "")
        res = ExecutionResult(rc, stdout, stderr, duration, crashed,
                              verdict["signature"] if crashed else None)
        res.command = cmd
        res.seed_file = seed_file
        res.tool = tool
        res.verdict = verdict["kind"]
        return res

    def _downgrade_unverified(self, seed_file, workdir, verdict):
        """`hlo-opt --passes=...` runs the named passes on the parsed module
        without verifying it first, and a pass CHECK-ing on an ill-typed
        module is not a finding: passes are written against verified HLO.
        On a pass-mode crash, re-run the module through the CPU HLO
        pipeline (whose first pass is the verifier); if the verifier
        rejects it the crash is reclassified as a rejection. Costs one
        extra run per pass-mode crash only."""
        chk = (f"{self._ulimit()}{self.hlo_opt} --platform=cpu "
               f"--stage=hlo --o=/dev/null {seed_file}")
        try:
            _rc, out, err = self._run_command(chk, cwd=workdir)
        except Exception:
            return verdict
        if "hlo verifier" in (out or "") + (err or ""):
            return {"kind": "rejected", "signature": None,
                    "is_bug": False, "is_valid": False}
        return verdict

    # ── oracle hooks (BaseDriver) ─────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        return bool(classify((stdout or "") + "\n" + (stderr or ""),
                             return_code=return_code)["is_bug"])

    def extract_crash_signature(self, stdout, stderr, return_code):
        return classify((stdout or "") + "\n" + (stderr or ""),
                        return_code=return_code)["signature"]
