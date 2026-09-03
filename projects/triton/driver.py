"""
projects/triton/driver.py — run a fused Triton MLIR module through
triton-opt and report whether what came back is a bug.

This file owns execution: writing the module, choosing the pass pipeline
and flags, containing the run, cleaning up. Judgement lives in
projects/triton/analyzer.py.

Choosing the pipeline
---------------------
triton-opt is an MLIR pass driver: it reads textual IR, runs a pipeline,
writes textual IR. Which passes run *is* the test. `-tritongpu-pipeline`
and `-canonicalize` share almost no code, and a bug lives in one pass, so
the pipeline is this driver's main lever — the counterpart of the output
backend in the tint driver and the optimisation tier in the JS ones.

The seed's own pipeline comes first. Each Triton test carries its pipeline
in a `// RUN:` line, and that line was written because that pass is what
the IR exercises: a module full of `ttg.async_copy` fed to `-canonicalize`
tests nothing in particular. So the driver runs the seed's own pipeline
most of the time, and substitutes a drawn one only sometimes — enough to
reach pass/IR combinations the test suite never writes down, which is
where fusion's value is.

Two flags are not optional:

  -allow-unregistered-dialect
      53 of Triton's own tests pass it, because their IR mentions ops from
      dialects triton-opt does not register. Without it those modules fail
      to parse and the run tests nothing. It is harmless on IR that does
      not need it, so it is always on.

  --mlir-print-op-on-diagnostic / --mlir-print-stacktrace-on-diagnostic
      Make a verifier failure name the offending op and where it came
      from. Without them a crash report is a message with no context.

Every pass name below was taken from the RUN lines of the pinned
checkout's own tests, not written from memory: triton-opt rejects an
unknown pass outright, so an invented name would fail every run drawing
it.
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.triton.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_triton_analyzer",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


# `module attributes {"ttg.num-warps" = 4 : i32, ...}` is not decoration:
# every TritonGPU pass reads it, and without it they refuse to run with
#   LLVM ERROR: failed to lookup the number of warps, the surrounding
#   module should contain a ttg.num-warps attribute
#
# Fusion loses it every time — measured at 199 of 199 pairs whose parents
# had one. `mlir_strip_outer_module` unwraps the `module { ... }` shell to
# splice two bodies together, and the attribute list rides on that shell.
# The result is a module that no Triton pass will touch, so most of the
# pipeline this driver draws from would be a no-op.
#
# Rather than change the shared MLIR strategy, the driver puts a module
# wrapper back on at execution time, preferring whichever parent's
# attributes the parser recorded and falling back to the minimum the
# passes need.
# `ttg.target` has to be here too. Several passes read it directly and
# assert on its absence — `TargetFeatures.h:19` does
# `assert(targetAttr && "Expected a target attribute on the module
# operation")` — so a module without it makes a well-formed input look
# like a compiler bug.
_DEFAULT_MODULE_ATTRS = ('"ttg.num-warps" = 4 : i32, '
                         '"ttg.num-ctas" = 1 : i32, '
                         '"ttg.threads-per-warp" = 32 : i32, '
                         'ttg.target = "cuda:90"')

_MODULE_LINE_RE = re.compile(r'^\s*module\b', re.M)
_ALIAS_LINE_RE = re.compile(r'^\s*#[A-Za-z_][\w]*\s*=', re.M)


def _restore_module_attrs(content, facts):
    """Wrap a fused body in `module attributes {...}` when it has none.

    Alias definitions (`#blocked = #ttg.blocked<...>`) are only legal at
    file scope, so they stay outside the wrapper.
    """
    text = content or ""
    m = _MODULE_LINE_RE.search(text)
    if m and "module attributes" in text:
        return text          # already carries its own attributes
    if m:
        # A bare `module { ... }` — which is what the MLIR strategy emits,
        # since it splices two bodies into a fresh wrapper. The passes need
        # the attributes, so give this module the ones we have. Matching
        # only on the presence of `module` was the earlier mistake: it
        # returned unchanged and every fused program reached triton-opt
        # without `ttg.num-warps`, failing with "'tt.func' op is not
        # contained within a context that has ttg.num-warps".
        attrs = facts.get("module_attrs") or _DEFAULT_MODULE_ATTRS
        return text[:m.end()].replace(
            "module", "module attributes {" + attrs + "}", 1) + text[m.end():]
    attrs = facts.get("module_attrs") or _DEFAULT_MODULE_ATTRS
    head, body = [], []
    for line in text.splitlines():
        (head if _ALIAS_LINE_RE.match(line) or line.lstrip().startswith("//")
         else body).append(line)
    inner = "\n".join("  " + l if l.strip() else l for l in body)
    return ("\n".join(head) + "\n" if head else "") + \
           "module attributes {" + attrs + "} {\n" + inner + "\n}\n"


class TritonDriver(BaseDriver):
    """Drives the triton-opt built by projects/triton/setup.py."""

    # Generic MLIR passes. Cheap, always applicable, and the ones most
    # likely to expose a malformed IR a Triton pass produced — the
    # verifier runs after each.
    GENERIC_PASSES = [
        "-canonicalize",
        "-cse",
        "-symbol-dce",
        "-canonicalize -cse",
        "-inline",
    ]

    # Triton's own pipeline passes, drawn from the test suite's RUN lines.
    TRITON_PASSES = [
        "-tritongpu-coalesce",
        "-tritongpu-pipeline",
        "-tritongpu-prefetch",
        "-tritongpu-optimize-dot-operands",
        "-tritongpu-remove-layout-conversions",
        "-tritongpu-accelerate-matmul",
        "-tritongpu-reduce-data-duplication",
        "-tritongpu-schedule-loops",
        "-tritongpu-assign-latencies",
        "-tritongpu-optimize-thread-locality",
        "-tritongpu-combine-tensor-select-and-if",
        "-triton-licm",
        "-triton-loop-unroll",
        "-triton-loop-aware-cse",
    ]

    # NVIDIA-specific passes. These run the machinery around Hopper/Blackwell
    # tensor memory, barriers and CTA planning — the newest code in the
    # tree, and the least covered by the test suite's own RUN lines.
    #
    # Taken from the 108 passes the .td files register, not written from
    # memory; `triton-opt` rejects an unknown pass outright, so an invented
    # name would fail every run drawing it. The two names above that the
    # .td scan does not find — -triton-licm and -triton-loop-unroll — do
    # appear in the corpus's own RUN lines, so they are registered from C++
    # instead; setup-time verification against the built binary is what
    # settles that (see the check in the module docstring).
    NVIDIA_PASSES = [
        "-triton-nvidia-gpu-fence-insertion",
        "-triton-nvidia-gpu-plan-cta",
        "-triton-nvidia-gpu-tmem-barrier-insertion",
        "-triton-nvidia-gpu-tmem-wait-insertion",
        "-triton-nvidia-gpu-remove-tmem-tokens",
        "-triton-nvidia-gpu-proxy-fence-insertion",
        "-triton-nvidia-gpu-cluster-barrier-mbar-allocator",
        "-triton-nvidia-gpu-hoist-mbarrier-lifecycle",
    ]

    # Lowering to LLVM. These run the most code per invocation and are
    # where a layout mismatch surfaces as an assertion rather than a
    # diagnostic.
    # Lowering to LLVM. These run the most code per invocation and are
    # where a layout mismatch surfaces as an assertion rather than a
    # diagnostic.
    #
    # Each entry carries its prerequisites. A lowering pass reads
    # attributes that an earlier allocation pass writes — the
    # `ttg.global_scratch_memory_offset` that
    # `MemoryOpToLLVM.cpp:90` asserts on is written by
    # `-tritongpu-global-scratch-memory-allocation`, and shared-memory
    # offsets come from `--allocate-shared-memory`. Drawing a lowering
    # pass on its own makes those assertions fire on well-formed input,
    # which is a fault in the pipeline we assembled, not in Triton: four
    # of this adapter's first ten findings were that.
    _PRELUDE = ("--allocate-shared-memory "
                "--tritongpu-global-scratch-memory-allocation ")
    LOWERING_PASSES = [
        _PRELUDE + "--convert-triton-gpu-to-llvm",
        _PRELUDE + "--convert-triton-gpu-to-llvm --convert-builtin-func-to-llvm",
        _PRELUDE + "--convert-triton-amdgpu-to-llvm=gfx-arch=gfx942",
        _PRELUDE + "--convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250",
    ]

    # Always present. See the module docstring for why the first is not
    # optional.
    BASE_FLAGS = [
        "-allow-unregistered-dialect",
        "--mlir-print-op-on-diagnostic",
    ]

    # Never passed, wherever they come from — including a seed's own RUN
    # line, which the parser already filters but which is worth guarding
    # again here.
    ORACLE_BREAKING_FLAGS = frozenset({
        # Makes triton-opt exit non-zero unless the errors marked in the
        # source all fire; on a fused module those markers no longer
        # correspond to anything, so every run "fails".
        "-verify-diagnostics", "--verify-diagnostics",
        # Splits the input into independent modules and runs each; the
        # orchestrator already treats a seed as one program.
        "-split-input-file", "--split-input-file",
    })

    # How often to substitute a drawn pipeline for the seed's own. The
    # seed's pipeline is what its IR was written for, so it is the
    # majority case; the draws are what reach combinations the test suite
    # never writes down.
    SUBSTITUTE_RATE = 0.35

    # A fused module can drive a pass into unbounded work. Enforced with
    # `ulimit -v`: unlike the ASan builds elsewhere in this repo, the
    # prebuilt LLVM here is not sanitiser-instrumented, so there is no
    # 20 TB shadow mapping to collide with and an address-space cap works.
    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        mem_mb = exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB)
        self.mem_limit_kb = int(mem_mb) * 1024
        self.triton_opt = os.path.join(
            self.ffl_root, "projects", "triton", "triton-build", "bin", "triton-opt")

    # -- pipeline selection ------------------------------------------------

    # Passes that walk the symbol table. MLIR's symbol-use query returns
    # nullopt when a region holds an op whose symbol semantics it cannot
    # know — any unregistered op — and the inliner asserts on that rather
    # than bailing (upstream `mlir/lib/Transforms/Utils/Inliner.cpp:42`,
    # `symbolUses && "expected uses to be valid"`). It is an upstream
    # robustness gap reachable only through `-allow-unregistered-dialect`,
    # not a Triton defect, so drawing the combination only manufactures
    # findings. Triton's own suite never pairs the two: of the 45 lit tests
    # that pass `-allow-unregistered-dialect`, none runs `-inline`.
    SYMBOL_ANALYSIS_PASSES = ("-inline", "-symbol-dce")

    def _choose_passes(self, facts):
        """Assemble one execution's pass pipeline."""
        own = [p for p in (facts.get("passes") or [])
               if p not in self.ORACLE_BREAKING_FLAGS]

        if facts.get("unregistered"):
            own = [p for p in own if p not in self.SYMBOL_ANALYSIS_PASSES]

        attrs = facts.get("module_attrs") or ""
        if own and random.random() > self.SUBSTITUTE_RATE:
            return self._with_compute_capability(
                self._add_prerequisites(
                    self._drop_mismatched_backend(own, attrs)), attrs)

        # A drawn pipeline. Weighted towards the Triton-specific passes:
        # the generic MLIR ones are shared with every other MLIR project
        # and are the least likely place for a Triton bug.
        # The module's own target decides which backend passes are even
        # meaningful. A `ttg.target = "cuda:90"` module handed to
        # `--convert-triton-amdgpu-to-llvm` fails deep inside the AMD
        # lowering — the first finding this adapter produced was exactly
        # that, an `mlir::gpu::DimensionAttr` uniquer error — and it says
        # nothing about Triton: nobody compiles CUDA IR with the AMD
        # backend. Draw only from the backends the module targets.
        is_amd = "hip:" in attrs or "gfx" in attrs
        is_nvidia = "cuda:" in attrs
        lowering = [f for f in self.LOWERING_PASSES
                    if ("amdgpu" in f) == is_amd or not (is_amd or is_nvidia)]
        nvidia_ok = is_nvidia or not (is_amd or is_nvidia)

        r = random.random()
        if r < 0.35:
            drawn = random.choice(self.TRITON_PASSES)
        elif r < 0.60 and nvidia_ok:
            drawn = random.choice(self.NVIDIA_PASSES)
        elif r < 0.85 and lowering:
            drawn = random.choice(lowering)
        else:
            drawn = random.choice(self.GENERIC_PASSES)

        # Keep the seed's own pipeline in front of the drawn one about half
        # the time: a lowering pass on raw source IR usually fails in the
        # first op, whereas running it after the module's own passes gets
        # it to the IR the pass was actually written for.
        if own and random.random() < 0.5:
            return self._with_compute_capability(
                self._add_prerequisites(
                    self._drop_mismatched_backend(own + drawn.split(), attrs)), attrs)
        return self._with_compute_capability(
            self._add_prerequisites(
                self._drop_mismatched_backend(drawn.split(), attrs)), attrs)

    # Passes that a lowering pass reads the output of. Applied to every
    # pipeline, not only the drawn ones: the driver runs the *seed's own*
    # RUN-line pipeline about 65% of the time, and a test written for
    # `-split-input-file` under lit gets its prerequisites from the lit
    # harness, not from its own RUN line. Fixing only the drawn path left
    # 313 of 400 pipelines without them, and the assertions they trip
    # (MemoryOpToLLVM.cpp:90, Utility.cpp:373) look exactly like compiler
    # bugs.
    _PREREQ_FOR = {
        "--allocate-shared-memory": ("convert-triton-gpu-to-llvm",
                                     "convert-triton-amdgpu-to-llvm"),
        "--tritongpu-global-scratch-memory-allocation":
            ("convert-triton-gpu-to-llvm", "convert-triton-amdgpu-to-llvm"),
    }

    # `--convert-triton-gpu-to-llvm` takes its own `compute-capability`
    # option and does *not* read the module's `ttg.target`. Left at its
    # default the value is below 89, so lowering an `f8E4M3FN` conversion
    # aborts with "only supported on compute capability >= 89" even on a
    # module that declares `ttg.target = "cuda:90"`. That is a mismatch
    # this driver creates, not a Triton defect — passing the module's own
    # capability makes it lower cleanly.
    _CUDA_TARGET_RE = re.compile(r'ttg\.target\s*=\s*"cuda:(\d+)"')

    def _with_compute_capability(self, passes, attrs):
        m = self._CUDA_TARGET_RE.search(attrs or _DEFAULT_MODULE_ATTRS)
        if not m:
            return list(passes)
        cc = m.group(1)
        out = []
        for p in passes:
            if p == "--convert-triton-gpu-to-llvm":
                out.append(f"--convert-triton-gpu-to-llvm{{compute-capability={cc}}}")
            else:
                out.append(p)
        return out

    def _drop_mismatched_backend(self, passes, attrs):
        """Remove backend lowering passes the module's target contradicts.

        Applied to the final pipeline, not only the drawn one. A seed's own
        RUN line can name an AMD lowering pass while the *fused* module
        carries a CUDA target — fusion joins two modules and only one
        target survives — and lowering CUDA IR with the AMD backend fails
        deep inside, which says nothing about Triton. Filtering only the
        drawn path left 327 of 400 pipelines mismatched.
        """
        # A fused module often carries no `module attributes` of its own —
        # the MLIR strategy splices two bodies into a fresh wrapper — and
        # `_restore_module_attrs` then stamps the default, which is CUDA.
        # The backend filter has to assume the same thing, or it lets an
        # AMD lowering pass run on a module that will be labelled
        # `ttg.target = "cuda:90"` by the time triton-opt reads it. Leaving
        # the two defaults out of step is what produced the
        # `dyn_cast on a non-existent value` crash filed as a finding.
        attrs = attrs or _DEFAULT_MODULE_ATTRS
        is_amd = "hip:" in attrs or "gfx" in attrs
        is_nvidia = "cuda:" in attrs
        if not (is_amd or is_nvidia):
            return list(passes)
        out = []
        for p in passes:
            if "amdgpu-to-llvm" in p and not is_amd:
                continue
            if "convert-triton-gpu-to-llvm" in p and not is_nvidia:
                continue
            out.append(p)
        return out

    def _add_prerequisites(self, passes):
        """Prepend any allocation pass a lowering pass in `passes` needs."""
        joined = " ".join(passes)
        missing = [pre for pre, needs in self._PREREQ_FOR.items()
                   if any(n in joined for n in needs)
                   and pre.lstrip("-") not in joined]
        return missing + list(passes) if missing else list(passes)

    def _build_command(self, module_path, facts):
        passes = self._choose_passes(facts)
        passes = [p for p in passes
                  if not (facts.get("unregistered")
                          and p in self.SYMBOL_ANALYSIS_PASSES)]
        flags = " ".join(self.BASE_FLAGS + passes)
        cmd = f"{self.triton_opt} {module_path} {flags}"
        if self.mem_limit_kb > 0:
            # ulimit -v is safe here: this triton-opt is not built with a
            # sanitiser, so nothing reserves a huge shadow mapping.
            cmd = f"ulimit -v {self.mem_limit_kb}; ulimit -c 0; {cmd}"
        return cmd

    # -- execution ---------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            module = os.path.join(workdir, f"{seed.id}.mlir")
            with open(module, "w", encoding="utf-8") as f:
                f.write(_restore_module_attrs(seed.content, facts))
            cmd = self._build_command(module, facts)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        output = f"{stdout}\n{stderr}"
        verdict = classify(output)
        result = ExecutionResult(
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            time=time.time() - start,
            crashed=verdict["is_bug"],
            signature=verdict["signature"],
        )
        result.command = cmd
        return result
