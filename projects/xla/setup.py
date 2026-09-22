"""
projects/xla/setup.py — fetch and build XLA's HLO tools for FusionFuzz.

Called by main.py as setup(project_root) inside the ffe-xla container.
Leaves two binaries under projects/xla/bin/:

  run_hlo_module   parses an HLO text module, compiles it for the CPU
                   backend, runs it on random inputs and (by default)
                   compares the result against XLA's HLO evaluator
                   ("interpreter") — a differential oracle on top of the
                   crash oracle.
  hlo-opt          the HLO pass driver: parse + verify, run a named list
                   of HLO passes or the whole CPU optimisation pipeline,
                   or lower all the way to LLVM IR (`--stage=llvm`).

Why build from source, and how
------------------------------
XLA is only distributed as part of JAX/TensorFlow wheels, with no
assertions and no standalone tools, so the tools are built with Bazel from
a shallow checkout of openxla/xla. The build is configured for the CPU
backend only (this host has no GPU) and with

  -c opt --copt=-UNDEBUG

`-c opt` alone defines NDEBUG, which compiles out every `assert()` and
every absl `DCHECK` — in XLA itself, in the vendored LLVM and in MLIR —
and those are exactly the invariant checks a fuzzer wants live. Undefining
NDEBUG again after the toolchain's own flags keeps them all while keeping
the optimised code paths (a `-c dbg` build of XLA+LLVM would be several
times slower to compile *and* to run). absl `CHECK`, `TF_RET_CHECK` and
`LOG(FATAL)` are unconditional and fire in any build.

Sanitizers are opt-in (FFL_XLA_SANITIZERS=address or address,undefined):
an ASan build of XLA+LLVM roughly doubles the build time and the binaries
are ~3x slower, and run_hlo_module is tagged `noasan` upstream because the
instrumented link exceeds linker limits — so the sanitized build covers
hlo-opt only and lands in bin/hlo-opt.asan beside the plain one.

GPU backend without a GPU (FFL_XLA_GPU=1): XLA's GPU compiler can be run
compile-only on a CPU host — `hlo-opt --platform=gpu
--xla_gpu_target_config_filename=<spec>` takes the device description
from a text proto (xla/backends/gpu/target_config/specs/*.txtpb) and runs
the GPU HLO pipeline, the emitters and NVPTX codegen down to PTX, which is
how XLA's own gpu_opt lit tests run on CPU-only CI. It needs the CUDA
toolchain at *build* time, which Bazel fetches hermetically
(`--config=cuda_clang`, several GB); the result lands in bin/hlo-opt.gpu
and the driver draws it when present.

Toolchain: XLA's own .bazelrc defaults to a hermetic clang that Bazel
downloads (USE_HERMETIC_CC_TOOLCHAIN=1), which is what upstream CI uses
and needs nothing from the image. FFL_XLA_LOCAL_CLANG=1 switches to the
image's clang-18 through configure.py (`--config clang_local`).

Resources: the build is LLVM-sized. FFL_XLA_JOBS caps Bazel's parallelism
(default: min(nproc, 10) — 16 clang processes on the 27 GB box swap).
Bazel's cache lives in the container's home (~/.cache/bazel), so a
rebuild after a container restart is incremental as long as the container
itself survives; the finished binaries are copied out to projects/xla/bin
and survive anything.

Seed corpus
-----------
projects/xla/parser.py collects .hlo files plus the HLO modules embedded
as raw strings in XLA's own C++ tests (`R"(HloModule ...)"`), which is
where most of the test suite lives: ~500 files against ~8,000 inline
modules.
"""

import multiprocessing
import re
import os
import shutil
import subprocess
import sys

XLA_REPO = "https://github.com/openxla/xla.git"
XLA_BRANCH = os.environ.get("FFL_XLA_BRANCH", "main")

TARGETS = {
    "run_hlo_module": "//xla/tools:run_hlo_module",
    "hlo-opt": "//xla/tools:hlo-opt",
}


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:160]}{'...' if len(cmd) > 160 else ''}", flush=True)
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    v = os.environ.get("FFL_XLA_JOBS")
    if v:
        return int(v)
    return max(2, min(multiprocessing.cpu_count(), 10))


def _sanitizers():
    v = os.environ.get("FFL_XLA_SANITIZERS", "").strip().lower()
    return "" if v in ("", "none", "off", "0") else v


def _common_flags():
    flags = [
        "-c opt",
        # keep assert()/DCHECK in XLA, LLVM and MLIR (see module docstring)
        "--copt=-UNDEBUG",
        "--host_copt=-UNDEBUG",
        # symbol names in the crash stack traces the oracle groups by
        "--strip=never",
        f"--jobs={_jobs()}",
        # The CPU-only tag filter configure.py would write; harmless for an
        # explicit target list but keeps GPU-only deps out of the analysis.
        "--build_tag_filters=-gpu",
        # The hermetic clang toolchain turns on clang's layering check
        # (every #include must come from a declared dep); three files in
        # gRPC's channelz fail it and XLA's CI only enables it for its RBE
        # builds. Off: a dependency's include hygiene is not our oracle.
        "--features=-layering_check",
        "--verbose_failures",
    ]
    return " ".join(flags)


def _configure(src_root):
    """Write xla_configure.bazelrc. Hermetic clang by default; the image's
    clang-18 with FFL_XLA_LOCAL_CLANG=1."""
    rc = os.path.join(src_root, "xla_configure.bazelrc")
    if os.path.exists(rc):
        return
    if os.environ.get("FFL_XLA_LOCAL_CLANG") == "1":
        clang = shutil.which("clang") or "/usr/lib/llvm-18/bin/clang"
        _run(f"python3 configure.py --backend=CPU --host_compiler=CLANG "
             f"--clang_path={clang}", cwd=src_root)
    else:
        # configure.py insists on a host compiler choice; with no clang_path
        # it leaves the hermetic toolchain in force and only records the
        # CPU tag filters.
        _run("python3 configure.py --backend=CPU --host_compiler=CLANG "
             "--clang_path=/nonexistent 2>/dev/null || true", cwd=src_root)
        if not os.path.exists(rc):
            with open(rc, "w") as f:
                f.write("build --build_tag_filters=-gpu,-no_oss\n"
                        "test --test_tag_filters=-gpu,-no_oss\n")
        else:
            # Strip the clang_local lines configure.py wrote for the
            # nonexistent path so the hermetic toolchain stays selected.
            with open(rc) as f:
                lines = [l for l in f
                         if "clang_local" not in l and "/nonexistent" not in l]
            with open(rc, "w") as f:
                f.writelines(lines)


def _patch_grpc_module(src_root):
    """gRPC 1.81 (XLA's pin) ships checked-in copies of the well-known-type
    upb headers under src/core/ext/upb-gen/google/protobuf/ *and* generates
    the same headers through upb_c_proto_library. With the hermetic clang
    toolchain the checked-in copies win the include search, and since they
    are not declared inputs of anything Bazel rejects every compile that
    reaches them ("undeclared inclusion(s)", 378 actions across ~50 gRPC
    targets on 2026-09-21). Removing the shadow copies at fetch time makes
    the includes resolve to the generated, declared ones. Done through the
    `patch_cmds` of XLA's own grpc override in MODULE.bazel, so a re-fetch
    reapplies it; idempotent."""
    mod = os.path.join(src_root, "MODULE.bazel")
    with open(mod) as f:
        text = f.read()
    if "FFL: drop shadow upb headers v2" in text:
        return
    text = re.sub(r'    # FFL: drop shadow upb headers.*\n    patch_cmds = \[[^\]]*\],\n', '', text)
    marker = 'module_name = "grpc",\n    patch_strip = 1,\n'
    if marker not in text:
        print("  (MODULE.bazel grpc override not in the expected form; not patched)")
        return
    cmd = ('    # FFL: drop shadow upb headers v2 (see projects/xla/setup.py)\n'
           '    patch_cmds = ["rm -f src/core/ext/upb-gen/google/protobuf/*.upb.h '
           'src/core/ext/upb-gen/google/protobuf/*.upb_minitable.h '
           'src/core/ext/upb-gen/google/protobuf/*.upb_minitable.c '
           'src/core/ext/upbdefs-gen/google/protobuf/*.upbdefs.h '
           'src/core/ext/upbdefs-gen/google/protobuf/*.upbdefs.c"],\n')
    text = text.replace(marker, marker + cmd, 1)
    with open(mod, "w") as f:
        f.write(text)
    print("  MODULE.bazel: grpc override now drops the shadow upb headers")


def _build(src_root, out_dir, targets, extra_flags="", suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    tgt = " ".join(targets.values())
    _run(f"bazel build {_common_flags()} {extra_flags} {tgt}", cwd=src_root)
    for name, label in targets.items():
        rel = label.lstrip("/").replace(":", "/")
        src = os.path.join(src_root, "bazel-bin", rel)
        dst = os.path.join(out_dir, name + suffix)
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        print(f"  installed {dst}")


def _install_gpu_runtime(src_root, bin_dir):
    """hlo-opt.gpu links the CUDA runtime libraries (cudart, cublas, cudnn,
    nvrtc, cufft, cusparse, nvjitlink, nccl) and the driver library
    dynamically; none of them exist on a host without a GPU. Bazel's
    hermetic CUDA redistribution has all of them (including a real
    libcuda.so.1, whose cuInit simply fails with NO_DEVICE — the compile-only
    path never needs a device). Copy exactly the ones `ldd` reports missing
    into bin/gpu_lib, and the pieces the NVPTX backend looks up at run time
    (ptxas, nvlink, nvvm/libdevice) into bin/cuda_sdk, so the fuzzer keeps
    working after the Bazel cache is cleaned. The driver sets
    LD_LIBRARY_PATH=bin/gpu_lib and --xla_gpu_cuda_data_dir=bin/cuda_sdk."""
    import glob
    gpu = os.path.join(bin_dir, "hlo-opt.gpu")
    lib_dir = os.path.join(bin_dir, "gpu_lib")
    sdk_dir = os.path.join(bin_dir, "cuda_sdk")
    os.makedirs(lib_dir, exist_ok=True)
    ext_dirs = glob.glob(os.path.expanduser("~/.cache/bazel/_bazel_*/*/external"))
    out = subprocess.run(["ldd", gpu], capture_output=True, text=True).stdout
    missing = [l.split()[0] for l in out.splitlines() if "not found" in l]
    for lib in missing:
        if os.path.exists(os.path.join(lib_dir, lib)):
            continue
        hits = []
        for e in ext_dirs:
            hits += [h for h in glob.glob(os.path.join(e, "rules_ml_toolchain*", "lib", lib))
                     if "lib32" not in h]
        if not hits:
            print(f"  (no hermetic copy of {lib}; the GPU binary will not run)")
            continue
        shutil.copy2(hits[0], os.path.join(lib_dir, lib))
    print(f"  bin/gpu_lib: {len(os.listdir(lib_dir))} CUDA libraries")
    for e in ext_dirs:
        for nvcc in glob.glob(os.path.join(e, "rules_ml_toolchain*cuda_nvcc")):
            for rel in ("bin/ptxas", "bin/nvlink", "bin/fatbinary",
                        "nvvm/libdevice/libdevice.10.bc"):
                srcf = os.path.join(nvcc, rel)
                if os.path.exists(srcf):
                    dst = os.path.join(sdk_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.copy2(srcf, dst)
    print(f"  bin/cuda_sdk: {'ok' if os.path.exists(os.path.join(sdk_dir, 'nvvm/libdevice/libdevice.10.bc')) else 'libdevice missing'}")


def setup(project_root):
    print(f"Setting up XLA in: {project_root}", flush=True)
    src_root = os.path.join(project_root, "xla")
    bin_dir = os.path.join(project_root, "bin")
    have = all(os.path.exists(os.path.join(bin_dir, n)) for n in TARGETS)
    san = _sanitizers()
    have_san = (not san) or os.path.exists(os.path.join(bin_dir, "hlo-opt.asan"))
    want_gpu = os.environ.get("FFL_XLA_GPU", "") in ("1", "true", "yes")
    have_gpu = (not want_gpu) or os.path.exists(os.path.join(bin_dir, "hlo-opt.gpu"))
    if have and have_san and have_gpu:
        print("XLA tools already built:", ", ".join(sorted(os.listdir(bin_dir))))
        return

    if not os.path.exists(os.path.join(src_root, "WORKSPACE")):
        print("Cloning openxla/xla...", flush=True)
        _run(f"git clone --depth=1 --branch {XLA_BRANCH} {XLA_REPO} {src_root}")

    _configure(src_root)
    _patch_grpc_module(src_root)

    try:
        if not have:
            _build(src_root, bin_dir, TARGETS)
        if san and not have_san:
            # hlo-opt only: run_hlo_module is `noasan` upstream (link limit).
            flags = (f"--copt=-fsanitize={san} --linkopt=-fsanitize={san} "
                     "--copt=-fno-omit-frame-pointer "
                     "--copt=-fsanitize-recover=all")
            _build(src_root, bin_dir, {"hlo-opt": TARGETS["hlo-opt"]},
                   extra_flags=flags, suffix=".asan")
        if want_gpu and not have_gpu:
            # Compile-only GPU backend: hermetic CUDA + clang, no device.
            # `--config=cuda_clang` does not work here: the hermetic clang
            # (18, and 21 alike) cannot compile the CUDA 13.2 headers
            # (texture_fetch_functions.h, math_functions.hpp), and pinning
            # CUDA 12.9 instead runs into NCCL >= 2.28 device headers that
            # clang 18 rejects. XLA's own CUDA 13 configs (pjrt_cuda13) use
            # nvcc for device code with clang as the host compiler, which
            # also keeps the CPU build's object cache valid.
            _build(src_root, bin_dir, {"hlo-opt": TARGETS["hlo-opt"]},
                   # the nvcc config has no capability list of its own and
                   # falls back to one with compute_52, which nvcc 13 rejects
                   extra_flags="--config=cuda_nvcc --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="
                               "sm_75,sm_80,sm_90,sm_100,compute_120",
                   suffix=".gpu")
        if want_gpu and not os.path.isdir(os.path.join(bin_dir, "cuda_sdk")):
            _install_gpu_runtime(src_root, bin_dir)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

    # A smoke test that also records the pass list the driver draws from.
    hlo_opt = os.path.join(bin_dir, "hlo-opt")
    try:
        out = subprocess.run([hlo_opt, "--platform=cpu", "--list-passes"],
                             capture_output=True, text=True, timeout=120)
        # hlo-opt prints the passes as one comma-separated line
        names = [n.strip() for l in (out.stdout + out.stderr).splitlines()
                 if l.strip() and " " not in l.strip()
                 for n in l.split(",") if n.strip()]
        with open(os.path.join(bin_dir, "passes.txt"), "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"  hlo-opt lists {len(names)} passes")
    except Exception as e:  # pragma: no cover
        print(f"  (could not list passes: {e})")
    print("XLA setup complete.")


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
