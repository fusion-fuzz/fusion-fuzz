"""
projects/cuda/setup.py — toolchain discovery and seed collection for the
CUDA adapter. Called by main.py as setup(project_root) inside ffe-cuda.

Nothing is built here. The CUDA toolkit (nvcc, cudafe++, cicc, ptxas,
headers, libdevice) comes with the image (nvidia/cuda:*-devel); the clang
CUDA frontend is projects/clang's from-source build when that tree is
present under the same repository mount — it is a trunk clang configured
with LLVM_ENABLE_ASSERTIONS=ON, which is what makes an ICE detectable —
and the distro clang otherwise (release build, assertions off: crashes
still show, assertions do not). What was found is recorded in
projects/cuda/toolchain.json for the driver.

Seeds (projects/cuda/seeds/):
  clang/     every *.cu under clang/test (SemaCUDA, CodeGenCUDA, Driver,
             CIR, Parser, ...): small, hand-written regression tests, each
             aimed at one frontend path. The lit-only
             `#include "Inputs/cuda.h"` (a stub cuda.h for tests run
             without a toolkit) is removed; with a real toolkit clang's
             own __clang_cuda_runtime_wrapper.h provides the same
             declarations. Tests that include other Inputs/ files are
             excluded by config.yaml.
  samples/   NVIDIA/cuda-samples: real kernels (reductions, scans,
             stencils, cooperative groups, warp intrinsics, texture and
             surface use). Their helper headers live in Common/, which
             the driver passes as -I.
"""

import glob
import json
import os
import re
import shutil
import subprocess

SAMPLES_REPO = "https://github.com/NVIDIA/cuda-samples.git"
LLVM_REPO = "https://github.com/llvm/llvm-project.git"
MAX_SEED_BYTES = 200_000
_INPUTS_CUDA_H_RE = re.compile(r'^\s*#\s*include\s+"(?:\.\./)*(?:[\w-]+/)*Inputs/cuda\.h"\s*\n', re.M)


def _run(cmd, cwd=None):
    print(f"[run] {cmd[:160]}", flush=True)
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd)


def _find_clang(project_root):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(project_root)))
    own = os.path.join(repo, "projects", "clang", "llvm-clang-install", "bin", "clang++")
    if os.access(own, os.X_OK):
        return own, "from-source (projects/clang, assertions on)"
    for name in ("clang++-18", "clang++"):
        p = shutil.which(name)
        if p:
            return p, "distro (assertions off)"
    return None, "missing"


def _cuda_path():
    for p in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if p and os.path.exists(os.path.join(p, "bin", "nvcc")):
            return p
    nvcc = shutil.which("nvcc")
    return os.path.dirname(os.path.dirname(nvcc)) if nvcc else None


def _version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[0][:120]
    except Exception as e:  # pragma: no cover
        return f"? ({e})"


def _clang_test_tree(project_root):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(project_root)))
    shared = os.path.join(repo, "projects", "clang", "llvm-project", "clang", "test")
    if os.path.isdir(shared):
        return shared
    own = os.path.join(project_root, "llvm-project")
    if not os.path.isdir(os.path.join(own, "clang", "test")):
        print("Sparse-cloning llvm-project (clang/test only)...", flush=True)
        shutil.rmtree(own, ignore_errors=True)
        _run(f"git clone --depth=1 --filter=blob:none --sparse {LLVM_REPO} {own}")
        _run("git sparse-checkout set clang/test", cwd=own)
    return os.path.join(own, "clang", "test")


def _collect_clang_tests(test_root, dst):
    n = 0
    for src in glob.glob(os.path.join(test_root, "**", "*.cu"), recursive=True):
        if "/Inputs/" in src or os.path.getsize(src) > MAX_SEED_BYTES:
            continue
        rel = os.path.relpath(src, test_root).replace(os.sep, "__")
        with open(src, encoding="utf-8", errors="replace") as f:
            text = f.read()
        text = _INPUTS_CUDA_H_RE.sub("", text)
        with open(os.path.join(dst, rel), "w", encoding="utf-8") as f:
            f.write(text)
        n += 1
    return n


def _collect_samples(project_root, dst):
    samples = os.path.join(project_root, "cuda-samples")
    if not os.path.isdir(os.path.join(samples, "Common")):
        print("Cloning NVIDIA/cuda-samples...", flush=True)
        shutil.rmtree(samples, ignore_errors=True)
        _run(f"git clone --depth=1 {SAMPLES_REPO} {samples}")
    # the 2026 tree keeps the samples under cpp/<n>_<Topic>/<name>/
    # (older ones under Samples/); take every .cu below either
    n = 0
    for src in glob.glob(os.path.join(samples, "**", "*.cu"), recursive=True):
        if os.path.getsize(src) > MAX_SEED_BYTES:
            continue
        rel = os.path.relpath(src, samples).replace(os.sep, "__")
        shutil.copy2(src, os.path.join(dst, rel))
        n += 1
    # A sample's .cu often includes a sibling header (simpleVote_kernel.cuh,
    # FunctionPointers_kernels.h, ../inc/piestimator.h). The seed tree is
    # flat, so gather every header of every sample into one include
    # directory the driver passes with -I (name collisions are rare and
    # only cost that seed).
    inc = os.path.join(samples, "_ffl_inc")
    os.makedirs(os.path.join(inc, "inc"), exist_ok=True)
    for src in glob.glob(os.path.join(samples, "cpp", "**", "*.[hc]uh"), recursive=True) + \
               glob.glob(os.path.join(samples, "cpp", "**", "*.h"), recursive=True):
        shutil.copy2(src, os.path.join(inc, os.path.basename(src)))
        if os.path.basename(os.path.dirname(src)) == "inc":
            shutil.copy2(src, os.path.join(inc, "inc", os.path.basename(src)))
    return n, [os.path.join(samples, "Common"), inc]


CCCL_REPO = "https://github.com/NVIDIA/cccl.git"


def _collect_cccl(project_root, dst):
    """NVIDIA/cccl (Thrust, CUB, libcu++): the most template-heavy CUDA
    code there is. Thrust's tests compile against the in-repo
    `unittest/unittest.h`; libcu++'s lit tests (.cu, and .cpp meant for
    `-x cuda`) against `test/support`; CUB's need Catch2, which CMake
    fetches, so they are not collected. The repo's own headers go first on
    the include path so tests match the library they were written for."""
    cccl = os.path.join(project_root, "cccl")
    if not os.path.isdir(os.path.join(cccl, "thrust")):
        print("Cloning NVIDIA/cccl...", flush=True)
        shutil.rmtree(cccl, ignore_errors=True)
        _run(f"git clone --depth=1 {CCCL_REPO} {cccl}")
    n = 0
    for base, tag in ((os.path.join(cccl, "thrust", "testing"), "thrust"),
                      (os.path.join(cccl, "libcudacxx", "test", "libcudacxx"), "libcudacxx")):
        for src in glob.glob(os.path.join(base, "**", "*.cu"), recursive=True) + \
                   (glob.glob(os.path.join(base, "**", "*.pass.cpp"), recursive=True) if tag == "libcudacxx" else []):
            if os.path.getsize(src) > MAX_SEED_BYTES or "/support/" in src:
                continue
            rel = tag + "__" + os.path.relpath(src, base).replace(os.sep, "__")
            rel = rel[:-4] + ".cu" if rel.endswith(".cpp") else rel
            shutil.copy2(src, os.path.join(dst, rel))
            n += 1
    incs = [os.path.join(cccl, d) for d in ("libcudacxx/include", "thrust", "cub", "thrust/testing",
                                            "libcudacxx/test/support", "c2h/include")]
    return n, [d for d in incs if os.path.isdir(d)]


def setup(project_root):
    project_root = os.path.abspath(project_root)
    print(f"Setting up CUDA in: {project_root}", flush=True)
    clang, clang_kind = _find_clang(project_root)
    cuda = _cuda_path()
    nvcc = os.path.join(cuda, "bin", "nvcc") if cuda else None
    print(f"  clang++: {clang} [{clang_kind}]")
    print(f"  cuda:    {cuda}  nvcc: {nvcc}")
    if not clang and not nvcc:
        raise RuntimeError("neither a clang++ nor nvcc is available")

    seeds = os.path.join(project_root, "seeds")
    for sub in ("clang", "samples", "cccl"):
        os.makedirs(os.path.join(seeds, sub), exist_ok=True)
    n_clang = _collect_clang_tests(_clang_test_tree(project_root), os.path.join(seeds, "clang"))
    n_samples, inc_dirs = _collect_samples(project_root, os.path.join(seeds, "samples"))
    n_cccl, cccl_incs = _collect_cccl(project_root, os.path.join(seeds, "cccl"))
    inc_dirs = cccl_incs + inc_dirs
    print(f"  seeds: {n_clang} clang tests, {n_samples} cuda-samples, {n_cccl} cccl tests")

    info = {
        "clang": clang, "clang_kind": clang_kind,
        "clang_version": _version([clang, "--version"]) if clang else None,
        "nvcc": nvcc, "nvcc_version": _version([nvcc, "--version"]) if nvcc else None,
        "cuda_path": cuda,
        "include_dirs": [d for d in inc_dirs if os.path.isdir(d)],
    }
    with open(os.path.join(project_root, "toolchain.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Smoke test: one kernel through both compilers, device side only.
    smoke = os.path.join(project_root, "smoke.cu")
    with open(smoke, "w") as f:
        f.write("__global__ void k(float *x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x;"
                " if (i < n) x[i] = __expf(x[i]) * 2.0f; }\n"
                "int main() { float *p = 0; k<<<1, 32>>>(p, 32); return 0; }\n")
    if clang:
        r = subprocess.run([clang, "-x", "cuda", f"--cuda-path={cuda}", "--cuda-gpu-arch=sm_80",
                            "--cuda-device-only", "-Wno-unknown-cuda-version", "-S", "-o", "/dev/null", smoke],
                           capture_output=True, text=True)
        print(f"  clang device-only smoke: rc={r.returncode} {r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ''}")
    if nvcc:
        r = subprocess.run([nvcc, "-arch=sm_80", "--ptx", "-o", "/dev/null", smoke], capture_output=True, text=True)
        print(f"  nvcc --ptx smoke: rc={r.returncode} {r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ''}")
    os.remove(smoke)
    print("CUDA setup complete.")


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
