"""
projects/tvm/setup.py — build TVM from source and collect TVMScript seeds.

Called by main.py as setup(project_root) inside ffe-tvm.

GPU backends are the primary fuzz target and are exercised compile-only:
no GPU is present and nothing is ever launched, so the device-side
codegens are reached through the source/IR they emit.

  * cuda / opencl / metal / webgpu: source-level codegens. They need no
    toolkit; on a USE_CUDA=OFF build the generated source comes back
    through the fallback module (`inspect_source`), and the driver hands
    the CUDA source to the image's nvcc (`--ptx`) as a second stage.
  * nvptx and rocm: LLVM's NVPTX and AMDGPU back ends. Both need device
    bitcode, which the image provides (CUDA_PATH for libdevice, ROCM_PATH
    for rocm-device-libs) — again no driver and no device.
  * vulkan: USE_VULKAN=ON builds TVM's SPIR-V codegen against the Vulkan
    headers and SPIRV-Tools in the image (the validator runs on the
    generated module; the Vulkan loader is never asked for a device).

Build: CMake + Ninja from a shallow apache/tvm checkout with its tvm-ffi
submodule (dlpack, libbacktrace). Oracles kept live:

  * TVM's ICHECK/CHECK are unconditional; `assert()` in TVM and in the
    linked LLVM would be compiled out by CMake's RelWithDebInfo (-DNDEBUG),
    so the flags are set to "-O2 -g" without it.
  * LLVM: USE_LLVM points at projects/clang's from-source LLVM build tree
    when present (same repository mount; `llvm-config --assertion-mode`
    is ON), so LLVM's own assertions in the codegen fire. Fallback: a
    distro llvm-config, assertions off.
  * libbacktrace for symbolised stacks in the InternalError message.

Seeds: projects/tvm/extract_seeds.py pulls every `@I.ir_module` class
(and standalone prim_func/relax function) out of tests/python into a
standalone file under projects/tvm/seeds/tests.
"""

import glob
import os
import shutil
import subprocess
import sys

TVM_REPO = "https://github.com/apache/tvm.git"


def _run(cmd, cwd=None):
    print(f"[run] {cmd[:160]}", flush=True)
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd)


def _llvm_config(project_root):
    """FFL_TVM_LLVM_CONFIG, else the newest distro llvm-config (18 from
    apt.llvm.org in the image), else projects/clang's assertion-on trunk
    LLVM. The trunk one is preferred for its assertions but TVM's LLVM
    backend follows released LLVM APIs and did not compile against LLVM 24
    (codegen_nvptx.cc, codegen_blob.cc); set FFL_TVM_LLVM_CONFIG to it to
    retry after a TVM bump."""
    env = os.environ.get("FFL_TVM_LLVM_CONFIG")
    if env and os.access(env, os.X_OK):
        return env, "FFL_TVM_LLVM_CONFIG"
    for name in ("llvm-config-18", "llvm-config-17", "llvm-config-20", "llvm-config-19", "llvm-config"):
        p = shutil.which(name) or (os.path.join("/usr/lib", name.replace("llvm-config-", "llvm-"), "bin", "llvm-config")
                                   if name != "llvm-config" else None)
        if p and os.access(p, os.X_OK):
            return p, "distro (assertions off)"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(project_root)))
    own = os.path.join(repo, "projects", "clang", "llvm-clang-build", "bin", "llvm-config")
    if os.access(own, os.X_OK):
        return own, "projects/clang build tree (assertions on)"
    return None, "missing"


def _jobs():
    v = os.environ.get("FFL_TVM_JOBS")
    return int(v) if v and v.isdigit() else 12


def setup(project_root):
    project_root = os.path.abspath(project_root)
    print(f"Setting up TVM in: {project_root}", flush=True)
    src = os.path.join(project_root, "tvm")
    if not os.path.isdir(os.path.join(src, "src")):
        print("Cloning apache/tvm (shallow)...", flush=True)
        shutil.rmtree(src, ignore_errors=True)
        _run(f"git clone --depth=1 {TVM_REPO} {src}")
    if not os.path.isdir(os.path.join(src, "3rdparty", "tvm-ffi", "3rdparty", "dlpack", "include")):
        _run("git submodule update --init --depth 1 3rdparty/tvm-ffi", cwd=src)
        _run("git submodule update --init --depth 1", cwd=os.path.join(src, "3rdparty", "tvm-ffi"))

    build = os.path.join(src, "build")
    lib = os.path.join(build, "lib", "libtvm_compiler.so")
    # A build configured before the GPU codegens were wanted has to be
    # reconfigured: the marker below is written with the config.
    cfg = os.path.join(build, "config.cmake")
    if os.path.exists(lib) and not (os.path.exists(cfg)
                                    and "FFL_GPU_CODEGEN" in open(cfg).read()):
        print("  reconfiguring for the GPU codegens (USE_VULKAN)", flush=True)
        os.remove(lib)
    if not os.path.exists(lib):
        llvm, kind = _llvm_config(project_root)
        print(f"  llvm-config: {llvm} [{kind}]")
        if not llvm:
            raise RuntimeError("no llvm-config; install llvm-dev or build projects/clang")
        os.makedirs(build, exist_ok=True)
        shutil.copy2(os.path.join(src, "cmake", "config.cmake"), os.path.join(build, "config.cmake"))
        with open(os.path.join(build, "config.cmake"), "a") as f:
            f.write(f'\nset(USE_LLVM "{llvm} --link-static")\nset(USE_LIBBACKTRACE ON)\n'
                    'set(USE_RPC OFF)\nset(USE_RANDOM ON)\nset(USE_SORT ON)\n')
            # FFL_GPU_CODEGEN: marker read above, and the GPU codegens.
            # Vulkan is the one that needs a build option — the SPIR-V
            # codegen is only compiled in with USE_VULKAN; cuda, nvptx,
            # rocm, opencl, metal and webgpu are always built and are
            # reached compile-only (see the module docstring).
            gpu = "# FFL_GPU_CODEGEN\n"
            if os.path.exists("/usr/include/vulkan/vulkan.h") and \
                    glob.glob("/usr/lib/*/libSPIRV-Tools*"):
                gpu += 'set(USE_VULKAN ON)\nset(USE_KHRONOS_SPIRV /usr)\n'
            f.write(gpu)
        # TVM's sources need a C++17 compiler newer than clang 14 (Ubuntu
        # 22.04's): `reference to local binding declared in enclosing
        # function` in relax/ir/binding_rewrite.cc. GCC 11 compiles it, and
        # links fine against the clang-built static LLVM (same libstdc++).
        cc = shutil.which("gcc") or shutil.which("clang") or "cc"
        cxx = shutil.which("g++") or shutil.which("clang++") or "c++"
        _run(f"cmake -S {src} -B {build} -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo "
             f"-DCMAKE_C_COMPILER={cc} -DCMAKE_CXX_COMPILER={cxx} "
             f'-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g" -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g" '
             f"-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld", cwd=src)
        _run(f"cmake --build {build} --parallel {_jobs()}")
    if not os.path.exists(lib):
        raise RuntimeError("TVM build failed: build/lib/libtvm_compiler.so not found")
    # tvm_ffi's Python package needs its compiled `core` extension, which
    # TVM's CMake build does not produce: install the submodule's package
    # (scikit-build-core, builds the extension and bundles libtvm_ffi.so).
    try:
        import tvm_ffi.core  # noqa: F401
    except Exception:
        _run(f"{sys.executable} -m pip install --user -q {os.path.join(src, '3rdparty', 'tvm-ffi')}")

    seeds = os.path.join(project_root, "seeds", "tests")
    if not os.path.isdir(seeds) or not os.listdir(seeds):
        sys.path.insert(0, project_root)
        import extract_seeds
        n = extract_seeds.main(os.path.join(src, "tests", "python"), seeds)
        print(f"  seeds: {n} TVMScript modules extracted")

    import site
    pp = os.path.join(src, "python") + ":" + site.getusersitepackages()
    env = dict(os.environ, PYTHONPATH=pp, TVM_LIBRARY_PATH=os.path.join(build, "lib"))
    smoke = os.path.join(project_root, "smoke.py")
    with open(smoke, "w") as f:
        f.write('import tvm\nfrom tvm.script import ir as I\nfrom tvm.script import tirx as T\n'
                '@I.ir_module\nclass Module:\n    @T.prim_func\n'
                '    def main(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):\n'
                '        for i in range(16):\n            B[i] = A[i] * 2.0\n')
    r = subprocess.run([sys.executable, os.path.join(project_root, "runner.py"), smoke, "--mode", "diff"],
                       capture_output=True, text=True, env=env, cwd=project_root)
    print(f"  smoke: rc={r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1][:120] if (r.stdout + r.stderr).strip() else ''}")
    os.remove(smoke)
    probe = os.path.join(project_root, "gpu_probe.py")
    with open(probe, "w") as f:
        f.write("import tvm, tvm_ffi\n"
                "for k in ('cuda','nvptx','rocm','vulkan','opencl','metal','webgpu'):\n"
                "    f = tvm_ffi.get_global_func('target.build.' + k, allow_missing=True)\n"
                "    print('  codegen', k, 'yes' if f is not None else 'no')\n")
    r = subprocess.run([sys.executable, probe], capture_output=True, text=True, env=env, cwd=project_root)
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "codegen" in l))
    os.remove(probe)
    print("TVM setup complete.")


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
