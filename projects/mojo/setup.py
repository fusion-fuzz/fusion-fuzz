"""
projects/mojo/setup.py — toolchain and seeds for the Mojo adapter.

Called by main.py as setup(project_root) inside ffe-mojo.

The Mojo compiler is open source in modular/modular (Mojo/lib, Mojo/tools;
C++ on MLIR/LLVM, built with Bazel through ./bazelw, which fetches its own
Bazel, a hermetic C++ toolchain and the pinned LLVM source). Two
toolchains are supported and recorded in projects/mojo/toolchain.json:

  from source   `./bazelw build --config=build-mojo -c opt --copt=-UNDEBUG
                //Mojo/tools/mojo:mojo //Mojo/tools/kgen-translate`, i.e.
                the repo's own dbg-mode assertions (LLVM's, MLIR's and the
                compiler's `assert`/`llvm_unreachable`) kept while the
                code is optimised. The build compiles LLVM+MLIR: hours.
                The binaries and the stdlib package it builds land in
                projects/mojo/bin. Set FFL_MOJO_FROM_SOURCE=1 to require it.
  released      the `mojo` wheel from PyPI in projects/mojo/venv (release
                build, no assertions; internal errors still crash with the
                driver's bug-report banner, and -D ASSERT=all runtime
                checks are independent of the compiler's own asserts).
                Used until the source build exists, or always with
                FFL_MOJO_WHEEL=1.

Seeds (projects/mojo/seeds/): Mojo/stdlib/test (330 files, the stdlib's own
tests, `TestSuite` runners), Mojo/test/mojo-parser (514 lit tests, many
negative `expected-error` ones), Mojo/test/mojo-integration (283 programs
run by lit), Mojo/test/mojo-tool (143) and Mojo/examples. Files under a
test's `inputs`/package directories are skipped (they are modules, not
programs).
"""

import glob
import json
import os
import shutil
import subprocess

MODULAR_REPO = "https://github.com/modular/modular.git"
MOJO_WHEEL_VERSION = os.environ.get("FFL_MOJO_WHEEL_VERSION", "mojo==1.1.0")
SEED_DIRS = {
    "stdlib": "Mojo/stdlib/test",
    "parser": "Mojo/test/mojo-parser",
    "integration": "Mojo/test/mojo-integration",
    "tool": "Mojo/test/mojo-tool",
    "examples": "Mojo/examples",
}
SKIP_DIR_PARTS = ("inputs", "test_package", "test-packages", "test_utils", "_plugin", "lsp", "debuginfo")
MAX_SEED_BYTES = 120_000


def _run(cmd, cwd=None):
    print(f"[run] {cmd[:160]}", flush=True)
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd)


def _ensure_repo(project_root):
    src = os.path.join(project_root, "modular")
    if not os.path.isdir(os.path.join(src, "Mojo", "stdlib")):
        print("Cloning modular/modular (shallow)...", flush=True)
        shutil.rmtree(src, ignore_errors=True)
        _run(f"git clone --depth=1 {MODULAR_REPO} {src}")
    return src


def _collect_seeds(src, seeds_dir):
    counts = {}
    for group, rel in SEED_DIRS.items():
        dst = os.path.join(seeds_dir, group)
        os.makedirs(dst, exist_ok=True)
        base = os.path.join(src, rel)
        n = 0
        for f in glob.glob(os.path.join(base, "**", "*.mojo"), recursive=True):
            relp = os.path.relpath(f, base)
            parts = relp.lower().split(os.sep)
            if any(p in SKIP_DIR_PARTS for p in parts[:-1]) or os.path.getsize(f) > MAX_SEED_BYTES:
                continue
            if os.path.basename(f) == "__init__.mojo":
                continue
            shutil.copy2(f, os.path.join(dst, relp.replace(os.sep, "__")))
            n += 1
        counts[group] = n
    return counts


#: Mojo's Bazel build configures the pinned LLVM with three backends
#: (bazel/public-patches/llvm_project.bzl: AArch64, RISCV, X86). Adding
#: NVPTX and AMDGPU through the extension's tag works and was measured on
#: 2026-09-24 (one hour of rebuild, the backends end up in the binary),
#: but it is NOT enough to compile for a GPU, so it is opt-in
#: (FFL_MOJO_GPU_BACKENDS=1) rather than the default:
#:
#:   * the open-source tree registers one target backend, HostBackend
#:     (Mojo/lib/Compiler/ObjectCompiler/Target/Host). `TargetBackendRegistry
#:     ::lookup` therefore fails for `nvptx64-nvidia-cuda` with "target ...
#:     is not supported by this build" whichever LLVM backends are linked,
#:     both through `--target-accelerator` and through the stdlib's
#:     `compile_info[f, target=get_gpu_target["sm_90"]()]`.
#:   * even with one registered, `requireMaxForAccelerator`
#:     (Mojo/lib/Target/TargetTraits.cpp:26) refuses an accelerator target
#:     unless MAX is installed: "please install MAX for accelerator support".
#:
#: The released wheel's compiler does have the GPU backends (it lists the
#: NVIDIA and AMD architectures), but its stdlib ships only `std.mojoc`
#: and the GPU API now lives in the `max` package, which is not
#: distributed in a form this compiler can import ("many stdlib items
#: recently moved to the `max` package"). So Mojo GPU code generation is
#: not reachable with what Modular publishes today.
_GPU_BACKENDS_TAG = 'llvm_configure.configure(extra_targets = ["NVPTX", "AMDGPU"])'


def _enable_gpu_backends(src):
    """Ask the Bazel LLVM configuration for the NVPTX and AMDGPU backends.
    Idempotent: returns True when it changed MODULE.bazel, which means the
    LLVM libraries have to be rebuilt."""
    mod = os.path.join(src, "MODULE.bazel")
    try:
        with open(mod) as f:
            text = f.read()
    except OSError:
        return False
    if _GPU_BACKENDS_TAG in text:
        return False
    marker = 'use_repo(llvm_configure, "llvm-project")'
    if marker not in text:
        print("  MODULE.bazel does not configure llvm_configure; GPU backends not enabled")
        return False
    text = text.replace(marker, marker + "\n" + _GPU_BACKENDS_TAG, 1)
    with open(mod, "w") as f:
        f.write(text)
    print("  MODULE.bazel: requested LLVM backends NVPTX and AMDGPU")
    return True


def _has_gpu_backends(mojo, home):
    """Does this compiler binary have the NVPTX/AMDGPU LLVM backends? Asked
    of the binary itself, so a tree that was built before the backends were
    requested is rebuilt rather than silently kept."""
    env = dict(os.environ, MODULAR_MOJO_MAX_PACKAGE_ROOT=home,
               MODULAR_MOJO_MAX_IMPORT_PATH=os.path.join(home, "lib", "mojo"),
               MODULAR_CRASH_REPORTING_ENABLED="false", MODULAR_TELEMETRY_ENABLED="false")
    try:
        out = subprocess.run([mojo, "build", "--print-supported-targets"],
                             capture_output=True, text=True, env=env, timeout=120)
    except Exception:
        return False
    text = out.stdout + out.stderr
    return "nvptx64" in text and "amdgcn" in text


def _from_source(project_root, src):
    """Build the compiler with Bazel and lay it out the way the released
    wheel is (bin/mojo, bin/kgen-translate, bin/lld, lib/*.so,
    lib/mojo/std.mojoc) under projects/mojo/bin/home. The driver reads its
    paths from the `mojo-max` config section, every key of which can be
    given by environment: MODULAR_MOJO_MAX_PACKAGE_ROOT for the home and
    MODULAR_MOJO_MAX_IMPORT_PATH for the compiled stdlib. Returns the
    toolchain dict, or None when nothing is built and
    FFL_MOJO_FROM_SOURCE is unset."""
    home = os.path.join(project_root, "bin", "home")
    mojo = os.path.join(home, "bin", "mojo")
    bazel_bin = os.path.join(src, "bazel-bin")
    built = os.path.exists(os.path.join(bazel_bin, "Mojo", "tools", "mojo", "mojo"))
    # GPU code generation needs LLVM's NVPTX/AMDGPU backends, which the
    # repo's Bazel configuration leaves out; asking for them changes
    # MODULE.bazel and invalidates the LLVM libraries.
    want_gpu = os.environ.get("FFL_MOJO_GPU_BACKENDS") == "1"
    gpu_changed = _enable_gpu_backends(src) if want_gpu else False
    stale_gpu = (want_gpu and not gpu_changed and os.path.exists(mojo)
                 and not _has_gpu_backends(mojo, home))
    if stale_gpu:
        print("  compiler has no GPU backends; rebuilding", flush=True)
    if (not built or gpu_changed or stale_gpu) and (
            os.environ.get("FFL_MOJO_FROM_SOURCE") == "1" or gpu_changed or stale_gpu):
        jobs = os.environ.get("FFL_MOJO_JOBS", "12")
        _run(f"./bazelw build --config=build-mojo -c opt --copt=-UNDEBUG --host_copt=-UNDEBUG "
             f"--jobs={jobs} //Mojo/tools/mojo:mojo //Mojo/tools/kgen-translate //Mojo/stdlib/std:std", cwd=src)
        built = True
        if os.path.exists(mojo):
            os.remove(mojo)          # force the layout copy below to refresh
    if built and not os.path.exists(mojo):
        std = glob.glob(os.path.join(src, "bazel-out", "*", "bin", "Mojo", "stdlib", "std", "std.mojoc"))
        if not std:
            _run("./bazelw build --config=build-mojo -c opt --copt=-UNDEBUG --host_copt=-UNDEBUG "
                 "//Mojo/stdlib/std:std", cwd=src)
            std = glob.glob(os.path.join(src, "bazel-out", "*", "bin", "Mojo", "stdlib", "std", "std.mojoc"))
        os.makedirs(os.path.join(home, "bin"), exist_ok=True)
        os.makedirs(os.path.join(home, "lib", "mojo"), exist_ok=True)
        def _install(src_path, dst_path):
            # Bazel's outputs are read-only, and so is a copy of one: a
            # rebuild has to replace the previous install, not write into
            # it (PermissionError otherwise).
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy2(src_path, dst_path)
            os.chmod(dst_path, 0o755)

        for rel, name in (("Mojo/tools/mojo/mojo", "mojo"),
                          ("Mojo/tools/kgen-translate/kgen-translate", "kgen-translate")):
            _install(os.path.join(bazel_bin, rel), os.path.join(home, "bin", name))
        _install(os.path.join(bazel_bin, "Mojo", "libKGENCompilerRTShared.so"),
                 os.path.join(home, "lib", "libKGENCompilerRTShared.so"))
        for so in glob.glob(os.path.join(bazel_bin, "_solib_k8", "*", "*.so")):
            _install(so, os.path.join(home, "lib", os.path.basename(so)))
        _install(std[0], os.path.join(home, "lib", "mojo", "std.mojoc"))
        # the tree builds no linker; the released wheel's lld links the
        # executables (the linker is not what is under test)
        wheel_lld = glob.glob(os.path.join(project_root, "venv", "lib", "python3*", "site-packages",
                                           "modular", "bin", "lld"))
        lld = wheel_lld[0] if wheel_lld else shutil.which("ld.lld") or shutil.which("lld")
        if lld:
            shutil.copy2(lld, os.path.join(home, "bin", "lld"))
    if not os.path.exists(mojo):
        return None
    kt = os.path.join(home, "bin", "kgen-translate")
    return {"kind": "from-source (assertions on)", "mojo": mojo,
            "kgen_translate": kt if os.path.exists(kt) else None,
            "search_paths": [os.path.join(home, "lib", "mojo")],
            "env": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": home,
                    "MODULAR_MOJO_MAX_IMPORT_PATH": os.path.join(home, "lib", "mojo")}}


def _wheel(project_root):
    venv = os.path.join(project_root, "venv")
    mojo = os.path.join(venv, "bin", "mojo")
    if not os.path.exists(mojo):
        print(f"Installing the released toolchain ({MOJO_WHEEL_VERSION}) into venv...", flush=True)
        _run(f"python3 -m venv {venv} && {venv}/bin/pip install -q {MOJO_WHEEL_VERSION}")
    return {"kind": f"released wheel ({MOJO_WHEEL_VERSION}, no assertions)", "mojo": mojo,
            "kgen_translate": None, "search_paths": [], "env": {}}


def setup(project_root):
    project_root = os.path.abspath(project_root)
    print(f"Setting up Mojo in: {project_root}", flush=True)
    src = _ensure_repo(project_root)
    counts = _collect_seeds(src, os.path.join(project_root, "seeds"))
    print("  seeds:", ", ".join(f"{k}={v}" for k, v in counts.items()))

    tc = None if os.environ.get("FFL_MOJO_WHEEL") == "1" else _from_source(project_root, src)
    if tc is None:
        tc = _wheel(project_root)
    tc["include_dirs"] = [os.path.join(src, "Mojo", "stdlib", "test"),
                          os.path.join(src, "Mojo", "test", "test-packages")]
    tc["env"].update({"MODULAR_CRASH_REPORTING_ENABLED": "false", "MODULAR_TELEMETRY_ENABLED": "false"})
    with open(os.path.join(project_root, "toolchain.json"), "w") as f:
        json.dump(tc, f, indent=2)
    print(f"  toolchain: {tc['kind']}: {tc['mojo']}")

    smoke = os.path.join(project_root, "smoke.mojo")
    with open(smoke, "w") as f:
        f.write("def main():\n    var xs = List[Int]()\n    for i in range(5):\n        xs.append(i * i)\n"
                "    print(xs[4])\n")
    env = dict(os.environ, **tc["env"])
    r = subprocess.run([tc["mojo"], "run", "-DASSERT=all", smoke], capture_output=True, text=True, env=env)
    print(f"  smoke: rc={r.returncode} out={r.stdout.strip()[:40]!r} {r.stderr.strip().splitlines()[-1][:100] if r.stderr.strip() else ''}")
    os.remove(smoke)
    print("Mojo setup complete.")


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
