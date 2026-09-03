"""
projects/triton/setup.py — fetch and build triton-opt for FusionFuzz.

Called by main.py as setup(project_root). Leaves an MLIR opt driver at
projects/triton/triton-build/bin/triton-opt plus a seed corpus built from
Triton's own .mlir test suite.

Why this is cheap compared to the clang adapter
-----------------------------------------------
Triton pins an LLVM revision in cmake/llvm-info.json and publishes a
prebuilt tarball of it, which its own build downloads rather than
compiling. We do the same. That skips the multi-hour LLVM build the clang
adapter needs, and — this is the part that matters for fuzzing — the
published build is configured with LLVM_ENABLE_ASSERTIONS=ON
(scripts/build-llvm-project.sh in the Triton tree), so MLIR's internal
assertions are live and an ICE is detectable. Without assertions most of
what this fuzzer looks for would be silent.

Two CMake variables have to be supplied that a normal `pip install
triton` would fill in for us:

  TRITON_CACHE_PATH   CMakeLists.txt hard-fails without it. Its message
                      says the value is "derivable from HOME", but the
                      derivation lives in Triton's *Python* setup.py, not
                      in CMake — invoking cmake directly skips it, so the
                      cache directory has to be named here.
  CMAKE_BUILD_TYPE    RelWithDebInfo, with NDEBUG removed from its flags.
                      This is the single most important setting here.
                      `Release` implies `-O3 -DNDEBUG`, and NDEBUG compiles
                      out every `assert()` — all 658 of them in Triton's
                      own lib/. The prebuilt LLVM keeps its assertions
                      either way (it is configured with
                      LLVM_ENABLE_ASSERTIONS=ON), so a Release build still
                      catches MLIR-level failures, but Triton's own
                      invariants become invisible and most of what this
                      fuzzer looks for is silently skipped. Verified by
                      `strings triton-opt | grep Assertion` coming back
                      empty on a Release build.
                      `-g1` keeps stack frames symbolised without the size
                      of full debug info.

  TRITON_OFFLINE_BUILD
                      Stops the build fetching NVIDIA/AMD toolchain
                      binaries (ptxas, cuobjdump) at configure time. This
                      adapter never emits PTX — it runs MLIR passes — and
                      there is no GPU here to run the result on.

Only `triton-opt` is built. It is a standard MLIR opt driver: it reads
textual IR, runs a pass pipeline, writes textual IR. No GPU, no CUDA
runtime, no Python extension module.

Seed corpus
-----------
Triton's test/ tree, ~297 .mlir files. Small next to the GCC adapter's
82k, but dense: each one is a hand-written regression test for a specific
pass, and they carry the two things a Triton seed cannot do without —
the pass pipeline in their RUN line and the layout attribute aliases the
IR references.
"""

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

TRITON_REPO = "https://github.com/triton-lang/triton.git"
TRITON_BRANCH = os.environ.get("FFL_TRITON_BRANCH", "main")

# Where Triton publishes the prebuilt LLVM its own build consumes.
LLVM_BASE_URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds"
JSON_URL_TMPL = "https://github.com/nlohmann/json/releases/download/{v}/include.zip"

# triton-opt links both codegen backends unconditionally
# (bin/RegisterTritonDialects.h includes amd/ and nvidia/ headers), so
# neither can be dropped to save build time.
CODEGEN_BACKENDS = "nvidia;amd"


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_TRITON_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    # Linking triton-opt pulls in most of MLIR; leave headroom so a
    # parallel link step does not get OOM-killed.
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


def _download(url, dest, sha256=None):
    if os.path.exists(dest):
        print(f"  (cached) {os.path.basename(dest)}")
        return dest
    print(f"Downloading {url} ...")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if sha256:
        digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if digest != sha256:
            os.unlink(tmp)
            raise RuntimeError(f"sha256 mismatch for {url}: {digest} != {sha256}")
        print("  sha256 ok")
    os.rename(tmp, dest)
    return dest


def _fetch_llvm(project_root, src_root):
    """Download and unpack the LLVM build Triton pins.

    The revision, build number and per-platform checksums all come from the
    checked-out tree's cmake/llvm-info.json, so this follows whatever the
    cloned Triton expects rather than a version hardcoded here — the two
    must agree or the build fails on ABI mismatch.
    """
    info_path = os.path.join(src_root, "cmake", "llvm-info.json")
    with open(info_path) as f:
        info = json.load(f)
    rev = info["llvm_hash"][:8]
    suffix = "ubuntu-x64"          # this adapter's image is ubuntu:22.04/x86-64
    name = f"llvm-{rev}-{suffix}-{info['build_number']}.tar.gz"
    sha = info.get("sha256sum", {}).get(suffix)

    deps = os.path.join(project_root, "deps")
    os.makedirs(deps, exist_ok=True)
    out_dir = os.path.join(deps, f"llvm-{rev}-{suffix}")
    if os.path.isdir(out_dir):
        print(f"LLVM already unpacked at {out_dir}")
        return out_dir

    tarball = _download(f"{LLVM_BASE_URL}/{name}", os.path.join(deps, name), sha)
    print(f"Unpacking {name} (~375 MB compressed) ...")
    with tarfile.open(tarball) as tf:
        tf.extractall(deps)
    # The archive's top-level directory is the name without .tar.gz.
    unpacked = os.path.join(deps, name[: -len(".tar.gz")])
    if os.path.isdir(unpacked) and unpacked != out_dir:
        shutil.move(unpacked, out_dir)
    if not os.path.isdir(out_dir):
        raise RuntimeError(f"LLVM unpack produced no {out_dir}")
    return out_dir


def _fetch_json(project_root, src_root):
    version = open(os.path.join(src_root, "cmake", "json-version.txt")).read().strip()
    deps = os.path.join(project_root, "deps")
    os.makedirs(deps, exist_ok=True)
    out_dir = os.path.join(deps, f"json-{version}")
    if os.path.isdir(os.path.join(out_dir, "include")):
        return out_dir
    archive = _download(JSON_URL_TMPL.format(v=version),
                        os.path.join(deps, f"json-{version}-include.zip"))
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# Test directories worth taking. Left out:
#   Tools/, Plugins/  — exercise triton-opt's own CLI, not the dialects.
#   LLVMIR/           — plain LLVM IR, not Triton IR.
# Proton/NVWS/Gluon are kept: they are Triton dialects triton-opt registers.
CORPUS_DIRS = ["TritonGPU", "Conversion", "TritonNvidiaGPU", "Triton",
               "Analysis", "Proton", "NVWS", "Gluon", "Hopper"]

_MAX_SEED_BYTES = 256 * 1024


_SECTION_SEP_RE = re.compile(r"^//\s*-----\s*$", re.M)
_RUN_LINE_RE = re.compile(r"^//\s*RUN:.*$", re.M)


def _split_sections(content):
    """Split a `-split-input-file` test into its independent modules.

    Returns [content] unchanged when there are no `// -----` separators.
    Each section keeps the file's RUN lines, which describe the pipeline
    every section is run under.
    """
    if not _SECTION_SEP_RE.search(content):
        return [content]
    run_lines = _RUN_LINE_RE.findall(content)
    header = ("\n".join(run_lines) + "\n") if run_lines else ""
    out = []
    for part in _SECTION_SEP_RE.split(content):
        part = _RUN_LINE_RE.sub("", part).strip("\n")
        if part.strip():
            out.append(header + part + "\n")
    return out or [content]


def _collect_mlir_seeds(project_root, src_root):
    """Copy Triton's .mlir tests into projects/triton/seeds/.

    Files are copied whole, RUN lines included: projects/triton/parser.py
    needs them. A Triton test without its pass pipeline is not a reduced
    test case, it is a different test — `-tritongpu-pipeline=num-stages=3`
    is the thing being exercised, and running that IR through some other
    pipeline exercises nothing in particular.
    """
    seeds = os.path.join(project_root, "seeds")
    test_root = os.path.join(src_root, "test")
    if not os.path.isdir(test_root):
        print(f"Warning: no test tree at {test_root}; corpus will be empty.")
        return 0

    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(test_root, rel)
        if not os.path.isdir(root):
            print(f"  (skipping {rel}: not present)")
            continue
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".mlir"):
                    continue
                src = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(src) > _MAX_SEED_BYTES:
                        skipped += 1
                        continue
                    content = open(src, encoding="utf-8", errors="ignore").read()
                except OSError:
                    skipped += 1
                    continue
                if not content.strip():
                    skipped += 1
                    continue
                # Tests whose declared correct outcome is a failure. Like
                # clang's -verify tests, the diagnostic *is* the expected
                # result: `-verify-diagnostics` makes triton-opt exit
                # non-zero unless the errors marked with `expected-error`
                # all fire. Fusing them produces a program whose success
                # criterion no longer exists, and every execution on one
                # reports a failure that is the test working as designed.
                if "expected-error" in content or "verify-diagnostics" in content:
                    skipped += 1
                    continue
                # `split-file` tests are a bundle of several files in one,
                # separated by `//--- name` markers, and the RUN line feeds
                # triton-opt one member. As a single seed the bundle is not
                # valid IR.
                if "RUN: split-file" in content:
                    skipped += 1
                    continue
                # Same-named files recur across directories (many are
                # pr*.mlir), and seed identifiers must be unique or they
                # overwrite each other in corpus.db.
                flat = os.path.relpath(src, test_root).replace(os.sep, "_")

                # Split `// -----` sections into separate seeds.
                #
                # A `-split-input-file` test is several *independent*
                # modules in one file, and lit runs each on its own. They
                # routinely reuse names across sections — 129 of the 247
                # usable tests define e.g. `#blocked1` more than once, and
                # every one of those files is a sectioned test. Kept whole
                # the file is not valid IR: the second definition is a
                # redefinition, so the module fails to parse before any
                # pass runs. Measured on 200 fused pairs, 154 collided.
                #
                # The RUN line belongs to every section, so it is carried
                # onto each.
                sections = _split_sections(content)
                for idx, section in enumerate(sections):
                    if not section.strip():
                        skipped += 1
                        continue
                    name = flat if len(sections) == 1 else f"{flat[:-5]}__s{idx}.mlir"
                    with open(os.path.join(seeds, name), "w", encoding="utf-8") as f:
                        f.write(section)
                    n += 1
        print(f"  {rel:20s} -> {n} files")
        copied += n
    print(f"Collected {copied} .mlir files into {seeds} ({skipped} skipped)")
    return copied


def _build_corpus(project_root):
    """Parse seeds/ into corpus.db through the project's own parser, which
    is what splits `// -----` sections apart and records each seed's pass
    pipeline and module attributes."""
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)      # corpus is derived entirely from the test tree
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_triton_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "triton-src")
    build_dir = os.path.join(project_root, "triton-build")
    triton_opt = os.path.join(build_dir, "bin", "triton-opt")

    if not os.path.exists(triton_opt):
        if not os.path.exists(os.path.join(src_root, "CMakeLists.txt")):
            print(f"Cloning Triton ({TRITON_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {TRITON_BRANCH} {TRITON_REPO} {src_root}")

        llvm_dir = _fetch_llvm(project_root, src_root)
        json_dir = _fetch_json(project_root, src_root)

        os.makedirs(build_dir, exist_ok=True)
        configure = f"""
set -e
cd {build_dir}
cmake -G Ninja {src_root} \\
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g1" \\
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g1" \\
    -DCMAKE_C_COMPILER=clang \\
    -DCMAKE_CXX_COMPILER=clang++ \\
    -DLLVM_SYSPATH={llvm_dir} \\
    -DJSON_SYSPATH={json_dir} \\
    -DMLIR_DIR={llvm_dir}/lib/cmake/mlir \\
    -DLLVM_DIR={llvm_dir}/lib/cmake/llvm \\
    -DTRITON_CACHE_PATH={build_dir}/triton-cache \\
    -DTRITON_OFFLINE_BUILD=ON \\
    -DTRITON_BUILD_PYTHON_MODULE=OFF \\
    -DTRITON_BUILD_UT=OFF \\
    -DTRITON_CODEGEN_BACKENDS="{CODEGEN_BACKENDS}" \\
    -DLLVM_ENABLE_WERROR=OFF
"""
        _run(configure)
        # Only this target. A full `ninja` also builds the AMD/NVIDIA
        # codegen libraries' test binaries and the Proton profiler, none of
        # which this adapter runs.
        _run(f"cd {build_dir} && ninja -j{_jobs()} triton-opt")

    if not os.path.exists(triton_opt):
        raise RuntimeError(f"Triton build failed: {triton_opt} not found")

    ver = subprocess.run([triton_opt, "--version"], capture_output=True, text=True)
    print((ver.stdout or ver.stderr).strip().splitlines()[0] if (ver.stdout or ver.stderr)
          else "triton-opt built")

    _collect_mlir_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"Triton setup complete. triton-opt: {triton_opt}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Not separately useful here — the pass
    pipelines a seed exercises are named in its own RUN line — so this
    reuses the normal build."""
    setup(project_root)
