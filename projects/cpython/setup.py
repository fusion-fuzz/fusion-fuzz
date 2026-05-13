import os
import sys
import subprocess
import zipfile


def setup(project_root):
    """
    Sets up the CPython fuzzing environment (runs inside the ffl-cpython container):
    1. Clones CPython if needed.
    2. Builds CPython with PyDebug + ASAN.
    3. Bootstraps pip and installs test dependencies.
    """
    print(f"Setting up CPython in: {project_root}")

    # Unzip fuzzing corpus if not already extracted
    project_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_zip = os.path.join(project_dir, "cpython-fuzzing-corpus.zip")
    corpus_dir = os.path.join(project_dir, "cpython-fuzzing-corpus")
    if not os.path.exists(corpus_dir):
        if os.path.exists(corpus_zip):
            print(f"Extracting {corpus_zip} ...")
            with zipfile.ZipFile(corpus_zip, "r") as zf:
                zf.extractall(project_dir)
            print(f"Corpus extracted to {corpus_dir}")
        else:
            print(f"Warning: corpus zip not found at {corpus_zip}")

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:80]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    cpython_dir = os.path.join(project_root, "cpython")
    build_dir = os.path.join(cpython_dir, "build")
    python_bin = os.path.join(build_dir, "python")

    # 1. Check if binary already exists
    if os.path.exists(python_bin):
        print(f"CPython binary already exists at {python_bin}")
        _run(f"{python_bin} --version")
    else:
        # 2. Clone CPython
        if not os.path.exists(cpython_dir):
            _run(f"git clone https://github.com/python/cpython.git {cpython_dir}")

        # 3. Configure and build
        build_script = f"""
set -e
mkdir -p {build_dir}
if [ ! -f {build_dir}/Makefile ]; then
    echo "Configuring CPython..."
    cd {build_dir} && ../configure --with-pydebug --enable-experimental-jit=yes --with-address-sanitizer
fi
echo "Building CPython..."
make -C {build_dir} -j$(nproc)
"""
        _run(build_script)
        _run(f"{python_bin} --version")

    # 4. Bootstrap pip + install test deps
    test_deps = ["xdrlib3", "telnetlib3", "pyasynchat", "legacy-cgi", "pytest"]
    deps_str = " ".join(test_deps)
    _run(
        f"{python_bin} -m ensurepip --upgrade && "
        f"{python_bin} -m pip install --quiet {deps_str}"
    )

    print("CPython setup complete.")
