import os
import sys
import shutil
import subprocess
from pathlib import Path
import json

# Name of the map, written next to the flat seed files, from each seed's
# flat filename back to the php-src test directory it came from.
ORIGINS_FILE = ".origins.json"


def _flat_name(path: Path, root: Path) -> str:
    """
    Derive a unique flat filename from a path relative to root.
    e.g. ext/curl/tests/basic.phpt  ->  ext_curl_tests_basic.phpt
    Path separators become underscores; the file extension is preserved.
    """
    rel = path.relative_to(root)
    parts = list(rel.parts)          # ['ext', 'curl', 'tests', 'basic.phpt']
    stem  = Path(parts[-1]).stem     # 'basic'
    ext   = Path(parts[-1]).suffix   # '.phpt'
    prefix_parts = parts[:-1]        # ['ext', 'curl', 'tests']
    flat = "_".join(prefix_parts + [stem]) if prefix_parts else stem
    return flat + ext


def collect_phpt(project_root, php_src_dir):
    """Copy php-src's .phpt files into phpt_seeds/ and their fixtures into
    phpt_deps/, and record where each seed came from.

    Seeds keep the flat `ext_curl_tests_basic.phpt` names the corpus is
    keyed by. Fixtures do *not*: they are laid out as
    `phpt_deps/ext/curl/tests/<file>`, mirroring php-src, because a test
    reaches them through `__DIR__ . '/foo.inc'` or `require 'foo.inc'`
    with its own name, and 384 test directories reuse names such as
    `server.inc`, `config.inc` and `test.inc` for different contents.
    The flat copy this replaced (`ext_ftp_tests_server.inc`) could be
    found by no seed at all: every one of the 2465 seeds that includes a
    fixture, and every one that opens a data file beside itself, failed
    with "Failed opening required" or "No such file", and so did every
    child fused from them.

    `phpt_seeds/.origins.json` maps flat seed name -> php-src test
    directory (relative). projects/php/parser.py stores it on each seed
    as `source_dir`, the fusion strategies carry it into the child as
    `dep_dirs`, and projects/php/driver.py links those directories'
    files beside the program it runs.
    """
    seeds_dir = Path(project_root) / "phpt_seeds"
    deps_dir = Path(project_root) / "phpt_deps"
    seeds_dir.mkdir(exist_ok=True)
    deps_dir.mkdir(exist_ok=True)

    php_src_path = Path(php_src_dir)
    if not php_src_path.exists():
        return
    phpt_files = list(php_src_path.rglob("*.phpt"))
    print(f"Found {len(phpt_files)} .phpt files. Collecting seeds...")

    origins = {}
    source_directories = set()
    for phpt_path in phpt_files:
        source_directories.add(phpt_path.parent)
        flat = _flat_name(phpt_path, php_src_path)
        origins[flat] = str(phpt_path.parent.relative_to(php_src_path))
        try:
            shutil.copy2(phpt_path, seeds_dir / flat)
        except shutil.Error:
            pass
    with open(seeds_dir / ORIGINS_FILE, "w") as f:
        json.dump(origins, f, indent=0, sort_keys=True)

    # The old flat fixture copies are unreachable by name (see above);
    # drop them so nothing keeps resolving against them by accident.
    for item in deps_dir.iterdir():
        if item.is_file():
            item.unlink()

    n_deps = 0
    for folder in source_directories:
        rel = folder.relative_to(php_src_path)
        target = deps_dir / rel
        target.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.is_file() and item.suffix != ".phpt":
                try:
                    shutil.copy2(item, target / item.name)
                    # The driver links fixtures into a per-execution
                    # directory; a program writing through such a link
                    # would rewrite the shared fixture for every later
                    # run. Read-only files turn that into a failed write.
                    os.chmod(target / item.name, 0o444)
                    n_deps += 1
                except (shutil.Error, OSError):
                    pass
    print(f"PHPT seeds collected: {len(phpt_files)} seeds, {n_deps} fixture files "
          f"in {len(source_directories)} directories.")


def seeds_need_collecting(project_root) -> bool:
    """True until collect_phpt has run with the current layout."""
    return not (Path(project_root) / "phpt_seeds" / ORIGINS_FILE).exists()



def setup(project_root):
    """
    Sets up the PHP fuzzing environment (runs inside the ffl-php container):
    1. Clones php-src if needed.
    2. Builds PHP with ASAN + debug assertions.
    3. Collects .phpt seed files.
    """
    print(f"Setting up PHP in: {project_root}")
    project_root = str(Path(project_root).resolve())

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    php_src_dir = os.path.join(project_root, "php-src")
    php_bin = os.path.join(php_src_dir, "sapi", "cli", "php")

    # 1. Check if binary already exists
    if os.path.exists(php_bin):
        print(f"PHP binary already exists at {php_bin}")
        _run(f"{php_bin} --version")
        # A build from before fixtures were laid out per directory still
        # needs its seeds re-collected; the binary itself is fine.
        if seeds_need_collecting(project_root):
            collect_phpt(project_root, php_src_dir)
        return

    # 2. Clone php-src
    if not os.path.exists(php_src_dir):
        _run(f"git clone https://github.com/php/php-src.git {php_src_dir}")

    # 3. Build PHP
    build_script = f"""
set -e
cd {php_src_dir}
echo "Configuring PHP..."
export CC=clang-12 CXX=clang++-12
export CFLAGS="-DZEND_VERIFY_TYPE_INFERENCE"
export CXXFLAGS="-DZEND_VERIFY_TYPE_INFERENCE"
./buildconf --force
./configure \\
    --enable-debug --enable-address-sanitizer --enable-undefined-sanitizer \\
    --enable-re2c-cgoto --enable-fpm --enable-phpdbg-debug --enable-zts \\
    --enable-bcmath --enable-calendar --enable-dba --enable-dl-test \\
    --enable-exif --enable-ftp --enable-gd --enable-mbstring \\
    --enable-pcntl --enable-shmop --enable-soap --enable-sockets \\
    --enable-sysvmsg --enable-zend-test --with-zlib --with-bz2 \\
    --with-curl --with-gmp --with-mhash --with-ldap --with-libedit \\
    --with-readline --with-sodium --with-xsl --with-zip \\
    --with-mysqli --with-pdo-mysql --with-sqlite3 --with-pdo-sqlite \\
    --with-webp --with-jpeg --with-freetype --enable-sigchild \\
    --with-pcre-jit --with-iconv
make -j$(nproc)
echo "PHP build complete."
"""
    _run(build_script)
    _run(f"{php_bin} --version")

    # 4. Collect .phpt seed files and their fixtures
    collect_phpt(project_root, php_src_dir)

    print("PHP setup complete.")


def setup_cov(project_root):
    """
    Build PHP with gcov instrumentation (no sanitizers) for line-coverage measurement.
    """
    print(f"Setting up PHP (gcov) in: {project_root}")
    project_root = str(Path(project_root).resolve())

    def _run(cmd_str, cwd=None):
        print(f"[run] {cmd_str[:100]}...")
        subprocess.run(["sh", "-c", cmd_str], check=True, cwd=cwd)

    php_src_dir = os.path.join(project_root, "php-src")
    php_bin = os.path.join(php_src_dir, "sapi", "cli", "php")

    if os.path.exists(php_bin):
        print(f"PHP binary already exists at {php_bin}")
        _run(f"{php_bin} --version")
        # A build from before fixtures were laid out per directory still
        # needs its seeds re-collected; the binary itself is fine.
        if seeds_need_collecting(project_root):
            collect_phpt(project_root, php_src_dir)
        return

    if not os.path.exists(php_src_dir):
        _run(f"git clone https://github.com/php/php-src.git {php_src_dir}")

    build_script = f"""
set -e
cd {php_src_dir}
echo "Configuring PHP (gcov, no sanitizers)..."
export CC=gcc CXX=g++
export CFLAGS="-DZEND_VERIFY_TYPE_INFERENCE"
export CXXFLAGS="-DZEND_VERIFY_TYPE_INFERENCE"
./buildconf --force
./configure \\
    --enable-debug --enable-gcov \\
    --enable-re2c-cgoto --enable-fpm --enable-phpdbg-debug --enable-zts \\
    --enable-bcmath --enable-calendar --enable-dba --enable-dl-test \\
    --enable-exif --enable-ftp --enable-gd --enable-mbstring \\
    --enable-pcntl --enable-shmop --enable-soap --enable-sockets \\
    --enable-sysvmsg --enable-zend-test --with-zlib --with-bz2 \\
    --with-curl --with-gmp --with-mhash --with-ldap --with-libedit \\
    --with-readline --with-sodium --with-xsl --with-zip \\
    --with-mysqli --with-pdo-mysql --with-sqlite3 --with-pdo-sqlite \\
    --with-webp --with-jpeg --with-freetype --enable-sigchild \\
    --with-pcre-jit --with-iconv
make -j$(nproc)
echo "PHP (gcov) build complete."
"""
    _run(build_script)
    _run(f"{php_bin} --version")

    collect_phpt(project_root, php_src_dir)

    print("PHP (gcov) setup complete.")
