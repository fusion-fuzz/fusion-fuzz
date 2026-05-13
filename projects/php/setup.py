import os
import sys
import shutil
import subprocess
from pathlib import Path


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

    # 4. Collect .phpt seed files
    seeds_dir = Path(project_root) / "phpt_seeds"
    deps_dir = Path(project_root) / "phpt_deps"
    seeds_dir.mkdir(exist_ok=True)
    deps_dir.mkdir(exist_ok=True)

    php_src_path = Path(php_src_dir)
    if php_src_path.exists():
        phpt_files = list(php_src_path.rglob("*.phpt"))
        print(f"Found {len(phpt_files)} .phpt files. Collecting seeds...")

        def _flat_name(path: Path, root: Path) -> str:
            """
            Derive a unique flat filename from a path relative to root.
            e.g. ext/curl/tests/basic.phpt  →  ext_curl_tests_basic.phpt
            Path separators become underscores; the file extension is preserved.
            """
            rel = path.relative_to(root)
            parts = list(rel.parts)          # ['ext', 'curl', 'tests', 'basic.phpt']
            stem  = Path(parts[-1]).stem     # 'basic'
            ext   = Path(parts[-1]).suffix   # '.phpt'
            prefix_parts = parts[:-1]        # ['ext', 'curl', 'tests']
            flat = "_".join(prefix_parts + [stem]) if prefix_parts else stem
            return flat + ext

        source_directories = set()
        for phpt_path in phpt_files:
            source_directories.add(phpt_path.parent)
            dst = seeds_dir / _flat_name(phpt_path, php_src_path)
            try:
                shutil.copy2(phpt_path, dst)
            except shutil.Error:
                pass

        for folder in source_directories:
            for item in folder.iterdir():
                if item.is_file() and item.suffix != ".phpt":
                    dst = deps_dir / _flat_name(item, php_src_path)
                    try:
                        shutil.copy2(item, dst)
                    except shutil.Error:
                        pass

        print("PHPT seeds collected.")

    print("PHP setup complete.")
