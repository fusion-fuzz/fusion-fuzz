#!/usr/bin/env python3
"""
Drop seeds that PHP's own compiler rejects from projects/php/corpus.db.

php-src's test suite is the seed source, and a good part of it exists to
pin down *diagnostics*: tests whose --EXPECT-- section is a "Parse error"
or a compile-time "Fatal error" (an invalid escape sequence, `Duplicate
type array is redundant`, a private final constant). Such a seed cannot
be fused into anything the compiler accepts — every child inherits the
same compile-time failure, on the same line, and the run learns nothing
new from it. Measured before this pass, "Parse error: syntax error,
unexpected token" alone was 25-49% of every strategy's invalid children.

Only compile-time failures are pruned, i.e. what `php -l` rejects on the
--FILE-- section. A seed that compiles and then fails at *runtime* (an
uncaught exception, a memory-limit test, a deprecated call) stays: the
fused child wraps it in try/catch, the interpreter paths it reaches are
real, and a runtime fatal is a legitimate thing to feed a fuzzer.

Run inside the fuzzing container (needs the built php binary):

    python3 projects/php/prune_corpus.py            # prune
    python3 projects/php/prune_corpus.py --dry-run  # report only

projects/php/parser.py also calls prune_lint_failures() at the end of
--setup, so a fresh install gets this automatically.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from core.parser import record_pruned  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHP_BIN = os.path.join(PROJECT_ROOT, "php-src", "sapi", "cli", "php")
TIMEOUT = 30   # ASan php starts slowly; a lint never runs user code

_SECTION_RE = re.compile(r'^--([A-Z_]+)--\s*$', re.M)


def file_section(phpt: str) -> str:
    """The --FILE-- section of a .phpt, or the whole text if it has none."""
    m = re.search(r'^--FILE--\s*$', phpt, re.M)
    if not m:
        return phpt
    rest = phpt[m.end():]
    n = _SECTION_RE.search(rest)
    return rest[:n.start()] if n else rest


def lint(php_bin: str, code: str, workdir: str, tag: str):
    """(ok, first diagnostic line) for `php -l` on `code`."""
    path = os.path.join(workdir, f"lint_{tag}.php")
    try:
        with open(path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(code)
        proc = subprocess.run(
            [php_bin, "-n", "-d", "display_errors=1", "-d", "error_reporting=-1",
             "-d", "log_errors=0", "-l", path],
            capture_output=True, text=True, errors="replace",
            timeout=TIMEOUT, cwd=workdir,
            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"},
        )
        ok = proc.returncode == 0
        diag = ""
        if not ok:
            for line in (proc.stdout + proc.stderr).splitlines():
                if "error" in line.lower():
                    diag = line.strip()[:160]
                    break
        return ok, diag
    except subprocess.TimeoutExpired:
        return False, "lint timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"lint failed: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def prune_lint_failures(db_path: str, php_bin: str = PHP_BIN, workers: int = None,
                        dry_run: bool = False, verbose: bool = True) -> int:
    """Lint every seed's --FILE-- section; record the rejects as pruned.
    Returns the number pruned (or that would be, with dry_run)."""
    if not os.path.isfile(php_bin) or not os.access(php_bin, os.X_OK):
        if verbose:
            print(f"[prune] php binary not found at {php_bin}; skipping lint prune")
        return 0
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT identifier, content, metadata FROM seeds").fetchall()
    conn.close()
    workers = workers or max(2, (os.cpu_count() or 4))
    if verbose:
        print(f"[prune] linting {len(rows)} seeds with {php_bin} ({workers} workers)")

    rejects = []
    reasons = {}
    with tempfile.TemporaryDirectory(prefix="phplint") as workdir:
        def _one(idx_row):
            idx, (ident, content, meta) = idx_row
            ok, diag = lint(php_bin, file_section(content or ""), workdir, str(idx))
            return ident, ok, diag
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for n, (ident, ok, diag) in enumerate(pool.map(_one, enumerate(rows)), 1):
                if not ok:
                    rejects.append(ident)
                    key = re.sub(r'\b\d+\b', 'N', diag.split(' in /')[0])[:90]
                    reasons[key] = reasons.get(key, 0) + 1
                if verbose and n % 2000 == 0:
                    print(f"[prune]   {n}/{len(rows)} linted, {len(rejects)} rejected so far")

    if verbose:
        print(f"[prune] {len(rejects)}/{len(rows)} seeds rejected by php -l")
        for key, cnt in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            print(f"[prune]   {cnt:6d}  {key}")
    if rejects and not dry_run:
        added = record_pruned(db_path, rejects, reason="php -l: compile-time error")
        if verbose:
            print(f"[prune] recorded {added} newly pruned identifiers in {db_path}")
    return len(rejects)


# A "Fatal error" that is not an uncaught exception is one the engine
# raises while *building* the program — trait composition conflicts,
# incompatible signatures, abstract instantiation, a memory-limit test —
# and the fused child's try/catch cannot catch it. Seeds that end this way
# on their own end every child the same way.
_FATAL_RE = re.compile(r'^(?:PHP )?Fatal error: (?!Uncaught\b)([^\n]{0,120})', re.M)


def prune_runtime_fatal(db_path: str, workers: int = None, dry_run: bool = False,
                        verbose: bool = True) -> int:
    """Execute every seed through projects/php/driver.py (fixtures, INI,
    sandbox as in fuzzing) and prune those that die with a non-exception
    fatal error. Returns the number pruned."""
    if not os.path.exists(db_path):
        return 0
    sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..")))
    from core.config_loader import load_project_config
    from core.driver import get_driver
    from core.fusion import Seed
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("ffl_php_parser_prune", os.path.join(PROJECT_ROOT, "parser.py"))
    parser = _ilu.module_from_spec(spec)
    spec.loader.exec_module(parser)

    seeds = parser.load_corpus(db_path)
    config = load_project_config("php")
    driver = get_driver(config)
    driver.dryrun_mode = True
    workers = workers or max(2, (os.cpu_count() or 4))
    if verbose:
        print(f"[prune] executing {len(seeds)} seeds for compile-time fatals ({workers} workers)")

    rejects, reasons = [], {}

    def _one(sd):
        seed = Seed(content=sd["content"], metadata={**sd["metadata"], "filename": sd["filename"]})
        try:
            res = driver.execute(seed)
        except Exception as e:  # noqa: BLE001
            return sd["filename"], None
        if res.return_code == 0:
            return sd["filename"], None
        m = _FATAL_RE.search((res.stdout or "") + "\n" + (res.stderr or ""))
        return sd["filename"], (m.group(1) if m else None)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (ident, fatal) in enumerate(pool.map(_one, seeds), 1):
            if fatal:
                rejects.append(ident)
                key = re.sub(r'\b\d+\b', 'N', fatal.split(' in /')[0])[:80]
                reasons[key] = reasons.get(key, 0) + 1
            if verbose and n % 2000 == 0:
                print(f"[prune]   {n}/{len(seeds)} executed, {len(rejects)} fatal so far")

    if verbose:
        print(f"[prune] {len(rejects)}/{len(seeds)} seeds die with a compile-time fatal error")
        for key, cnt in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            print(f"[prune]   {cnt:6d}  {key}")
    if rejects and not dry_run:
        added = record_pruned(db_path, rejects, reason="compile-time fatal error on its own")
        if verbose:
            print(f"[prune] recorded {added} newly pruned identifiers in {db_path}")
    return len(rejects)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "corpus.db"))
    ap.add_argument("--php", default=PHP_BIN)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-lint", action="store_true", help="only the execution pass")
    ap.add_argument("--skip-exec", action="store_true", help="only the php -l pass")
    args = ap.parse_args()
    if not args.skip_lint:
        prune_lint_failures(args.db, args.php, args.workers, args.dry_run)
    if not args.skip_exec:
        prune_runtime_fatal(args.db, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
