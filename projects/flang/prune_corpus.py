#!/usr/bin/env python3
"""
Drop seeds that flang itself rejects from projects/flang/corpus.db.

Fusion can only produce a valid child from valid parents, so an invalid
seed costs every pair it takes part in. Two groups of seeds are dropped:

  * anything `flang -fsyntax-only` does not accept on its own. flang's
    regression suite (the seed source) is full of tests that are *meant*
    to fail — negative semantics tests, files that USE a module defined in
    a sibling file, `#include`s of headers that aren't there — and the
    LLM-translated bug corpus injected by --bug-corpus is worse still
    (measured 35% valid: truncated files, stray markdown fences).
  * fixed-form sources (.f/.F). They are <1% of the corpus, and fusing
    one with a free-form seed yields a file that is neither.

Run inside the fuzzing container (needs `flang` on PATH):

    python3 projects/flang/prune_corpus.py            # prune
    python3 projects/flang/prune_corpus.py --dry-run  # report only
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from core.parser import record_pruned  # noqa: E402

FIXED_FORM_EXTS = ('.f', '.F')
TIMEOUT = 20


def _check(args):
    workdir, seed_id, identifier, content, ext = args
    path = os.path.join(workdir, f"s{seed_id}{ext}")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        proc = subprocess.run(
            f"ulimit -v 3145728; flang -fsyntax-only {path}",
            shell=True, capture_output=True, text=True,
            timeout=TIMEOUT, cwd=workdir,
        )
        ok = proc.returncode == 0
    except Exception:
        ok = False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return identifier, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "corpus.db"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not shutil.which("flang"):
        sys.exit("flang not found in PATH — run this inside the fuzzing container.")

    conn = sqlite3.connect(args.db)
    rows = conn.execute("SELECT id, identifier, content, metadata FROM seeds").fetchall()
    conn.close()
    print(f"{len(rows)} seeds in {args.db}")

    drop_fixed = []
    todo = []
    for seed_id, identifier, content, metadata in rows:
        try:
            meta = json.loads(metadata or "{}")
        except ValueError:
            meta = {}
        ext = meta.get("extension") or ".f90"
        if ext in FIXED_FORM_EXTS:
            drop_fixed.append(identifier)
            continue
        todo.append((seed_id, identifier, content, ext))

    workdir = tempfile.mkdtemp(prefix="flang-prune-")
    try:
        with cf.ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(
                _check, ((workdir, i, ident, c, e) for i, ident, c, e in todo)))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    drop_invalid = [identifier for identifier, ok in results if not ok]
    keep = len(rows) - len(drop_fixed) - len(drop_invalid)
    print(f"fixed-form: {len(drop_fixed)} dropped")
    print(f"rejected by flang: {len(drop_invalid)} dropped")
    print(f"keeping {keep} ({100.0 * keep / max(len(rows), 1):.1f}%)")

    if args.dry_run:
        return

    backup = args.db + ".prebackup"
    if not os.path.exists(backup):
        shutil.copy2(args.db, backup)
        print(f"backup written to {backup}")

    record_pruned(args.db, drop_fixed, reason="fixed-form")
    record_pruned(args.db, drop_invalid, reason="rejected by flang -fsyntax-only")
    conn = sqlite3.connect(args.db)
    conn.execute("VACUUM")
    conn.close()
    print(f"pruned {len(drop_fixed) + len(drop_invalid)} seeds "
          f"(recorded so --setup/--bug-corpus will not re-add them)")


if __name__ == "__main__":
    main()
