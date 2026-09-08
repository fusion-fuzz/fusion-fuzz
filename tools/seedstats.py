#!/usr/bin/env python3
"""
tools/seedstats.py — what --pre-analysis / --dry-run learned about a
project's seeds, by seed type.

A fused child can only be valid if both parents are, so the share of
seeds the target rejects on their own bounds the fused rate from above.
This reads the `rc` each seed's own execution recorded into corpus.db and
prints the distribution per seed type (project seeds vs. the injected
bug corpus), so the corpus half of an invalid rate can be told apart from
the fusion half.

    python3 tools/seedstats.py --project go
    python3 tools/seedstats.py --project php --by-dir   # per source directory
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
from core.parser import pruned_identifiers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--by-dir", action="store_true", help="also group project seeds by their first path component")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    db = os.path.join("projects", args.project, "corpus.db")
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT identifier, metadata FROM seeds").fetchall()
    conn.close()
    pruned = pruned_identifiers(db)

    by_type = collections.defaultdict(collections.Counter)
    by_dir = collections.defaultdict(collections.Counter)
    n_pruned = 0
    n_noinfo = 0
    for ident, meta_json in rows:
        if ident in pruned:
            n_pruned += 1
            continue
        meta = json.loads(meta_json or "{}")
        t = meta.get("type") or "?"
        rc = meta.get("rc")
        if rc is None:
            n_noinfo += 1
            key = "not executed"
        else:
            key = "rc=0" if rc == 0 else ("timeout" if rc == 124 else f"rc={rc}")
        by_type[t][key] += 1
        if args.by_dir and t != "bug_corpus":
            top = ident.split("/")[0] if "/" in ident else ident.split("_")[0]
            by_dir[top][key] += 1

    print(f"{args.project}: {len(rows)} seeds in corpus.db, {n_pruned} pruned, {n_noinfo} never executed")
    for t, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        ok = c.get("rc=0", 0)
        print(f"  type={t:<12s} n={total:6d}  rc=0: {ok:6d} ({100.0*ok/total:5.1f}%)  "
              + "  ".join(f"{k}:{v}" for k, v in c.most_common(6) if k != "rc=0"))
    if args.by_dir:
        print("  by directory (project seeds):")
        for d, c in sorted(by_dir.items(), key=lambda kv: -sum(kv[1].values()))[:args.top]:
            total = sum(c.values())
            ok = c.get("rc=0", 0)
            print(f"    {d:<24s} n={total:6d}  rc=0: {100.0*ok/total:5.1f}%")


if __name__ == "__main__":
    main()
