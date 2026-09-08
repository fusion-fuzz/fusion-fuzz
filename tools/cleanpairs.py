#!/usr/bin/env python3
"""Validity over *clean pairs*: pairs where neither parent seed expects a
diagnostic (clang `expected-error`, gcc `dg-error`), next to the
all-pairs rate. Kept-on-purpose error seeds (config
`drop_seeds_failing_alone_except`) produce their own error whatever the
fusion does, so this is the rate the fusion is actually responsible for.

    python3 tools/cleanpairs.py --project gcc --tag r05f
    python3 tools/cleanpairs.py --project clang --tag r05f --regex 'expected-error'
"""
import argparse
import collections
import json
import os
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RX = {
    "clang": r"expected-error",
    "gcc": r"dg-error|dg-excess-errors",
    "go": r"// ?ERROR",
    "rust": r"//~\^* *ERROR|compile-fail",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regex", help="parent content pattern marking an error-expecting seed")
    args = ap.parse_args()
    rx = re.compile(args.regex or DEFAULT_RX.get(args.project, r"expected-error|dg-error"))
    con = sqlite3.connect(os.path.join(ROOT, "projects", args.project, "corpus.db"))
    expects = {}
    for (meta, content) in con.execute("select metadata, content from seeds"):
        expects[json.loads(meta).get("filename")] = bool(rx.search(content))
    vdir = os.path.join(ROOT, "output", "validrate")
    print(f"{'sample':6s} {'strategy':12s} {'all':>8s} {'clean pairs':>12s} {'clean valid':>12s}")
    for s in (1, 2):
        fs = sorted(x for x in os.listdir(vdir)
                    if x.startswith(f"{args.project}-{args.tag}-s{s}-") and x.endswith(".jsonl"))
        if not fs:
            continue
        tot, inv, clean, cleanv = (collections.Counter() for _ in range(4))
        for fn in fs:
            for line in open(os.path.join(vdir, fn)):
                r = json.loads(line)
                st = r["strategy"]
                if r["outcome"] in ("noviable", "unfused"):
                    continue
                tot[st] += 1
                e = expects.get(r["parent_a"], False) or expects.get(r["parent_b"], False)
                if r["outcome"] == "invalid":
                    inv[st] += 1
                if not e:
                    clean[st] += 1
                    cleanv[st] += r["outcome"] != "invalid"
        for st in ("dataflow", "state", "declaration", "combined"):
            if not tot[st]:
                continue
            print(f"s{s:<5d} {st:12s} {100 * (tot[st] - inv[st]) / tot[st]:7.1f}% "
                  f"{clean[st]:5d}/{tot[st]:<6d} {100 * cleanv[st] / max(1, clean[st]):11.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
