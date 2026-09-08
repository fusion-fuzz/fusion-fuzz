#!/usr/bin/env python3
"""
tools/ratetable.py — the per-project rate table for the six-hour report,
from output/validrate/history.tsv.

For every project and sample seed: the first row tagged `baseline`, the
previous report's row (if a --since timestamp is given) and the latest
row. Rates are dataflow/state/declaration/combined.

    python3 tools/ratetable.py                 # latest vs baseline
    python3 tools/ratetable.py --since 20260907-120000
"""
import argparse
import collections
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="output/validrate/history.tsv")
    ap.add_argument("--since", default=None, help="timestamp (YYYYmmdd-HHMMSS) of the previous report")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.history), delimiter="\t"))
    by = collections.defaultdict(list)
    for r in rows:
        if r["tag"] in ("smoke", "diag", "now1"):
            continue
        by[(r["project"], r["sample_seed"])].append(r)

    KEYS = ("dataflow_rate", "state_rate", "declaration_rate", "combined_rate")

    def fmt(r):
        if r is None:
            return "-"
        return "/".join((r.get(k) or "-") for k in KEYS)

    def latest_per_strategy(rs):
        """A later run may have measured only some strategies; for each
        strategy take the most recent run that has it."""
        out = {}
        for k in KEYS:
            out[k] = next((r[k] for r in reversed(rs) if r.get(k)), "-")
        return out

    out = []
    header = ("project", "sample", "baseline", "previous", "now", "tag", "seeds")
    out.append(header)
    for (proj, seed), rs in sorted(by.items()):
        # `baseline2` = HEAD re-measured after a metric correction (MLIR:
        # host-only fallback children no longer count); it supersedes.
        base = (next((r for r in rs if r["tag"] == "baseline2"), None)
                or next((r for r in rs if r["tag"] == "baseline"), None))
        # Baseline rows are HEAD measurements, whenever they were taken.
        progress = [r for r in rs if not r["tag"].startswith("baseline")] or rs
        latest = progress[-1]
        prev = None
        if args.since:
            older = [r for r in progress if r["time"] <= args.since]
            prev = latest_per_strategy(older) if older else None
        out.append((proj, f"s{seed}", fmt(base), fmt(prev), fmt(latest_per_strategy(progress)), latest["tag"], latest["seeds"]))

    widths = [max(len(str(row[i])) for row in out) for i in range(len(header))]
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for row in out[1:]:
            print("| " + " | ".join(str(c) for c in row) + " |")
    else:
        for row in out:
            print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
