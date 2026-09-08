#!/usr/bin/env python3
"""tools/divtable.py — diversity proxies, baseline vs latest, per project
and strategy, from output/validrate/*-summary.json."""
import glob, json, os, collections
runs = collections.defaultdict(list)
for f in sorted(glob.glob("output/validrate/*-summary.json")):
    _m = __import__("re").search(r'-(\d{8}-\d{6})-summary\.json$', f)
    d = json.load(open(f))
    d["_stamp"] = _m.group(1) if _m else ""
    if d["tag"] in ("smoke", "diag", "now1"):
        continue
    runs[(d["project"], d["sample_seed"])].append(d)
print(f"{'project':9s} {'s':2s} {'strategy':12s} {'valid%':>13s} {'distinct valid outputs':>24s} {'behaviour differs from both parents %':>38s}")
for (proj, seed), rs in sorted(runs.items()):
    rs = sorted(rs, key=lambda r: r.get("_stamp", ""))
    base = (next((r for r in rs if r["tag"] == "baseline2"), None)
            or next((r for r in rs if r["tag"] == "baseline"), None))
    progress = [r for r in rs if not r["tag"].startswith("baseline")] or rs
    latest = progress[-1]
    for strat in ("dataflow", "state", "declaration", "combined"):
        b = (base or {}).get("strategies", {}).get(strat, {})
        l = latest["strategies"].get(strat, {})
        if not l and not b:
            continue
        # a later run may have measured only some strategies; fall back to the last run that has it
        if not l:
            for r in reversed(rs):
                if strat in r["strategies"]:
                    l = r["strategies"][strat]; break
        bd, ld = b.get("diversity", {}), l.get("diversity", {})
        print(f"{proj:9s} s{seed:<1} {strat:12s} {str(b.get('rate','-')):>5s} -> {str(l.get('rate','-')):>5s} "
              f"{str(bd.get('valid_fingerprints','-')):>10s} -> {str(ld.get('valid_fingerprints','-')):>10s} "
              f"{str(bd.get('novel','-')):>17s} -> {str(ld.get('novel','-')):>17s}")
