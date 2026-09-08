#!/usr/bin/env python3
"""
tools/flagnoise.py — how often does the driver's *random flag draw* alone
reject a seed that passed pre-analysis?

Fused children are judged with the same random flags the fuzzer uses, so a
flag that rejects a valid program (e.g. LFortran's `--std=legacy`, which
switches to fixed-form parsing) counts against every strategy's validity
without any fusion being involved. This runs N seeds with rc=0 from
pre-analysis K times each through the project driver and reports the share
of executions rejected, with the failure classes.

    python3 tools/flagnoise.py --project lfortran [--seeds 60 --repeat 3]
"""
import argparse, collections, os, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.validrate import load_corpus, first_diagnostic, normalise   # noqa: E402
from core.config_loader import load_project_config                    # noqa: E402
from core.driver import get_driver                                    # noqa: E402
from core.orchestrator import FusionFuzzLoop                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--sample-seed", type=int, default=1)
    a = ap.parse_args()
    config = load_project_config(a.project)
    config.setdefault("execution", {})["concurrency"] = a.concurrency
    driver = get_driver(config)
    loop = FusionFuzzLoop.__new__(FusionFuzzLoop)
    loop.config = config
    seeds = [s for s in load_corpus(a.project, config, True) if (s.metadata or {}).get("rc") == 0]
    random.Random(a.sample_seed).shuffle(seeds)
    seeds = seeds[:a.seeds]
    jobs = [(s, k) for s in seeds for k in range(a.repeat)]
    print(f"[flagnoise] {a.project}: {len(seeds)} rc=0 seeds x {a.repeat}")
    classes = collections.Counter(); bad = 0; total = 0; per_seed = collections.Counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(driver.execute, s): s for s, _ in jobs}
        for f in as_completed(futs):
            s = futs[f]; total += 1
            try:
                res = f.result()
            except Exception as e:
                classes[f"<driver error {type(e).__name__}>"] += 1; bad += 1; continue
            if loop._is_syntax_error(res):
                bad += 1; per_seed[s.id] += 1
                classes[normalise(first_diagnostic(res.stdout or "", res.stderr or "", res.return_code))] += 1
    print(f"[flagnoise] rejected {bad}/{total} executions = {100.0*bad/max(total,1):.1f}%  "
          f"(seeds rejected at least once: {len(per_seed)}/{len(seeds)}, always: {sum(1 for v in per_seed.values() if v == a.repeat)})")
    for k, v in classes.most_common(8):
        print(f"  {v:4d}  {k[:110]}")


if __name__ == "__main__":
    main()
