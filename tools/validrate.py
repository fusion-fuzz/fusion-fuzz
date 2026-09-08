#!/usr/bin/env python3
"""
tools/validrate.py — measure the fused-program validity rate of one project
over a fixed, reproducible sample, and classify the failures.

Why a separate tool
-------------------
The fuzzer's status bar prints FuseValidRate, but it is measured over a
different random sample every run, mixes every strategy into one number,
and throws the failing outputs away. A before/after comparison needs the
same seeds, the same pairs, the same strategy per pair, and the actual
diagnostics grouped by class — "82% of failures are X" is what points at
a fix, "the rate is 24.5%" is not.

What it does
------------
1. Load the project corpus exactly as main.py does (parser.load_corpus,
   bug-corpus filter, paths.seed_exclude_patterns).
2. Draw N seeds with a fixed RNG seed (--sample-seed) so the sample is the
   same tomorrow. Optionally reuse a saved sample (--sample-file).
3. For each strategy in the pool (dataflow / state / declaration) and for
   the "combined" chain the fuzzer really runs, fuse P ordered pairs drawn
   with a fixed RNG seed.
4. Execute every child through the project's own driver (the built
   toolchain under test, never the host's), in parallel.
5. Judge each result with core.orchestrator.FusionFuzzLoop._is_syntax_error
   — the very function the fuzzer's metric uses, not a proxy.
6. Group failures by a normalised first diagnostic line and print the
   distribution, per strategy and overall. Write every record to a JSONL
   file so a later run can be diffed against it.

Run inside the project's container, from the repo root:

    python3 tools/validrate.py --project php --seeds 200 --pairs 300 \
        --sample-seed 1 --tag baseline

The RNG seed governs seed sampling, pair selection and every random choice
the strategies make, so two runs with the same seed and the same code
produce the same children. A code change is measured by re-running with
the same seed (development sample) and then with a different one
(held-out sample, e.g. --sample-seed 2).
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.getcwd())

from core.config_loader import load_project_config          # noqa: E402
from core.driver import get_driver                          # noqa: E402
from core.fusion import get_strategies, Seed                # noqa: E402
from core.orchestrator import FusionFuzzLoop                # noqa: E402
from core.degradation import degradations                   # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FFL.ValidRate")


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_DIAG_RE = re.compile(
    r'(?i)\b(?:fatal error|parse error|syntax error|error|exception|panic|'
    r'warning: .*deprecated|undefined|cannot|expected|invalid|missing|'
    r'not used|redeclared|mismatch|too many|not enough)\b'
    r'|^\S+\.go:\d+:\d+: ')
_PATH_RE = re.compile(r'(?:/[\w.\-]+)+(?::\d+)*')
_QUOTED_RE = re.compile(r'(["\'`])(?:(?!\1).){0,80}\1')
_NUM_RE = re.compile(r'\b\d+\b')
_HEX_RE = re.compile(r'0x[0-9a-fA-F]+')
_ID_HASH_RE = re.compile(r'\b[0-9a-f]{8}\b')


_PY_EXC_LINE_RE = re.compile(r'^(?:[\w.]+\.)?\w*(?:Error|Exception|Warning|Exit|Interrupt)\b.*')


def first_diagnostic(stdout: str, stderr: str, rc: int) -> str:
    """The first line that looks like a diagnostic, stderr first.

    A Python traceback is read from the end: its last line is the
    exception, everything before it is frames and (for unittest output)
    rulers, which are not the failure."""
    err = _ANSI_RE.sub('', stderr or "")
    if "Traceback (most recent call last)" in err:
        for line in reversed(err.splitlines()):
            s = line.strip()
            if s and _PY_EXC_LINE_RE.match(s):
                return s
    # A located diagnostic (`file:12:3: error: ...`) beats a summary line
    # such as flang's "error: Semantic errors in x.f90", whichever stream
    # each lands on.
    for text in (stderr or "", stdout or ""):
        lines = _ANSI_RE.sub('', text).splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if re.search(r':\d+:\d+: (?:fatal )?error: ', s):
                # GHC's header carries only the code (`error: [GHC-88464]`);
                # the message is on the next line. Without it every Haskell
                # failure classified as "[GHC-<n>]".
                if re.search(r'error: \[GHC-\d+\]\s*$', s) or s.endswith('error:'):
                    for nxt in lines[i + 1:i + 4]:
                        if nxt.strip():
                            return s + " " + nxt.strip()
                return s
    for text in (stderr or "", stdout or ""):
        lines = _ANSI_RE.sub('', text).splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or re.match(r'^# \S+$', s):      # go's "# pkgname" header
                continue
            if _DIAG_RE.search(s):
                # GHC (and some others) put the message on the line
                # *after* "file:l:c: error: [GHC-12345]"; the header
                # alone classifies nothing.
                if re.search(r'error: \[GHC-\d+\]\s*$', s) or s.endswith('error:'):
                    for nxt in lines[i + 1:i + 4]:
                        if nxt.strip():
                            return s + " " + nxt.strip()
                return s
    # No diagnostic-looking line: fall back to first non-empty stderr line,
    # then a return-code label.
    for line in (stderr or "").splitlines():
        if line.strip():
            return line.strip()
    return f"<no diagnostic, rc={rc}>"


def normalise(diag: str) -> str:
    """Collapse a diagnostic into a class: drop paths, numbers, quoted
    names, hex and seed ids so the same kind of failure groups together."""
    s = _ANSI_RE.sub('', diag)
    s = _PATH_RE.sub('<path>', s)
    s = _QUOTED_RE.sub('<q>', s)
    s = _HEX_RE.sub('<hex>', s)
    s = _ID_HASH_RE.sub('<id>', s)
    s = _NUM_RE.sub('<n>', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:140]


# ---------------------------------------------------------------------------
# Diversity / semantic-depth proxies
# ---------------------------------------------------------------------------
#
# Validity is a means, not the end: a child that is valid because it is
# one parent with the other parent's code dropped finds nothing. These
# proxies are cheap and language-agnostic; none is a substitute for
# coverage, but together they catch the failure mode where a "fix" makes
# children valid by making them trivial.
#
#   distinct     share of executed children whose text is unique
#   retained     share of children keeping >= 50% of BOTH parents' lines
#   novel        share of children whose (normalised) output differs from
#                both parents' outputs — the fusion changed behaviour
#   fingerprints number of distinct normalised outputs among valid children

_FP_NOISE_RE = re.compile(r'(?:0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b|/[\w./\-]+|\b[0-9a-f]{8,}\b)')


def output_fingerprint(res) -> str:
    text = ((res.stderr or "")[:4000] + "\n" + (res.stdout or "")[:4000])
    text = _ANSI_RE.sub('', text)
    text = _FP_NOISE_RE.sub('#', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _code_lines(text: str):
    return {ln.strip() for ln in (text or "").splitlines() if ln.strip() and len(ln.strip()) > 3}


def retention(child: str, a: str, b: str):
    """(share of A's lines present in child, share of B's)."""
    cl = _code_lines(child)
    la, lb = _code_lines(a), _code_lines(b)
    ra = len(la & cl) / len(la) if la else 1.0
    rb = len(lb & cl) / len(lb) if lb else 1.0
    return ra, rb


# ---------------------------------------------------------------------------
# Corpus loading (mirrors main.py)
# ---------------------------------------------------------------------------

def _load_parser_module(project):
    parser_path = os.path.join("projects", project, "parser.py")
    spec = importlib.util.spec_from_file_location("project_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_corpus(project, config, bug_corpus):
    from main import filter_excluded_seeds
    db = os.path.join("projects", project, "corpus.db")
    if not os.path.exists(db):
        sys.exit(f"corpus not found: {db} (run main.py --setup first)")
    module = _load_parser_module(project)
    raw = module.load_corpus(db)
    if not bug_corpus:
        raw = [s for s in raw if s["metadata"].get("type") != "bug_corpus"]
    seeds = [Seed(content=s["content"], metadata={**s["metadata"], "filename": s["filename"]})
             for s in raw]
    seeds = filter_excluded_seeds(seeds, config)
    # Same rule main.py applies after --pre-analysis, so the measured
    # corpus is the one the fuzzer actually draws from.
    if (config.get("analysis") or {}).get("drop_seeds_failing_alone"):
        before = len(seeds)
        keep_rx = (config.get("analysis") or {}).get("drop_seeds_failing_alone_except")
        keep_rx = re.compile(keep_rx) if keep_rx else None
        seeds = [s for s in seeds if (s.metadata or {}).get("rc") in (None, 0)
                 or (keep_rx is not None and keep_rx.search(s.content or ""))]
        print(f"[corpus] dropped {before - len(seeds)} seeds failing alone (analysis.drop_seeds_failing_alone)")
    return seeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--seeds", type=int, default=200, help="seeds to sample from the corpus")
    ap.add_argument("--pairs", type=int, default=300, help="ordered pairs to fuse per strategy")
    ap.add_argument("--sample-seed", type=int, default=1, help="RNG seed for sampling and fusion")
    ap.add_argument("--sample-file", default=None,
                    help="JSON list of seed filenames to use instead of sampling (written on first run)")
    ap.add_argument("--strategies", default="dataflow,state,declaration,combined",
                    help="comma list of: dataflow, state, declaration, combined")
    ap.add_argument("--no-pre-analysis", action="store_true", help="measure the lightweight (no --pre-analysis) path")
    ap.add_argument("--bug-corpus", action="store_true", help="include injected bug-corpus seeds")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--fusion-rate", type=float, default=0.8)
    ap.add_argument("--tag", default="run", help="label for the output files")
    ap.add_argument("--out-dir", default="output/validrate")
    ap.add_argument("--top", type=int, default=15, help="failure classes to print")
    ap.add_argument("--dump-fail", type=int, default=0, help="write this many failing children per class to out-dir")
    ap.add_argument("--dump-all", action="store_true", help="write every child program + output to out-dir")
    ap.add_argument("--no-diversity", action="store_true", help="skip the diversity proxies (and parent execution)")
    args = ap.parse_args()

    config = load_project_config(args.project)
    if args.concurrency:
        config.setdefault("execution", {})["concurrency"] = args.concurrency
    workers = int(config.get("execution", {}).get("concurrency", 8))

    corpus = load_corpus(args.project, config, args.bug_corpus)
    print(f"[corpus] {len(corpus)} seeds loaded for {args.project}")

    rng = random.Random(args.sample_seed)
    if args.sample_file and os.path.exists(args.sample_file):
        wanted = json.load(open(args.sample_file))
        by_name = {s.metadata.get("filename"): s for s in corpus}
        sample = [by_name[n] for n in wanted if n in by_name]
        print(f"[sample] {len(sample)}/{len(wanted)} seeds from {args.sample_file}")
    else:
        sample = rng.sample(corpus, min(args.seeds, len(corpus)))
        if args.sample_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.sample_file)), exist_ok=True)
            json.dump([s.metadata.get("filename") for s in sample], open(args.sample_file, "w"), indent=0)
            print(f"[sample] wrote {len(sample)} seed names to {args.sample_file}")
    if len(sample) < 2:
        sys.exit("need at least 2 seeds")

    pre = not args.no_pre_analysis
    pool = get_strategies(args.project, dataflow_fusion=True, state_fusion=True,
                          declaration_fusion=True, pre_analysis_enabled=pre)
    by_kind = {}
    for s in pool:
        name = type(s).__name__
        if "Declaration" in name or "Struct" in name:
            by_kind["declaration"] = s
        elif "State" in name:
            by_kind["state"] = s
        else:
            by_kind["dataflow"] = s
    print(f"[strategies] {', '.join(f'{k}={type(v).__name__}' for k, v in by_kind.items())}")

    loop = FusionFuzzLoop(config=config, strategies=pool, initial_corpus=sample,
                          pre_analysis_enabled=pre, fusion_rate=args.fusion_rate)
    driver = loop.driver
    driver.prepare_environment()

    # Parents' own outputs, executed once, so a child can be compared
    # against both of them (see output_fingerprint / "novel").
    parent_fp = {}
    if not args.no_diversity:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(driver.execute, sd): sd for sd in sample}
            for f in as_completed(futs):
                sd = futs[f]
                try:
                    parent_fp[sd.id] = output_fingerprint(f.result())
                except Exception:
                    parent_fp[sd.id] = None
        print(f"[parents] executed {len(parent_fp)} sample seeds in {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out_dir, f"{args.project}-{args.tag}-s{args.sample_seed}-{stamp}")
    records_path = base + ".jsonl"
    dump_dir = base + "-dump"

    wanted = [k.strip() for k in args.strategies.split(",") if k.strip()]
    summary = {}
    all_records = []

    for kind in wanted:
        if kind != "combined" and kind not in by_kind:
            print(f"[{kind}] not available for {args.project}, skipping")
            continue
        # Fixed pair order per strategy: same RNG seed -> same pairs -> same
        # children as long as the code is unchanged.
        random.seed(args.sample_seed * 1000 + sum(map(ord, kind)))  # str hash is per-process randomised
        prng = random.Random(args.sample_seed)
        children = []
        t0 = time.time()
        for i in range(args.pairs):
            a, b = prng.sample(sample, 2)
            try:
                if kind == "combined":
                    usable = [s for s in pool if s.is_viable_pair(a, b)]
                    if not usable:
                        children.append((None, "noviable", a, b))
                        continue
                    chain = loop._pick_strategy_chain(usable)
                    host = a
                    for s in chain[:-1]:
                        host = s.fuse(host, b)
                    child = chain[-1].fuse(host, b)
                    mode = "+".join(type(s).__name__.replace("FusionStrategy", "") for s in chain)
                else:
                    strat = by_kind[kind]
                    if not strat.is_viable_pair(a, b):
                        children.append((None, "noviable", a, b))
                        continue
                    child = strat.fuse(a, b)
                    mode = str((child.metadata or {}).get("mode", kind)) if child else kind
                    # A strategy that gives up mid-way emits the host alone
                    # (`state_fallback_ab`, `decl_none_ab`, ...): a program
                    # the compiler already accepted, not a fused one. It
                    # counts as no child here, or the rate is the host's.
                    if re.search(r'fallback|nohostpoint|nocontinuation|_none_|_none$', mode):
                        children.append((None, f"unfused:{mode}", a, b))
                        continue
            except Exception as e:  # a strategy raising is itself a failure class
                children.append((None, f"raised:{type(e).__name__}", a, b))
                continue
            if child is None or not (child.content or "").strip():
                children.append((None, "empty", a, b))
                continue
            children.append((child, mode, a, b))
        fuse_secs = time.time() - t0

        # Execute
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(driver.execute, c): idx
                    for idx, (c, m, a, b) in enumerate(children) if c is not None}
            done = 0
            for f in as_completed(futs):
                idx = futs[f]
                try:
                    results[idx] = f.result()
                except Exception as e:
                    results[idx] = None
                    logger.error(f"driver error: {e}")
                done += 1
                if done % 50 == 0:
                    sys.stdout.write(f"\r[{kind}] executed {done}/{len(futs)}")
                    sys.stdout.flush()
        sys.stdout.write("\n")

        # Judge
        n_total = len(children)
        n_nochild = sum(1 for c, m, a, b in children if c is None)
        n_valid = 0
        n_invalid = 0
        n_timeout = 0
        n_crash = 0
        classes = collections.Counter()
        per_mode = collections.defaultdict(lambda: [0, 0])   # mode -> [valid, total]
        examples = collections.defaultdict(list)
        shas = set()
        n_retained = 0
        n_novel = 0
        n_novel_valid = 0
        valid_fps = set()
        ret_sum = [0.0, 0.0]
        for idx, (child, mode, a, b) in enumerate(children):
            rec = {"strategy": kind, "mode": mode,
                   "parent_a": a.metadata.get("filename"), "parent_b": b.metadata.get("filename")}
            if child is None:
                rec["outcome"] = mode
                classes[f"<no child: {mode}>"] += 1
                all_records.append(rec)
                continue
            res = results.get(idx)
            if res is None:
                rec["outcome"] = "driver-error"
                classes["<driver error>"] += 1
                all_records.append(rec)
                continue
            invalid = loop._is_syntax_error(res)
            rec.update({"rc": res.return_code, "crashed": bool(res.crashed),
                        "secs": round(res.execution_time, 3),
                        "sha": hashlib.sha1(child.content.encode("utf-8", "replace")).hexdigest()[:12]})
            per_mode[mode][1] += 1
            shas.add(rec["sha"])
            if not args.no_diversity:
                ra, rb = retention(child.content, a.content, b.content)
                ret_sum[0] += ra
                ret_sum[1] += rb
                rec["retained"] = [round(ra, 2), round(rb, 2)]
                if ra >= 0.5 and rb >= 0.5:
                    n_retained += 1
                fp = output_fingerprint(res)
                rec["novel"] = fp != parent_fp.get(a.id) and fp != parent_fp.get(b.id)
                if rec["novel"]:
                    n_novel += 1
            if res.return_code == 124:
                n_timeout += 1
            if res.crashed:
                n_crash += 1
                # A crash is the whole point; keep the child regardless of
                # --dump-fail so it can be reproduced and bundled.
                os.makedirs(dump_dir, exist_ok=True)
                _stem = os.path.join(dump_dir, f"crash-{kind}-{n_crash}")
                with open(_stem + loop._seed_extension(child), "w") as f:
                    f.write(child.content)
                with open(_stem + ".out", "w") as f:
                    f.write(f"# rc={res.return_code} mode={mode} parents={a.id} + {b.id}\n")
                    f.write((res.stderr or "")[:20000] + "\n--- stdout ---\n" + (res.stdout or "")[:5000])
            if invalid:
                n_invalid += 1
                diag = first_diagnostic(res.stdout, res.stderr, res.return_code)
                cls = normalise(diag)
                classes[cls] += 1
                rec.update({"outcome": "invalid", "diag": diag[:300], "class": cls})
                if len(examples[cls]) < max(args.dump_fail, 3):
                    examples[cls].append((child, res, mode))
            else:
                n_valid += 1
                per_mode[mode][0] += 1
                rec["outcome"] = "valid"
                if not args.no_diversity:
                    valid_fps.add(output_fingerprint(res))
                    if rec.get("novel"):
                        n_novel_valid += 1
            all_records.append(rec)
            if args.dump_all:
                os.makedirs(dump_dir, exist_ok=True)
                ext = loop._seed_extension(child)
                with open(os.path.join(dump_dir, f"{kind}-{idx:05d}{ext}"), "w") as f:
                    f.write(child.content)
                with open(os.path.join(dump_dir, f"{kind}-{idx:05d}.out"), "w") as f:
                    f.write(f"# rc={res.return_code} mode={mode} outcome={rec['outcome']}\n")
                    f.write((res.stderr or "") + "\n--- stdout ---\n" + (res.stdout or "")[:20000])

        executed = n_valid + n_invalid
        rate = 100.0 * n_valid / executed if executed else 0.0
        # Rate among pairs whose parents both compile/run on their own
        # (rc recorded by --pre-analysis): what the *fusion* costs, with
        # seed-level failures taken out. Only meaningful when rc is known.
        clean_v = clean_t = 0
        for idx, (child, mode, a, b) in enumerate(children):
            if child is None or results.get(idx) is None:
                continue
            if (a.metadata or {}).get("rc") == 0 and (b.metadata or {}).get("rc") == 0:
                clean_t += 1
                if not loop._is_syntax_error(results[idx]):
                    clean_v += 1
        clean_rate = (100.0 * clean_v / clean_t) if clean_t else None
        summary[kind] = {"pairs": n_total, "no_child": n_nochild, "executed": executed,
                         "valid": n_valid, "invalid": n_invalid, "timeouts": n_timeout,
                         "crashes": n_crash, "rate": round(rate, 1),
                         "clean_parent_pairs": clean_t,
                         "clean_parent_rate": round(clean_rate, 1) if clean_rate is not None else None,
                         "fuse_secs": round(fuse_secs, 1)}
        if not args.no_diversity and executed:
            div = {"distinct": round(100.0 * len(shas) / executed, 1),
                   "retained_both": round(100.0 * n_retained / executed, 1),
                   "mean_retained_a": round(100.0 * ret_sum[0] / executed, 1),
                   "mean_retained_b": round(100.0 * ret_sum[1] / executed, 1),
                   "novel": round(100.0 * n_novel / executed, 1),
                   "novel_valid": round(100.0 * n_novel_valid / n_valid, 1) if n_valid else 0.0,
                   "valid_fingerprints": len(valid_fps)}
            summary[kind]["diversity"] = div
        print(f"\n=== {args.project} / {kind}: valid {n_valid}/{executed} = {rate:.1f}%"
              f"  (no child: {n_nochild}, timeouts: {n_timeout}, crash-flagged: {n_crash}, "
              f"fusion {fuse_secs:.1f}s)")
        if clean_rate is not None:
            print(f"  clean-parent pairs (both parents rc=0): {clean_v}/{clean_t} = {clean_rate:.1f}%")
        if not args.no_diversity and executed:
            d = summary[kind]["diversity"]
            print(f"  diversity: distinct {d['distinct']}% | both parents >=50% retained {d['retained_both']}%"
                  f" (mean A {d['mean_retained_a']}%, B {d['mean_retained_b']}%) | behaviour differs from both"
                  f" parents {d['novel']}% (of valid: {d['novel_valid']}%) | {d['valid_fingerprints']} distinct"
                  f" valid outputs / {n_valid} valid")
        if len(per_mode) > 1:
            for mode, (v, t) in sorted(per_mode.items(), key=lambda kv: -kv[1][1])[:12]:
                print(f"    mode {mode:<40s} {v:4d}/{t:<4d} {100.0*v/t if t else 0:5.1f}%")
        print(f"  top failure classes ({len(classes)} distinct):")
        for cls, n in classes.most_common(args.top):
            print(f"    {n:5d}  {100.0*n/max(1,n_invalid+n_nochild):5.1f}%  {cls}")

        if args.dump_fail:
            os.makedirs(dump_dir, exist_ok=True)
            for cls, exs in examples.items():
                safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', cls)[:80]
                for j, (child, res, mode) in enumerate(exs[:args.dump_fail]):
                    ext = loop._seed_extension(child)
                    stem = os.path.join(dump_dir, f"{kind}-{safe}-{j}")
                    with open(stem + ext, "w") as f:
                        f.write(child.content)
                    with open(stem + ".out", "w") as f:
                        f.write(f"# rc={res.return_code} mode={mode}\n")
                        f.write((res.stderr or "") + "\n--- stdout ---\n" + (res.stdout or "")[:20000])

    with open(records_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    summary_path = base + "-summary.json"
    with open(summary_path, "w") as f:
        json.dump({"project": args.project, "tag": args.tag, "sample_seed": args.sample_seed,
                   "seeds": len(sample), "pairs": args.pairs, "pre_analysis": pre,
                   "strategies": summary, "time": stamp}, f, indent=1)

    # Append to the per-project history so the week reads as a series.
    hist = os.path.join(args.out_dir, "history.tsv")
    new = not os.path.exists(hist)
    with open(hist, "a") as f:
        if new:
            f.write("time\tproject\ttag\tsample_seed\tseeds\tpairs\t" +
                    "\t".join(f"{k}_rate" for k in ("dataflow", "state", "declaration", "combined")) + "\n")
        f.write(f"{stamp}\t{args.project}\t{args.tag}\t{args.sample_seed}\t{len(sample)}\t{args.pairs}\t" +
                "\t".join(str(summary.get(k, {}).get("rate", "")) for k in ("dataflow", "state", "declaration", "combined")) + "\n")

    print(f"\n[done] records: {records_path}\n       summary: {summary_path}")
    degradations.report()


if __name__ == "__main__":
    main()
