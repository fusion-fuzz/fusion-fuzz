"""
(i) Find fusion-specific bugs in output/bugs/<project>/ — crashes that do
    NOT reproduce from either parent alone (reuses core.reproduce_check).
(ii) For each one, build GitHub issue search terms from its crash signature
    — Stack dump frames, Assertion condition text, ASAN/UBSAN/LLVM ERROR
    detail, ... (reuses core.github_dedupe.build_search_terms, so any
    signature kind it knows how to query is covered here too) — and query
    the upstream tracker, trying terms most-specific-first exactly like
    core.github_dedupe.run() does. Reports how many matching issues are
    open vs closed (an OPEN match usually means the bug is already tracked
    upstream — check it before filing a new report) plus a clickable
    github.com/<repo>/issues?q=... URL for manual review.

Must run inside the fuzz-clang container (needs the real clang toolchain
for phase (i)) with network access to api.github.com for phase (ii).

Usage:
    python3 -m core.fusion_bug_github_check --project clang --repo llvm/llvm-project
"""
import argparse
import os
import time
import urllib.parse

from core.github_dedupe import build_search_terms, search_issues
from core.reproduce_check import _load_driver, check_bug_dir


def find_fusion_specific_bugs(bugs_dir, driver, timeout=10):
    """Phase (i): returns [(bug_dir_name, signature), ...] for crashes that
    reproduce with the fused test but with neither parent alone."""
    bug_dirs = sorted(
        d for d in os.listdir(bugs_dir)
        if os.path.isdir(os.path.join(bugs_dir, d))
    )

    fusion_specific = []
    for i, name in enumerate(bug_dirs, 1):
        path = os.path.join(bugs_dir, name)
        res = check_bug_dir(path, driver, timeout=timeout)
        status = "skipped"
        if res is not None and res["fused_reproduced"]:
            reproduces_in_parent = any(
                info["same_signature_as_fused"] for info in res["parents"].values()
            )
            if reproduces_in_parent:
                status = "reproduces in parent"
            else:
                status = "fusion-specific"
                fusion_specific.append((name, res["fused_signature"]))
        print(f"  [{i}/{len(bug_dirs)}] {name}: {status}")

    return fusion_specific


def _web_url(query, repo):
    return f"https://github.com/{repo}/issues?q=" + urllib.parse.quote(f"is:issue {query}", safe='')


def query_github_best(terms, repo, token=None, max_results=5, sleep_between=6.5):
    """Try each candidate query (most-specific first, as ordered by
    build_search_terms), stopping at the first that lands in a reviewable
    band (1-50 hits) — mirrors core.github_dedupe.run()'s selection logic."""
    best = None  # (total, items, query)
    for query in terms:
        total, items = search_issues(query, repo, max_results=max_results, token=token)
        time.sleep(sleep_between)  # stay under GitHub's unauthenticated search rate limit
        if total is None:
            continue
        if total == 0:
            best = best or (total, items, query)
            continue
        if total <= 50:
            best = (total, items, query)
            break
        if best is None or total < best[0]:
            best = (total, items, query)
    return best


def run(project, repo, bugs_dir=None, timeout=10, sleep_between=6.5, out_path=None):
    bugs_dir = bugs_dir or os.path.join("output", "bugs", project)
    driver = _load_driver(project)

    print(f"Phase 1: scanning {bugs_dir} for fusion-specific bugs...")
    fusion_specific = find_fusion_specific_bugs(bugs_dir, driver, timeout=timeout)
    print(f"\n{len(fusion_specific)} fusion-specific bugs found.\n")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    print(f"Phase 2: querying {repo} for each ({len(fusion_specific)} bugs)...\n")

    rows = []
    total_open = total_closed = 0
    open_matches = 0
    for i, (name, sig) in enumerate(fusion_specific, 1):
        print(f"  [{i}/{len(fusion_specific)}] {name}")
        terms = build_search_terms(sig)
        if not terms:
            rows.append((name, sig, "no_terms", None))
            print("    -> no usable search terms, skipping search")
            continue

        best = query_github_best(terms, repo, token=token, sleep_between=sleep_between)
        if best is None:
            rows.append((name, sig, "search_failed", None))
            print("    -> search failed")
            continue

        total, items, query = best
        opened = sum(1 for it in items if it["state"] == "open")
        closed = sum(1 for it in items if it["state"] == "closed")
        res = {
            "total": total, "items": items, "open": opened, "closed": closed,
            "web_url": _web_url(query, repo),
        }
        rows.append((name, sig, query, res))
        total_open += opened
        total_closed += closed
        if opened:
            open_matches += 1
        print(f"    -> query `{query}`: {total} matches: {opened} open, {closed} closed")

    lines = [
        f"# Fusion-specific bugs vs {repo}", "",
        f"{len(fusion_specific)} bugs reproduce only with the fused program "
        f"(neither parent alone triggers them).",
        f"{open_matches} of them have at least one OPEN matching issue upstream "
        f"(likely already tracked — check before filing a new report).",
        f"Aggregate across all successfully-queried bugs: "
        f"{total_open} open, {total_closed} closed matching issues.", "",
    ]
    for i, (name, sig, query, res) in enumerate(rows, 1):
        lines.append(f"## {i}. `{name}`")
        lines.append(f"- Signature: `{sig}`")
        if res is None:
            reason = ("search failed (rate-limited or network error) — retry later" if query == "search_failed"
                      else "no usable search terms (generic signal-only crash — needs a "
                           "reduced/minimized reproducer before it's searchable)")
            lines.append(f"- ⚠️ {reason}.")
            lines.append("")
            continue

        lines.append(f"- Query: `{query}`")
        lines.append(f"- {res['web_url']}")
        if res["total"] == 0:
            lines.append("- No matching issues found — looks like a candidate for a new report.")
        else:
            trunc = " (more not shown)" if res["total"] > len(res["items"]) else ""
            status = ("**OPEN issue exists — likely already tracked upstream**" if res["open"]
                       else "closed issue(s) only — may already be fixed")
            lines.append(f"- {res['total']} matching issue(s){trunc}: "
                          f"{res['open']} open, {res['closed']} closed — {status}")
            for it in res["items"][:5]:
                lines.append(f"  - [{it['state']}] #{it['number']} {it['title']} — {it['url']}")
        lines.append("")

    report = "\n".join(lines)
    out_path = out_path or os.path.join(bugs_dir, "FUSION_SPECIFIC_GITHUB_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")
    print(f"Totals: {total_open} open, {total_closed} closed ({open_matches} bugs have an open match)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo", default="llvm/llvm-project")
    ap.add_argument("--bugs-dir", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.project, args.repo, bugs_dir=args.bugs_dir, timeout=args.timeout, out_path=args.out)
