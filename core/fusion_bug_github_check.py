"""
(i) Find fusion-specific bugs in output/bugs/<project>/ — crashes that do
    NOT reproduce from either parent alone (reuses core.reproduce_check).
(ii) For each one, query GitHub issue search using its signature's top
    (up to 3) fully-qualified stack frames, quoted, e.g.:

    is:issue "Stack dump:" "clang::InitializationSequence::Perform"
              "clang::Sema::AddInitializerToDecl"
              "clang::Parser::ParseDeclarationAfterDeclaratorAndAttributes"

    and reports how many matching issues are open vs closed, plus a
    clickable github.com/<repo>/issues?q=... URL for manual review.

Must run inside the fuzz-clang container (needs the real clang toolchain
for phase (i)) with network access to api.github.com for phase (ii).

Usage:
    python3 -m core.fusion_bug_github_check --project clang --repo llvm/llvm-project
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from core.reproduce_check import _load_driver, check_bug_dir, extract_frames

GITHUB_API = "https://api.github.com"


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


def _gh_get(url, token=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "fusion-fuzz-dedupe"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_github(frames, repo, token=None, retries=3):
    """Phase (ii) for one bug. Returns a dict with the browsable web URL
    (matching the github.com/<repo>/issues?q=... format) plus open/closed
    counts and a few sample matches. total=None on search failure."""
    terms = ['"Stack dump:"'] + [f'"{f}"' for f in frames]
    web_query = " ".join(["is:issue"] + terms)
    web_url = f"https://github.com/{repo}/issues?q=" + urllib.parse.quote(web_query, safe='')

    api_query = f"repo:{repo} " + web_query
    api_url = f"{GITHUB_API}/search/issues?q={urllib.parse.quote(api_query, safe='')}&per_page=100"

    for attempt in range(retries):
        try:
            data = _gh_get(api_url, token=token)
            items = data.get("items", [])
            total = data.get("total_count", 0)
            return {
                "web_url": web_url,
                "total": total,
                "open": sum(1 for it in items if it["state"] == "open"),
                "closed": sum(1 for it in items if it["state"] == "closed"),
                "truncated": total > len(items),
                "items": items,
            }
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < retries - 1:
                time.sleep(10)
                continue
            return {"web_url": web_url, "total": None}
        except Exception:
            return {"web_url": web_url, "total": None}
    return {"web_url": web_url, "total": None}


def run(project, repo, bugs_dir=None, timeout=10, sleep_between=6.5, out_path=None):
    bugs_dir = bugs_dir or os.path.join("output", "bugs", project)
    driver = _load_driver(project)

    print(f"Phase 1: scanning {bugs_dir} for fusion-specific bugs...")
    fusion_specific = find_fusion_specific_bugs(bugs_dir, driver, timeout=timeout)
    print(f"\n{len(fusion_specific)} fusion-specific bugs found.\n")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    print(f"Phase 2: querying {repo} for each ({len(fusion_specific)} bugs, "
          f"~{len(fusion_specific) * sleep_between:.0f}s)...\n")

    rows = []
    total_open = total_closed = 0
    for i, (name, sig) in enumerate(fusion_specific, 1):
        frames = extract_frames(sig)
        print(f"  [{i}/{len(fusion_specific)}] {name}")
        if not frames:
            rows.append((name, sig, None))
            print("    -> no usable frame names, skipping search")
            continue
        res = query_github(frames, repo, token=token)
        rows.append((name, sig, res))
        if res.get("total") is None:
            print("    -> search failed")
        else:
            total_open += res["open"]
            total_closed += res["closed"]
            print(f"    -> {res['total']} matches: {res['open']} open, {res['closed']} closed")
        time.sleep(sleep_between)

    lines = [
        f"# Fusion-specific bugs vs {repo}", "",
        f"{len(fusion_specific)} bugs reproduce only with the fused program "
        f"(neither parent alone triggers them).",
        f"Aggregate across all successfully-queried bugs: "
        f"{total_open} open, {total_closed} closed matching issues.", "",
    ]
    for i, (name, sig, res) in enumerate(rows, 1):
        lines.append(f"## {i}. `{name}`")
        lines.append(f"- Signature: `{sig}`")
        if res is None:
            lines.append("- No usable frame names (unsymbolized or message-only crash) — "
                          "cannot build a precise query.")
            lines.append("")
            continue
        lines.append(f"- Query URL: {res['web_url']}")
        if res.get("total") is None:
            lines.append("- ⚠️ Search failed (rate-limited or network error) — retry later.")
        else:
            trunc = " (100+ truncated)" if res["truncated"] else ""
            lines.append(f"- {res['total']} matching issue(s){trunc}: "
                          f"**{res['open']} open, {res['closed']} closed**")
            for it in res["items"][:5]:
                lines.append(f"  - [{it['state']}] #{it['number']} {it['title']} — {it['html_url']}")
        lines.append("")

    report = "\n".join(lines)
    out_path = out_path or os.path.join(bugs_dir, "FUSION_SPECIFIC_GITHUB_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")
    print(f"Totals: {total_open} open, {total_closed} closed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo", default="llvm/llvm-project")
    ap.add_argument("--bugs-dir", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.project, args.repo, bugs_dir=args.bugs_dir, timeout=args.timeout, out_path=args.out)
