"""
Best-effort triage helper: groups fuzzer-found crashes in output/bugs/<project>/
by their (recomputed) crash signature, then searches the upstream GitHub
repo's issue tracker for each distinct signature so a human can check for
existing reports before filing new ones.

This does NOT auto-classify duplicates — GitHub's issue search is a keyword
match, not a semantic one, so results are candidates for a human to read and
confirm, not a verdict. Output is a Markdown report, nothing is filed/closed.

Usage:
    python3 -m core.github_dedupe --project clang --repo llvm/llvm-project
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from core.reproduce_check import extract_frames

GITHUB_API = "https://api.github.com"


def _load_driver_class(project_name):
    """Dynamically load the project's driver module and return its
    BaseDriver subclass (uninitialised — we only need extract_crash_signature,
    which is a pure function of (stdout, stderr, return_code))."""
    import inspect
    from core.driver import BaseDriver

    driver_path = os.path.join("projects", project_name, "driver.py")
    spec = importlib.util.spec_from_file_location(f"ffl_{project_name}_driver", driver_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for _, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and issubclass(obj, BaseDriver) and obj is not BaseDriver:
            return obj
    raise RuntimeError(f"No BaseDriver subclass found in {driver_path}")


def cluster_bug_dirs(bugs_dir, driver_cls):
    """Re-derive the canonical signature for every bug folder's test.out
    (using the *current* extract_crash_signature, which may be newer/fixer
    than whatever produced the folder name) and group folders that collapse
    to the same signature."""
    drv = driver_cls.__new__(driver_cls)
    drv.config = {"analysis": {"crash_patterns": []}}

    clusters = {}
    for name in sorted(os.listdir(bugs_dir)):
        out_path = os.path.join(bugs_dir, name, "test.out")
        if not os.path.isfile(out_path):
            continue
        text = open(out_path, encoding="utf-8", errors="replace").read()
        sig = drv.extract_crash_signature("", text, 1) or f"(unrecognized) {name}"
        clusters.setdefault(sig, []).append(name)
    return clusters


# ── Search-query construction ──────────────────────────────────────────

_FRAME_SHORT_NAME_RE = re.compile(r'([A-Za-z_~][A-Za-z0-9_]*)$')
# clang's own crash-dump printer boilerplate — appears in ~every parser-stage
# crash report ever pasted into an issue, so it's a useless search term alone.
_GENERIC_BOILERPLATE_RE = re.compile(
    r'current parser token|at annotation token|<eof> parser at end of file|'
    r'^Program arguments'
)


def _short_name(qualified: str) -> str:
    m = _FRAME_SHORT_NAME_RE.search(qualified.split('::')[-1])
    return m.group(1) if m else qualified


def _stack_dump_message(signature: str) -> str:
    """Return the crash-site message portion of a 'Stack dump: ...'
    signature, or '' if the signature is a bare (unbracketed) frame chain
    with no message captured — see extract_frames() for why unbracketed
    frame chains happen and why they must not be treated as a message."""
    m = re.match(r'^Stack dump:\s*(.*?)\s*\[.*\]\s*$', signature)
    if m:
        return m.group(1).strip()
    rest = signature[len("Stack dump:"):].strip()
    if '::' in rest or ' > ' in rest:
        return ''
    return rest


def build_search_terms(signature: str):
    """Return an ordered list of candidate query strings (most specific
    first) to try against GitHub's issue search for a given signature."""
    m = re.match(r'^(ASAN|UBSAN|LLVM ERROR|Assertion):\s*(.*)$', signature)
    if m:
        kind, detail = m.groups()
        detail = detail.strip()
        words = detail.split()
        terms = []
        if len(words) >= 2:
            # LLVM assert messages routinely embed their own literal quotes
            # (the `assert(cond && "message")` idiom, e.g. Assertion:
            # isa<To>(Val) && "cast<Ty>() argument of incompatible type!").
            # Strip them before wrapping in our own quotes, or the nested
            # quote closes the phrase early and corrupts the query.
            phrase = " ".join(words[:8]).replace('"', '')
            if phrase:
                terms.append('"' + phrase + '"')
        if words:
            terms.append(words[0].strip('`\'".,;:()'))
        return terms or [detail]

    if signature.startswith("Stack dump:"):
        frames = extract_frames(signature)
        terms = []
        # Most specific first: combined 2-frame query narrows down a single
        # common frame name (e.g. "getASTContext") to the actual call site.
        if len(frames) > 1:
            terms.append('"' + _short_name(frames[0]) + '" "' + _short_name(frames[1]) + '"')
        if frames:
            terms.append('"' + _short_name(frames[0]) + '"')
        # The crash-site "message" (current parser token 'X', at annotation
        # token, <eof> parser at end of file, ...) is clang's own generic
        # crash-dump boilerplate — it shows up in nearly every parser-stage
        # crash report ever pasted into an issue, so it's not a useful
        # search term on its own. Only fall back to it when we have no
        # frame names at all.
        message = _stack_dump_message(signature)
        if not frames and message and not _GENERIC_BOILERPLATE_RE.search(message):
            terms.append('"' + message + '"')
        return terms

    if signature in ("Segmentation fault", "Aborted"):
        return []  # no distinctive text to search on

    return ['"' + signature[:80] + '"']


# ── GitHub search ───────────────────────────────────────────────────────

def _gh_get(url, token=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "fusion-fuzz-dedupe"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_issues(query, repo, max_results=5, token=None, retries=3):
    q = f"repo:{repo} is:issue {query}"
    url = f"{GITHUB_API}/search/issues?q={urllib.parse.quote(q)}&per_page={max_results}"
    for attempt in range(retries):
        try:
            data = _gh_get(url, token=token)
            items = [
                {"title": it["title"], "url": it["html_url"],
                 "state": it["state"], "number": it["number"]}
                for it in data.get("items", [])
            ]
            return data.get("total_count", 0), items
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < retries - 1:
                time.sleep(10)
                continue
            return None, []
        except Exception:
            return None, []
    return None, []


# ── Report ──────────────────────────────────────────────────────────────

def run(project, repo, bugs_dir=None, max_results=5, out_path=None, sleep_between=6.5):
    bugs_dir = bugs_dir or os.path.join("output", "bugs", project)
    if not os.path.isdir(bugs_dir):
        print(f"No bug directory at {bugs_dir}")
        return

    driver_cls = _load_driver_class(project)
    clusters = cluster_bug_dirs(bugs_dir, driver_cls)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    print(f"{sum(len(v) for v in clusters.values())} bug folders -> "
          f"{len(clusters)} distinct signatures. Searching {repo}...")

    lines = [f"# Dedup report — {project} vs {repo}", ""]
    for i, (sig, folders) in enumerate(
            sorted(clusters.items(), key=lambda kv: -len(kv[1])), start=1):
        lines.append(f"## {i}. `{sig}`")
        lines.append(f"- Local instances ({len(folders)}): {', '.join(folders[:6])}"
                      + (f", … +{len(folders)-6} more" if len(folders) > 6 else ""))

        terms = build_search_terms(sig)
        if not terms:
            lines.append("- No distinctive search terms (generic signal-only crash — "
                          "needs a reduced/minimized reproducer before it's searchable).")
            lines.append("")
            continue

        # Try queries most-specific-first; stop as soon as one lands in a
        # band that's actually reviewable (1-50 hits). If every term is
        # either 0 or too broad, report the narrowest one we saw, flagged
        # as weak signal rather than silently picking an arbitrary term.
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

        if best is None:
            lines.append(f"- Query: `{terms[0]}`")
            lines.append("- ⚠️ Search failed (rate-limited or network error) — retry later.")
        else:
            total, items, query = best
            lines.append(f"- Query: `{query}`")
            if total == 0:
                lines.append("- No matching issues found — looks like a candidate for a new report.")
            else:
                caveat = "" if total <= 50 else " (⚠️ broad match, low precision — review carefully)"
                lines.append(f"- {total} potentially related issue(s){caveat} "
                              f"(top {len(items)} shown, review before treating as a duplicate):")
                for it in items:
                    lines.append(f"  - [{it['state']}] #{it['number']} {it['title']} — {it['url']}")
        lines.append("")

    report = "\n".join(lines)
    out_path = out_path or os.path.join(bugs_dir, "DEDUP_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo", default="llvm/llvm-project")
    ap.add_argument("--bugs-dir", default=None)
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.project, args.repo, bugs_dir=args.bugs_dir,
        max_results=args.max_results, out_path=args.out)
