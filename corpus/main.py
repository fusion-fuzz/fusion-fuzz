#!/usr/bin/env python3
"""
Universal Bug Corpus Manager

Usage:
    python3 ./corpus/main.py --project php                            # legacy: import php tests
    python3 ./corpus/main.py import --project php [--tests-dir PATH]
    python3 ./corpus/main.py list [--project php]
    python3 ./corpus/main.py show <id>
    python3 ./corpus/main.py translate <id> --lang go --file code.go
    python3 ./corpus/main.py translate-llm --target go [--project go] [--source php] [--concurrency 4]
    python3 ./corpus/main.py stats
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
import collections
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing core modules when run from repo root or corpus dir
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CORPUS_DIR = Path(__file__).parent
DEFAULT_DB = CORPUS_DIR / "corpus.db"

# Maps a project to the language family it *consumes*.
#
# The distinction matters and used to be wrong here: naga was mapped to
# "rust" because naga is written in Rust, but naga compiles WGSL — all 193
# of its seeds are WGSL — so translating for it produced Rust that it
# cannot read. Several projects were missing outright (flang, haskell,
# spidermonkey, tint, triton), and project_to_lang then fell through to the
# project name, inventing languages called "tint" and "spidermonkey".
#
# Families, not single languages: gcc and clang both take C and C++, v8 and
# spidermonkey both take JavaScript, tint and naga both take WGSL. That is
# what makes a seed reusable across projects without any translation at all.
PROJECT_FAMILY = {
    "gcc": "c/cpp", "clang": "c/cpp",
    "cpython": "python",
    "php": "phpt",
    "rust": "rust",
    "go": "go",
    "swift": "swift",
    "haskell": "haskell",
    "v8": "javascript", "spidermonkey": "javascript",
    "flang": "fortran", "lfortran": "fortran",
    "mlir": "mlir", "triton": "mlir",
    "tint": "wgsl", "naga": "wgsl",
}

# A family maps back to the language to generate for it. For families with
# several members, the one an LLM is most likely to get right.
FAMILY_LANG = {
    "c/cpp": "c", "python": "python", "phpt": "php", "rust": "rust",
    "go": "go", "swift": "swift", "haskell": "haskell",
    "javascript": "javascript", "fortran": "fortran", "mlir": "mlir",
    "wgsl": "wgsl",
}

# Kept only so an existing translations JSON stays readable.
LANG_MAP = dict(PROJECT_FAMILY)


def project_to_family(project: str) -> str:
    """The language family a project consumes."""
    p = (project or "").lower()
    if p in PROJECT_FAMILY:
        return PROJECT_FAMILY[p]
    # --target may name a language rather than a project.
    if p in FAMILY_LANG:
        return p
    for fam, lang in FAMILY_LANG.items():
        if lang == p:
            return fam
    return p


def project_to_lang(project: str) -> str:
    """The language to generate for this project or family."""
    return FAMILY_LANG.get(project_to_family(project), project_to_family(project))


def family_to_project(family: str) -> str:
    """A project whose config.yaml can supply settings for this family.

    A family is not a project: the fortran family is served by `flang` and
    `lfortran`, wgsl by `tint` and `naga`, and there is no projects/fortran
    to load a config from. Auto mode picks a family, so it has to come back
    to a real directory here or every target but mlir — which happens to
    share its name with a project — fails to start."""
    # Absolute, via _REPO_ROOT: this is called before the chdir below, and
    # the tool is normally run from inside corpus/, where a relative
    # "projects/..." does not exist. Resolving it relatively made every
    # family fall through to its own name and fail to load a config.
    members = [p for p, f in PROJECT_FAMILY.items() if f == family]
    for name in members:
        if (_REPO_ROOT / "projects" / name / "config.yaml").exists():
            return name
    for name in members:                       # exists but has no config
        if (_REPO_ROOT / "projects" / name).is_dir():
            return name
    return family


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project      TEXT    NOT NULL,
            name         TEXT,
            program      TEXT    NOT NULL,
            translations TEXT    NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON corpus(project)")
    conn.commit()
    return conn


def insert_program(conn: sqlite3.Connection, project: str, name: str, program: str) -> int:
    cur = conn.execute(
        "INSERT INTO corpus (project, name, program, translations) VALUES (?, ?, ?, ?)",
        (project, name, program, "{}"),
    )
    conn.commit()
    return cur.lastrowid


def program_exists(conn: sqlite3.Connection, project: str, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM corpus WHERE project = ? AND name = ?", (project, name)
    ).fetchone()
    return row is not None


def set_translation(conn: sqlite3.Connection, row_id: int, target_lang: str, code: str):
    row = conn.execute("SELECT translations FROM corpus WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        raise ValueError(f"No corpus entry with id={row_id}")
    translations = json.loads(row["translations"])
    translations[target_lang] = code
    conn.execute(
        "UPDATE corpus SET translations = ? WHERE id = ?",
        (json.dumps(translations), row_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Parsers (one per project type)
# ---------------------------------------------------------------------------

def parse_js(path: Path) -> str | None:
    """Read a JavaScript test file as-is."""
    text = path.read_text(errors="replace").strip()
    return text if text else None


def parse_phpt(path: Path) -> str | None:
    """Extract the --FILE-- section from a .phpt test file."""
    text = path.read_text(errors="replace")
    in_file = False
    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("--") and stripped.endswith("--"):
            section = stripped[2:-2]
            if section == "FILE":
                in_file = True
                continue
            elif in_file:
                break
        if in_file:
            lines.append(line)
    if not lines:
        return None
    return "".join(lines).strip()


_INFERREDBUGS_MAX_BYTES = 100_000  # skip files larger than 100 KB

def parse_inferredbugs(path: Path) -> str | None:
    """Read an InferredBugs file_before.txt as-is, skipping oversized files."""
    if path.stat().st_size > _INFERREDBUGS_MAX_BYTES:
        return None
    text = path.read_text(errors="replace").strip()
    return text if text else None


def parse_go(path: Path) -> str | None:
    """
    Read a Go compiler-test file from the Go issue corpus.

    Each file begins with a run-mode directive comment (e.g. ``// run``,
    ``// compile``, ``// errorcheck``).  Two categories are excluded:

    * ``// skip`` — explicitly marked as not applicable.
    * Directory-batch stubs — files whose only Go declaration is a bare
      ``package <name>`` with no functions, types, vars, or consts below it.
      These are partial pieces of multi-file ``compiledir``/``rundir`` tests
      and are not meaningful as standalone programs.

    Everything else (run, compile, errorcheck, build, …) is returned as-is;
    the content is already valid Go source.
    """
    text = path.read_text(errors="replace").strip()
    if not text:
        return None

    # Check first non-empty line for the run-mode directive.
    first_line = text.splitlines()[0].strip()
    # Skip directives that start with "// skip"
    if re.match(r"^//\s*skip\b", first_line, re.IGNORECASE):
        return None

    # Detect stub-only files: only a package declaration remains after
    # stripping comments and blank lines (no funcs, types, vars, consts).
    non_comment_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.strip().startswith("//")
    ]
    if not non_comment_lines:
        return None
    # A stub has exactly one meaningful line: the package declaration.
    if len(non_comment_lines) == 1 and re.match(r"^package\s+\w+", non_comment_lines[0].strip()):
        return None

    return text


# Registry: project name -> (glob pattern, parser function)
PROJECT_PARSERS: dict[str, tuple[str, callable]] = {
    "php":               ("*.phpt",              parse_phpt),
    "v8":                ("**/*.js",              parse_js),
    "go":                ("*.go",                 parse_go),
    "inferredbugs-java": ("**/file_before.txt",   parse_inferredbugs),
    "inferredbugs-csharp": ("**/file_before.txt", parse_inferredbugs),
}

INFERREDBUGS_ROOT = CORPUS_DIR / "InferredBugs" / "inferredbugs"


# ---------------------------------------------------------------------------
# Import command
# ---------------------------------------------------------------------------

def cmd_import(args, conn: sqlite3.Connection):
    project = args.project
    if project not in PROJECT_PARSERS:
        sys.exit(
            f"Unknown project '{project}'. Supported: {', '.join(PROJECT_PARSERS)}"
        )

    glob_pattern, parser = PROJECT_PARSERS[project]
    # Default source directory: corpus/<project>/tests for most projects,
    # but corpus/go/ for Go (files live directly in the project folder).
    if args.tests_dir:
        tests_dir = Path(args.tests_dir)
    elif project == "go":
        tests_dir = CORPUS_DIR / "go"
    elif project == "inferredbugs-java":
        tests_dir = INFERREDBUGS_ROOT / "java"
    elif project == "inferredbugs-csharp":
        tests_dir = INFERREDBUGS_ROOT / "csharp"
    else:
        tests_dir = CORPUS_DIR / project / "tests"
    if not tests_dir.exists():
        sys.exit(f"Tests directory not found: {tests_dir}")

    files = sorted(tests_dir.glob(glob_pattern))
    if not files:
        sys.exit(f"No files matching '{glob_pattern}' in {tests_dir}")

    added = skipped = failed = 0
    for f in files:
        # Use relative path as name to avoid collisions in recursive globs
        try:
            name = str(f.relative_to(tests_dir))
        except ValueError:
            name = f.name
        if program_exists(conn, project, name):
            skipped += 1
            continue
        try:
            program = parser(f)
        except Exception as e:
            print(f"  WARN: failed to parse {name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if program is None:
            print(f"  WARN: no program content in {name}", file=sys.stderr)
            failed += 1
            continue
        row_id = insert_program(conn, project, name, program)
        added += 1
        if args.verbose:
            print(f"  [{row_id}] {name}")

    print(f"Import complete: {added} added, {skipped} skipped (already exists), {failed} failed")


# ---------------------------------------------------------------------------
# List command
# ---------------------------------------------------------------------------

def cmd_list(args, conn: sqlite3.Connection):
    if args.project:
        rows = conn.execute(
            "SELECT id, project, name, length(program) as prog_len FROM corpus WHERE project = ? ORDER BY id",
            (args.project,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, project, name, length(program) as prog_len FROM corpus ORDER BY id"
        ).fetchall()

    if not rows:
        print("No entries found.")
        return

    fmt = "{:<6} {:<12} {:<50} {:>10}"
    print(fmt.format("ID", "PROJECT", "NAME", "PROG_BYTES"))
    print("-" * 82)
    for r in rows:
        print(fmt.format(r["id"], r["project"], r["name"] or "", r["prog_len"]))
    print(f"\nTotal: {len(rows)} entries")


# ---------------------------------------------------------------------------
# Show command
# ---------------------------------------------------------------------------

def cmd_show(args, conn: sqlite3.Connection):
    row = conn.execute("SELECT * FROM corpus WHERE id = ?", (args.id,)).fetchone()
    if row is None:
        sys.exit(f"No entry with id={args.id}")

    translations = json.loads(row["translations"])
    print(f"=== ID: {row['id']} | Project: {row['project']} | Name: {row['name']} ===")
    print()
    print("--- program ---")
    print(row["program"])
    if translations:
        for lang, code in translations.items():
            print(f"\n--- translation: {lang} ---")
            print(code)
    else:
        print("\n(no translations yet)")


# ---------------------------------------------------------------------------
# Translate (manual, single entry) command
# ---------------------------------------------------------------------------

def cmd_translate(args, conn: sqlite3.Connection):
    translation_file = Path(args.file)
    if not translation_file.exists():
        sys.exit(f"File not found: {translation_file}")
    code = translation_file.read_text(errors="replace").strip()
    set_translation(conn, args.id, args.lang, code)
    print(f"Translation '{args.lang}' saved for entry id={args.id}")


# ---------------------------------------------------------------------------
# Translate-LLM (batch LLM translation + --refine mode) command
# ---------------------------------------------------------------------------

def _has_column(conn: sqlite3.Connection, column: str) -> bool:
    return any(r[1] == column for r in conn.execute("PRAGMA table_info(corpus)"))


def _source_lang(row) -> str:
    """The language a seed is written in.

    The `language` column when it exists, and only then the contributing
    project. `project` says where a seed came from, not what it is: 13.2% of
    the corpus disagrees with what project_to_lang would infer — 3782 seeds
    filed under `clang` are PHP, 3246 under `php` are JavaScript. Telling the
    model to "translate this C program" over a PHP body is a wasted call."""
    try:
        lang = row["language"]
    except (IndexError, KeyError):
        lang = None
    return lang or project_to_lang(row["project"])


# Statuses worth trying again. 429 is the rate limit; 5xx and a bare
# connection failure are the server's problem, not the request's. A 400 or
# 401 will fail identically however many times it is sent.
_RETRY_STATUS = {429, 500, 502, 503, 504, 408, 409, 529}


class _Backoff:
    """A cooldown shared by every worker thread.

    Rate limiting is a property of the account, not of one request, so one
    thread meeting a 429 has to slow all of them down. Backing off only the
    thread that was refused leaves the other N-1 hammering the endpoint,
    which is what turns a brief limit into a sustained one.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._until = 0.0
        self.hits = 0

    def wait(self):
        while True:
            with self._lock:
                remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 2.0))

    def penalise(self, seconds):
        with self._lock:
            self.hits += 1
            self._until = max(self._until, time.monotonic() + seconds)


_BACKOFF = _Backoff()

# Set once per run by _run_auto. None means no accounting.
_BUDGET = None


class BudgetExhausted(Exception):
    """Raised to stop a run that has spent its token allowance."""



class _Progress:
    """Live progress for a translation batch.

    Reports on a timer rather than every N completions: at a few items per
    minute — which is what a serialising endpoint gives — a count-based
    report says nothing for ten minutes at a time. Every line is flushed,
    because stdout is a pipe whenever this is run under nohup and Python
    then buffers 8KB of it, which is more than a whole run produces.

    The rate shown is over a recent window, not since the start. A
    cumulative average hides exactly what one wants to see: that throughput
    has collapsed because requests are queueing behind each other.
    """

    WINDOW = 120.0          # seconds of history behind the "recent" rate

    def __init__(self, total, label, every=10.0):
        self.total = total
        self.label = label
        self.every = every
        self.start = time.monotonic()
        self.last_report = 0.0
        self.done = self.saved = self.failed = 0
        self.recent = collections.deque()      # completion timestamps
        self._lock = threading.Lock()

    def tick(self, saved):
        with self._lock:
            now = time.monotonic()
            self.done += 1
            if saved:
                self.saved += 1
            else:
                self.failed += 1
            self.recent.append(now)
            cutoff = now - self.WINDOW
            while self.recent and self.recent[0] < cutoff:
                self.recent.popleft()
            due = (now - self.last_report >= self.every) or self.done == self.total
            if due:
                self.last_report = now
                line = self._line(now)
            else:
                line = None
        if line:
            print(line, flush=True)

    def _line(self, now):
        elapsed = now - self.start
        span = min(elapsed, self.WINDOW) or 1e-9
        recent_rate = len(self.recent) / span * 60.0      # per minute
        overall = self.done / elapsed * 60.0 if elapsed else 0.0
        left = self.total - self.done
        if recent_rate > 0:
            eta = left / recent_rate
            eta_s = f"{eta:5.1f}m" if eta < 90 else f"{eta/60:5.1f}h"
        else:
            eta_s = "  n/a"
        rl = f" rate-limited={_BACKOFF.hits}" if _BACKOFF.hits else ""
        if _BUDGET is not None and _BUDGET.limit:
            rl += f" | tok {_BUDGET.used/1e6:.2f}M/{_BUDGET.limit/1e6:.1f}M"
        return (f"  [{self.label}] {self.done}/{self.total} "
                f"ok={self.saved} fail={self.failed}{rl} | "
                f"{recent_rate:5.1f}/min (avg {overall:4.1f}) | "
                f"elapsed {elapsed/60:5.1f}m eta {eta_s}")

    def final(self):
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed * 60.0 if elapsed else 0.0
        print(f"  [{self.label}] finished {self.done} in {elapsed/60:.1f}m "
              f"({rate:.1f}/min): {self.saved} saved, {self.failed} failed",
              flush=True)


def _translate_one(row_id: int, name: str, program: str, src_lang: str,
                   tgt_lang: str, config: dict) -> tuple[int, str | None]:
    """Worker: translate a single program. Returns (row_id, code | None).

    Retries transient failures. Without this, concurrency against a hosted
    API is self-defeating: llmgen returns None for every failure alike, so a
    rate-limited request was indistinguishable from a rejected one and the
    seed was dropped after a single attempt.
    """
    from core.llmgen import LLMGenerator

    tr = config.get("_corpus_translation", {}) or {}
    attempts = int(tr.get("max_retries", 4))
    base = float(tr.get("backoff_base", 2.0))

    llm = LLMGenerator(config)
    for attempt in range(attempts):
        # Checked before the call, not after: the allowance has to stop the
        # next request, and a request already sent is already charged.
        if _BUDGET is not None and _BUDGET.exhausted():
            raise BudgetExhausted()
        _BACKOFF.wait()
        try:
            llm.last_error = llm.last_status = None
            llm.last_usage = None
            seed = llm.translate(program, src_lang, tgt_lang)
            if _BUDGET is not None:
                out = seed.content if (seed and seed.content) else ""
                _BUDGET.charge(getattr(llm, "last_usage", None),
                               prompt_chars=len(program) + 400,
                               completion_chars=len(out))
            if seed and seed.content:
                return row_id, seed.content
        except Exception as e:                      # llmgen mostly swallows
            llm.last_error, llm.last_status = e, getattr(e, "status_code", None)

        status = getattr(llm, "last_status", None)
        err = getattr(llm, "last_error", None)
        transient = status in _RETRY_STATUS or (status is None and err is not None)
        if not transient or attempt == attempts - 1:
            if err:
                print(f"  WARN [{name}]: {status or ''} {err}", file=sys.stderr)
            return row_id, None

        # Respect Retry-After when the server sends one, else exponential
        # with jitter so the threads do not all resume together.
        delay = getattr(llm, "retry_after", None) or base ** attempt
        delay += random.uniform(0, delay * 0.25)
        if status == 429:
            _BACKOFF.penalise(delay)
        time.sleep(delay)
    return row_id, None


def _success_refine_one(row_id: int, name: str, program: str, src_lang: str,
                        current_code: str, tgt_lang: str,
                        config: dict) -> tuple[int, str | None, str]:
    """
    Worker: run current_code through the driver; if it fails, re-translate from
    program using the error output as context.
    Returns (row_id, new_code | None, status) where status is one of:
      "ok"       – translation already passes the driver (nothing to do)
      "improved" – was failing; LLM produced a new version
      "failed"   – was failing; LLM produced nothing
      "error"    – unexpected exception
    """
    try:
        from core.driver import get_driver
        from core.fusion import Seed
        from core.llmgen import LLMGenerator

        driver = get_driver(config)
        result = driver.execute(Seed(content=current_code))

        if result.return_code == 0:
            return row_id, None, "ok"

        error_parts = [p.strip() for p in (result.stderr, result.stdout) if p and p.strip()]
        error_msg = "\n".join(error_parts)[:2000]

        llm = LLMGenerator(config)
        new_seed = llm.translate(program, src_lang, tgt_lang, previous_error=error_msg)
        if new_seed and new_seed.content:
            return row_id, new_seed.content, "improved"
        return row_id, None, "failed"
    except Exception as e:
        print(f"  WARN [{name}]: {e}", file=sys.stderr)
        return row_id, None, "error"


def _refine_one(row_id: int, name: str, code: str, lang: str,
                avoid: list[str], extra: str, config: dict) -> tuple[int, str | None]:
    """Worker: refine a single translation. Returns (row_id, refined_code | None)."""
    try:
        from core.llmgen import LLMGenerator
        llm = LLMGenerator(config)
        seed = llm.refine(code, lang, avoid=avoid, extra_constraints=extra)
        if seed and seed.content:
            return row_id, seed.content
    except Exception as e:
        print(f"  WARN [{name}]: {e}", file=sys.stderr)
    return row_id, None


def _run_auto(conn, args, config, cfg_project):
    """Translate into whichever family needs seeds most, repeatedly.

    Each round re-picks the target, so the corpus levels up rather than
    pouring everything into one language: after MLIR passes Fortran, the
    next round goes to Fortran. Sources are chosen by
    corpus.seed_selection, which filters out seeds the target language
    cannot express and then spreads the selection across source languages
    and program shapes."""
    from corpus.seed_selection import (pick_target, select_sources,
                                       effective_counts, FAMILY_LANG)

    global _BUDGET
    conn.row_factory = sqlite3.Row
    _tr = config.get("_corpus_translation", {})
    per_round = args.limit or _tr.get("limit") or 500

    from corpus.token_budget import TokenBudget
    limit_tokens = args.token_budget or _tr.get("token_budget") or 0
    _BUDGET = TokenBudget(limit_tokens,
                          persist=bool(_tr.get("token_budget_persist", False)))
    if limit_tokens:
        print(f"  token budget: {_BUDGET.describe()}", flush=True)
    rounds = args.rounds or 1
    grand_saved = grand_failed = 0

    for rnd in range(1, rounds + 1):
        target_family, have = pick_target(conn)
        if target_family is None:
            print("No target family to fill.")
            break
        target_lang = FAMILY_LANG.get(target_family, target_family)
        # `config` was loaded once, from whichever project served the first
        # round's family, and is not reloaded as the target changes. That is
        # fine because everything translation uses — model, concurrency,
        # retries, token budget — comes from corpus/config.yaml; the project
        # config only supplies fallbacks for keys that file does not set.
        # Derive the round's seed rather than reusing one: with a fixed
        # seed every round draws the buckets in the same order, which only
        # differs at all because the previous round's output is filtered
        # out. Deriving keeps the run reproducible and the rounds distinct.
        round_seed = None if args.seed is None else args.seed * 1000 + rnd
        picked, stats = select_sources(conn, target_family, per_round,
                                       seed=round_seed)
        if not picked:
            print(f"Round {rnd}: nothing viable to translate into "
                  f"{target_family}.")
            break

        print(f"Round {rnd}/{rounds}: {target_family} has {have} seeds "
              f"(fewest) -> translating {len(picked)}")
        print(f"  candidates {stats['considered']}, "
              f"dropped as unviable {stats['rejected_unviable']}, "
              f"shape buckets {stats['buckets']}")
        if args.dry_run and rnd > 1:
            print("  (dry-run: later rounds would differ only by what the "
                  "earlier ones wrote; stopping)")
            break
        if args.dry_run:
            langs = Counter(r["language"] for r in picked)
            print("  sources: " + ", ".join(f"{k}={v}" for k, v in langs.most_common()))
            for r in picked[:5]:
                print(f"    [{r['id']}] {r['language']:10s} {r['name'][:52]}")
            print("  (dry-run - nothing translated)")
            continue

        concurrency = (args.concurrency or _tr.get("concurrency")
                       or config.get("execution", {}).get("concurrency", 4))
        prog = _Progress(len(picked), f"{target_lang} r{rnd}",
                         every=float(_tr.get("report_every", 10)))
        print(f"  concurrency={concurrency}", flush=True)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_translate_one, r["id"], r["name"] or str(r["id"]),
                          r["program"], _source_lang(r).capitalize(),
                          target_lang.capitalize(), config): r["id"]
                for r in picked
            }
            try:
                for fut in as_completed(futures):
                    try:
                        rid, code = fut.result()
                        if code:
                            set_translation(conn, rid, target_lang, code)
                            prog.tick(True)
                        else:
                            prog.tick(False)
                    except BudgetExhausted:
                        prog.tick(False)
                    except Exception as e:
                        print(f"  ERROR: {e}", file=sys.stderr, flush=True)
                        prog.tick(False)
            except KeyboardInterrupt:
                print("\nInterrupted.", flush=True)
                prog.final()
                break
        prog.final()
        if _BUDGET is not None and _BUDGET.limit:
            print(f"  {_BUDGET.describe()}", flush=True)
        saved, failed = prog.saved, prog.failed
        if _BUDGET is not None and _BUDGET.exhausted():
            print("  token budget spent — stopping.", flush=True)
            grand_saved += saved
            grand_failed += failed
            break
        grand_saved += saved
        grand_failed += failed

    if not args.dry_run:
        print(f"Done: {grand_saved} translations saved, {grand_failed} failed.")
        counts = effective_counts(conn)
        print("  seeds per family now:")
        for fam, n in sorted(counts.items(), key=lambda x: x[1]):
            print(f"    {fam:<12} {n}")


CORPUS_CONFIG = CORPUS_DIR / "config.yaml"


def load_corpus_config():
    """The corpus-level LLM settings, or {} when the file is absent."""
    if not CORPUS_CONFIG.exists():
        return {}
    try:
        import yaml
        with open(CORPUS_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  warning: could not read {CORPUS_CONFIG}: {e}", file=sys.stderr)
        return {}


def merge_corpus_config(config: dict) -> dict:
    """Overlay corpus/config.yaml onto a project config.

    Translation is a corpus-level job: which model translates should not
    depend on which project happens to own the target language. Only keys
    with a non-empty value override, so leaving `api_key` blank still falls
    through to the project's setting and then to LLM_API_KEY."""
    corpus_cfg = load_corpus_config()
    if not corpus_cfg:
        return config
    merged = dict(config)
    llm = dict(merged.get("llm") or {})
    overrides = {k: v for k, v in (corpus_cfg.get("llm") or {}).items()
                 if v not in ("", None)}
    llm.update(overrides)
    merged["llm"] = llm
    if overrides:
        shown = {k: ("***" if k == "api_key" else v) for k, v in overrides.items()}
        print(f"  LLM from corpus/config.yaml: {shown}")
    merged["_corpus_translation"] = corpus_cfg.get("translation") or {}
    return merged


def cmd_translate_llm(args, conn: sqlite3.Connection):
    """
    Batch-translate corpus entries with an LLM (default), or refine existing
    translations to remove unsuitable patterns (--refine mode).

    Translate mode (default):
        Finds entries that have no translation yet for --target and translates them.

    Refine mode (--refine):
        Finds existing translations that match --filter, then re-prompts the LLM
        to rewrite them without the APIs listed in --avoid.
    """
    if not args.target and not getattr(args, "auto", False):
        sys.exit("--target is required unless --auto is given")
    # In auto mode the target is chosen per round; --project still decides
    # whose LLM config is used, defaulting to the first neediest family.
    if getattr(args, "auto", False) and not args.target:
        from corpus.seed_selection import pick_target, FAMILY_LANG
        conn.row_factory = sqlite3.Row
        fam, _ = pick_target(conn)
        args.target = FAMILY_LANG.get(fam, fam)
    target_lang = project_to_lang(args.target)
    # The config comes from a project directory, and --target may name a
    # family that has no directory of its own.
    cfg_project = args.project or family_to_project(project_to_family(args.target))

    try:
        from core.config_loader import load_project_config
        if not os.path.exists(os.path.join("projects", cfg_project, "config.yaml")):
            os.chdir(str(_REPO_ROOT))
        config = load_project_config(cfg_project)
    except FileNotFoundError as e:
        sys.exit(str(e))

    config = merge_corpus_config(config)
    _tr = config.get("_corpus_translation", {})
    concurrency = (args.concurrency or _tr.get("concurrency")
                   or config.get("execution", {}).get("concurrency", 4))
    source_clause = f"AND project = '{args.source}'" if args.source else ""

    # ------------------------------------------------------------------
    # REFINE MODE
    # ------------------------------------------------------------------
    if args.refine:
        # ---- Success-rate refine (no --filter / --avoid) ----
        if not args.filter and not args.avoid:
            rows = conn.execute(
                f"""
                SELECT id, project, name, program, language,
                       json_extract(translations, '$.{target_lang}') AS code
                FROM   corpus
                WHERE  json_extract(translations, '$.{target_lang}') IS NOT NULL
                       {source_clause}
                ORDER  BY id
                """
            ).fetchall()

            if not rows:
                print(f"No '{target_lang}' translations found. Nothing to do.")
                return

            total = len(rows)
            print(f"Success-rate refine: checking {total} '{target_lang}' translations "
                  f"using project '{cfg_project}' driver + LLM config")

            if args.dry_run:
                for r in rows:
                    print(f"  [{r['id']}] {r['name']}")
                print("(dry-run — no changes made)")
                return

            done = ok_count = improved = failed = 0

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {
                    executor.submit(
                        _success_refine_one,
                        r["id"], r["name"] or str(r["id"]),
                        r["program"],
                        _source_lang(r).capitalize(),
                        r["code"],
                        target_lang.capitalize(),
                        config,
                    ): (r["id"], r["name"])
                    for r in rows
                }
                try:
                    for future in as_completed(future_map):
                        row_id, name = future_map[future]
                        done += 1
                        try:
                            rid, new_code, status = future.result()
                            if status == "ok":
                                ok_count += 1
                                if args.verbose:
                                    print(f"  [{rid}] {name} → already passing")
                            elif status == "improved":
                                set_translation(conn, rid, target_lang, new_code)
                                improved += 1
                                if args.verbose:
                                    print(f"  [{rid}] {name} → improved and saved")
                            else:
                                failed += 1
                                if args.verbose:
                                    print(f"  [{rid}] {name} → {status}", file=sys.stderr)
                        except Exception as e:
                            print(f"  ERROR [{name}]: {e}", file=sys.stderr)
                            failed += 1

                        if done % 20 == 0 or done == total:
                            print(f"  Progress: {done}/{total} "
                                  f"(ok={ok_count}, improved={improved}, failed={failed})")
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")

            print(f"Done: {ok_count} already passing, {improved} improved and saved, "
                  f"{failed} failed out of {done} processed.")
            return

        # ---- Pattern-based refine (requires both --filter and --avoid) ----
        if not args.filter or not args.avoid:
            sys.exit("--refine requires both --filter PATTERN and --avoid NAMES")

        avoid_list = [a.strip() for a in args.avoid.split(",") if a.strip()]
        filter_pat = args.filter

        rows = conn.execute(
            f"""
            SELECT id, project, name,
                   json_extract(translations, '$.{target_lang}') AS code
            FROM   corpus
            WHERE  json_extract(translations, '$.{target_lang}') IS NOT NULL
                   {source_clause}
            ORDER  BY id
            """
        ).fetchall()

        bad_rows = [r for r in rows if filter_pat in (r["code"] or "")]

        if not bad_rows:
            print(f"No '{target_lang}' translations contain '{filter_pat}'. Nothing to do.")
            return

        print(f"Found {len(bad_rows)} translations containing '{filter_pat}'.")
        if args.dry_run:
            for r in bad_rows:
                print(f"  [{r['id']}] {r['name']}")
            print("(dry-run — no changes made)")
            return

        done = saved = failed = 0
        total = len(bad_rows)
        print(f"Refining {total} entries (avoid: {', '.join(avoid_list)}) "
              f"using project '{cfg_project}' LLM config")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(
                    _refine_one,
                    r["id"], r["name"] or str(r["id"]), r["code"],
                    target_lang.capitalize(), avoid_list, args.extra or "", config,
                ): (r["id"], r["name"])
                for r in bad_rows
            }
            try:
                for future in as_completed(future_map):
                    row_id, name = future_map[future]
                    done += 1
                    try:
                        rid, code = future.result()
                        if code:
                            still_bad = any(a in code for a in avoid_list)
                            if still_bad and not args.force:
                                print(f"  SKIP [{rid}] {name}: refined output still contains "
                                      f"banned pattern", file=sys.stderr)
                                failed += 1
                            else:
                                set_translation(conn, rid, target_lang, code)
                                saved += 1
                                if args.verbose:
                                    print(f"  [{rid}] {name} → refined")
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"  ERROR [{name}]: {e}", file=sys.stderr)
                        failed += 1

                    if done % 20 == 0 or done == total:
                        print(f"  Progress: {done}/{total} (saved={saved}, failed={failed})")
            except KeyboardInterrupt:
                print("\nInterrupted by user.")

        print(f"Done: {saved} refined and saved, {failed} failed out of {done} processed.")
        return

    # ------------------------------------------------------------------
    # TRANSLATE MODE (default)
    # ------------------------------------------------------------------
    if getattr(args, "auto", False):
        return _run_auto(conn, args, config, cfg_project)

    force_clause = "" if args.force else f"AND json_extract(translations, '$.{target_lang}') IS NULL"

    # A seed already in the target family needs no translation — gcc and
    # clang read the same C, tint and naga the same WGSL. Without this the
    # C corpus gets "translated" into C: 59787 of 134646 seeds, 44% of every
    # run against a c/cpp target, spent producing what already exists.
    target_family = project_to_family(args.target)
    has_family = _has_column(conn, "family")
    family_clause = ""
    if has_family and not args.force:
        family_clause = f"AND (family IS NULL OR family != '{target_family}')"
    elif not has_family:
        print("  note: no `family` column — run corpus/add_language.py to skip "
              "seeds that are already in the target language", file=sys.stderr)

    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    rows = conn.execute(
        f"SELECT id, project, name, program"
        f"{', language' if _has_column(conn, 'language') else ''} "
        f"FROM corpus WHERE 1=1 {source_clause} {force_clause} {family_clause} "
        f"ORDER BY length(program) {limit_clause}"
    ).fetchall()

    if not rows:
        print(f"No entries to translate (target={target_lang}).")
        return

    total = len(rows)
    print(f"Translating {total} entries → {target_lang} "
          f"(family {target_family}) using project '{cfg_project}' LLM config")

    # --dry-run used to apply only to refine mode, so `--target X --dry-run`
    # quietly went ahead and spent tokens. Anything that costs money should
    # be inspectable first.
    if args.dry_run:
        for r in rows[:10]:
            lang = _source_lang(r)
            print(f"    [{r['id']}] {lang:10s} {(r['name'] or '')[:56]}")
        if total > 10:
            print(f"    ... and {total - 10} more")
        print("  (dry-run - nothing translated)")
        return

    done_count = saved = failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                _translate_one,
                r["id"], r["name"] or str(r["id"]), r["program"],
                _source_lang(r).capitalize(), target_lang.capitalize(), config,
            ): (r["id"], r["name"])
            for r in rows
        }
        try:
            for future in as_completed(future_map):
                row_id, name = future_map[future]
                done_count += 1
                try:
                    rid, code = future.result()
                    if code:
                        set_translation(conn, rid, target_lang, code)
                        saved += 1
                        if args.verbose:
                            print(f"  [{rid}] {name} → saved")
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  ERROR [{name}]: {e}", file=sys.stderr)
                    failed += 1

                if done_count % 50 == 0 or done_count == total:
                    print(f"  Progress: {done_count}/{total} (saved={saved}, failed={failed})")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")

    print(f"Done: {saved} saved, {failed} failed out of {done_count} processed.")


# ---------------------------------------------------------------------------
# Stats command
# ---------------------------------------------------------------------------

def cmd_stats(args, conn: sqlite3.Connection):
    proj_rows = conn.execute(
        "SELECT project, COUNT(*) as cnt FROM corpus GROUP BY project ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in proj_rows)
    proj_order = [r["project"] for r in proj_rows]
    proj_counts = {r["project"]: r["cnt"] for r in proj_rows}

    print(f"{'PROJECT':<20} {'COUNT':>8}")
    print("-" * 30)
    for r in proj_rows:
        print(f"{r['project']:<20} {r['cnt']:>8}")
    print("-" * 30)
    print(f"{'TOTAL':<20} {total:>8}")

    # Build a target-language x source-project translation matrix.
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    target_totals: Counter = Counter()
    for row in conn.execute("SELECT project, translations FROM corpus"):
        translations = json.loads(row["translations"]) if row["translations"] else {}
        for lang in translations:
            matrix[lang][row["project"]] += 1
            target_totals[lang] += 1

    if not matrix:
        return

    targets = sorted(matrix, key=lambda t: target_totals[t], reverse=True)
    # Abbreviate source-project column headers so the matrix stays narrow.
    abbrev = {p: p.replace("inferredbugs-", "ib-") for p in proj_order}
    abbrev = {p: (a if len(a) <= 8 else a[:7] + "…") for p, a in abbrev.items()}
    col_w = max(6, max(len(a) for a in abbrev.values())) + 1

    print()
    print("Translations by target language (rows) x source project (columns):")
    header = f"{'TARGET':<10}" + "".join(f"{abbrev[p]:>{col_w}}" for p in proj_order) + f"{'TOTAL':>{col_w+2}}" + f"{'%ALL':>7}"
    print(header)
    print("-" * len(header))
    for lang in targets:
        counts = [matrix[lang].get(p, 0) for p in proj_order]
        row_total = sum(counts)
        pct = 100.0 * row_total / total if total else 0.0
        line = f"{lang:<10}" + "".join(f"{c:>{col_w}}" for c in counts) + f"{row_total:>{col_w+2}}" + f"{pct:>6.1f}%"
        print(line)
    print("  (columns: " + ", ".join(f"{abbrev[p]}={p}" for p in proj_order if abbrev[p] != p) + ")"
          if any(abbrev[p] != p for p in proj_order) else "", end="")
    if any(abbrev[p] != p for p in proj_order):
        print()

    print()
    print("Coverage by source project (translated entries / total entries for that project):")
    for p in proj_order:
        translated_langs = {lang: matrix[lang].get(p, 0) for lang in targets if matrix[lang].get(p, 0)}
        if not translated_langs:
            print(f"  {p}: no translations")
            continue
        detail = ", ".join(f"{lang}={c}/{proj_counts[p]}" for lang, c in translated_langs.items())
        print(f"  {p}: {detail}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    # Shared parent so --db / -v can appear after the subcommand too
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database")
    shared.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(
        description="Universal Bug Corpus Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[shared],
    )

    sub = parser.add_subparsers(dest="command")

    # import
    p_import = sub.add_parser("import", parents=[shared], help="Import test suite into corpus")
    p_import.add_argument("--project", required=True, help="Project name (e.g. php)")
    p_import.add_argument("--tests-dir", help="Override default tests directory")

    # list
    p_list = sub.add_parser("list", parents=[shared], help="List corpus entries")
    p_list.add_argument("--project", help="Filter by project")

    # show
    p_show = sub.add_parser("show", parents=[shared], help="Show a corpus entry")
    p_show.add_argument("id", type=int)

    # translate (manual, single entry)
    p_trans = sub.add_parser("translate", parents=[shared], help="Attach a manual translation to an entry")
    p_trans.add_argument("id", type=int, help="Corpus entry id")
    p_trans.add_argument("--lang", required=True, help="Target language key (e.g. go, python, rust)")
    p_trans.add_argument("--file", required=True, help="File containing the translated code")

    # translate-llm (batch LLM translation + --refine mode)
    p_tlm = sub.add_parser(
        "translate-llm", parents=[shared],
        help="Batch-translate entries with LLM; use --refine to rewrite bad translations",
    )
    p_tlm.add_argument("--target", default=None,
                       help="Target language / project (e.g. go, rust, python, cpython). "
                            "Not needed with --auto, which picks it.")
    p_tlm.add_argument("--project",
                       help="Project whose LLM config to use (defaults to --target)")
    p_tlm.add_argument("--source",
                       help="Only process entries from this source project (e.g. php)")
    p_tlm.add_argument("--concurrency", type=int, default=None,
                       help="Parallel workers (defaults to project config)")
    p_tlm.add_argument("--auto", action="store_true",
                       help="Pick the target family with the fewest seeds and "
                            "translate into it, choosing sources by viability "
                            "and diversity rather than by length alone.")
    p_tlm.add_argument("--rounds", type=int, default=None,
                       help="(--auto) Repeat this many times, re-picking the "
                            "neediest target each round.")
    p_tlm.add_argument("--seed", type=int, default=None,
                       help="(--auto) Seed the source sampling, for a "
                            "reproducible selection.")
    p_tlm.add_argument("--token-budget", type=int, default=None,
                       help="Stop once this many tokens have been spent. "
                            "Counted from the provider's own usage field, "
                            "per run.")
    p_tlm.add_argument("--limit", type=int, default=None,
                       help="Translate at most N entries, shortest first. The point "
                            "is to fill a gap, not to translate everything: a target "
                            "with 724 seeds needs a few thousand, not 134646.")
    p_tlm.add_argument("--force", action="store_true",
                       help="Translate mode: re-translate even if a translation already exists. "
                            "Refine mode: save even if banned pattern still present in output.")
    # Refine-mode flags (ignored in translate mode)
    p_tlm.add_argument("--refine", action="store_true",
                       help="Refine mode: rewrite existing translations that match --filter")
    p_tlm.add_argument("--filter", metavar="PATTERN", default=None,
                       help="(--refine) Substring marking a bad translation, e.g. 'import ctypes'")
    p_tlm.add_argument("--avoid", metavar="NAMES", default=None,
                       help="(--refine) Comma-separated APIs the LLM must not use, e.g. 'ctypes,cffi'")
    p_tlm.add_argument("--extra", metavar="TEXT", default="",
                       help="(--refine) Extra constraint appended to the refine prompt")
    p_tlm.add_argument("--dry-run", action="store_true",
                       help="(--refine) List matching entries without making any changes")

    # stats
    sub.add_parser("stats", parents=[shared], help="Show corpus statistics")

    # Support legacy invocation: python3 main.py --project php [--tests-dir ...] [--db ...]
    if "--project" in sys.argv and not any(
        a in sys.argv for a in ("import", "list", "show", "translate", "translate-llm", "stats")
    ):
        legacy = argparse.ArgumentParser(add_help=False)
        legacy.add_argument("--project")
        legacy.add_argument("--tests-dir")
        legacy.add_argument("--db", default=str(DEFAULT_DB))
        legacy.add_argument("-v", "--verbose", action="store_true")
        largs = legacy.parse_args()
        conn = get_db(largs.db)
        cmd_import(largs, conn)
        return

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    conn = get_db(args.db)

    dispatch = {
        "import":        cmd_import,
        "list":          cmd_list,
        "show":          cmd_show,
        "translate":     cmd_translate,
        "translate-llm": cmd_translate_llm,
        "stats":         cmd_stats,
    }
    dispatch[args.command](args, conn)


if __name__ == "__main__":
    main()
