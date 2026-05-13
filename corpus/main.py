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
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing core modules when run from repo root or corpus dir
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CORPUS_DIR = Path(__file__).parent
DEFAULT_DB = CORPUS_DIR / "corpus.db"

# Maps project name -> canonical language key used in translations JSON
LANG_MAP = {
    "cpython": "python",
    "gcc":     "c",
    "clang":   "c",
    "go":      "go",
    "rust":    "rust",
    "php":     "php",
    "swift":   "swift",
    "lean":    "lean",
    "mlir":    "mlir",
    "naga":    "rust",
    "wgslc":   "wgsl",
    "sql":     "sql",
    "v8":      "javascript",
    "inferredbugs-java":   "java",
    "inferredbugs-csharp": "csharp",
}


def project_to_lang(project: str) -> str:
    return LANG_MAP.get(project.lower(), project.lower())


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

def _translate_one(row_id: int, name: str, program: str, src_lang: str,
                   tgt_lang: str, config: dict) -> tuple[int, str | None]:
    """Worker: translate a single program. Returns (row_id, translated_code | None)."""
    try:
        from core.llmgen import LLMGenerator
        llm = LLMGenerator(config)
        seed = llm.translate(program, src_lang, tgt_lang)
        if seed and seed.content:
            return row_id, seed.content
    except Exception as e:
        print(f"  WARN [{name}]: {e}", file=sys.stderr)
    return row_id, None


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
    target_lang = project_to_lang(args.target)
    cfg_project = args.project or args.target

    try:
        from core.config_loader import load_project_config
        if not os.path.exists(os.path.join("projects", cfg_project, "config.yaml")):
            os.chdir(str(_REPO_ROOT))
        config = load_project_config(cfg_project)
    except FileNotFoundError as e:
        sys.exit(str(e))

    concurrency = args.concurrency or config.get("execution", {}).get("concurrency", 4)
    source_clause = f"AND project = '{args.source}'" if args.source else ""

    # ------------------------------------------------------------------
    # REFINE MODE
    # ------------------------------------------------------------------
    if args.refine:
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
    force_clause = "" if args.force else f"AND json_extract(translations, '$.{target_lang}') IS NULL"

    rows = conn.execute(
        f"SELECT id, project, name, program FROM corpus WHERE 1=1 {source_clause} {force_clause} ORDER BY id"
    ).fetchall()

    if not rows:
        print(f"No entries to translate (target={target_lang}).")
        return

    total = len(rows)
    print(f"Translating {total} entries → {target_lang} using project '{cfg_project}' LLM config")

    done_count = saved = failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                _translate_one,
                r["id"], r["name"] or str(r["id"]), r["program"],
                project_to_lang(r["project"]).capitalize(), target_lang.capitalize(), config,
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
    rows = conn.execute(
        "SELECT project, COUNT(*) as cnt FROM corpus GROUP BY project ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    print(f"{'PROJECT':<20} {'COUNT':>8}")
    print("-" * 30)
    for r in rows:
        print(f"{r['project']:<20} {r['cnt']:>8}")
    print("-" * 30)
    print(f"{'TOTAL':<20} {total:>8}")


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
    p_tlm.add_argument("--target", required=True,
                       help="Target language / project (e.g. go, rust, python, cpython)")
    p_tlm.add_argument("--project",
                       help="Project whose LLM config to use (defaults to --target)")
    p_tlm.add_argument("--source",
                       help="Only process entries from this source project (e.g. php)")
    p_tlm.add_argument("--concurrency", type=int, default=None,
                       help="Parallel workers (defaults to project config)")
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
