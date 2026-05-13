"""
Download the internlm/Lean-Workbook dataset from HuggingFace and populate
the lean corpus.db with formal_statement seeds.

Run with the FusionFuzzLoop openai-env venv which has `datasets` installed:
    source ~/Desktop/FusionFuzzLoop/openai-env/bin/activate
    python projects/lean/dataset.py
"""

import os
import re
import json
import sqlite3

from datasets import load_dataset


# ------------------------------------------------------------------ #
# Regex for extracting lean metadata (mirrors parser.py)             #
# ------------------------------------------------------------------ #
_IMPORT_LINE_RE = re.compile(
    r'^\s*(?:public\s+|meta\s+)?import\s+([\w][\w.]*)\s*$'
)


def _hoist_imports(content: str) -> str:
    """Move all import lines to the top; strip public/meta modifiers."""
    imports, body_lines = [], []
    for line in content.splitlines():
        m = _IMPORT_LINE_RE.match(line)
        if m:
            imports.append(m.group(1))
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    if not imports:
        return body
    import_block = "\n".join(f"import {mod}" for mod in sorted(set(imports)))
    return f"{import_block}\n\n{body}" if body else import_block


def _parse_lean_content(content: str) -> dict:
    imports   = re.findall(r"^\s*import\s+([\w.]+)",                              content, re.MULTILINE)
    defs      = re.findall(r"^\s*(?:private\s+|protected\s+|noncomputable\s+)?def\s+(\w+)",   content, re.MULTILINE)
    theorems  = re.findall(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+(\w+)",      content, re.MULTILINE)
    structures= re.findall(r"^\s*(?:private\s+|protected\s+)?(?:structure|class|inductive|abbrev)\s+(\w+)", content, re.MULTILINE)
    namespaces= re.findall(r"^\s*namespace\s+(\w+)",                              content, re.MULTILINE)
    return {
        "imports":    imports,
        "defs":       defs,
        "theorems":   theorems,
        "structures": structures,
        "namespaces": namespaces,
    }


def download_and_save(db_path: str) -> int:
    """
    Download internlm/Lean-Workbook and insert every formal_statement
    into corpus.db.  Returns the number of newly inserted rows.
    """
    print("Downloading internlm/Lean-Workbook from HuggingFace …")
    ds = load_dataset("internlm/Lean-Workbook", split="train")
    print(f"  Dataset loaded — {len(ds)} rows")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS seeds (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT UNIQUE,
            content    TEXT,
            metadata   TEXT
        )
    """)
    conn.commit()

    inserted = 0
    skipped  = 0

    for i, row in enumerate(ds):
        formal = (row.get("formal_statement") or "").strip()
        if not formal:
            skipped += 1
            continue

        # Normalise: hoist any mid-file imports to the top
        content = _hoist_imports(formal)

        identifier = f"lean_workbook_{i:06d}"
        metadata   = _parse_lean_content(content)
        metadata["type"]   = "lean_workbook"
        metadata["source"] = "internlm/Lean-Workbook"
        # carry through useful fields for LLM context
        nls = (row.get("natural_language_statement") or "").strip()
        if nls:
            metadata["natural_language_statement"] = nls[:512]

        try:
            cur.execute(
                "INSERT INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)",
                (identifier, content, json.dumps(metadata)),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

        if (i + 1) % 5000 == 0:
            conn.commit()
            print(f"  … {i + 1}/{len(ds)} processed, {inserted} inserted so far")

    conn.commit()
    conn.close()
    print(f"Done — inserted {inserted} seeds, skipped {skipped}.")
    return inserted


if __name__ == "__main__":
    here   = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(here, "corpus.db")
    download_and_save(db_path)
