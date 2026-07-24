"""
core/llm_mapping.py — LLM-Assisted Fusion Adaptation

State fusion's per-language pattern config (core/state_analysis.py) knows
what a "resource release" or "type conversion" or "exception boundary"
looks like syntactically, but that knowledge is language-specific
(`drop()` in Rust vs `unset()` in PHP). Rather than hand-enumerating it
for every fuzzing target, this module asks an LLM to generate that
mapping — grounded in a sample of the project's own seed corpus — and
writes it to projects/<name>/state_patterns.json, which
state_analysis.get_patterns() merges on top of the hand-seeded defaults.

Two modes:
  generate_mapping() — propose a fresh mapping for a language with none yet.
  refine_mapping()   — read the validity-gap metric written by
                        core/orchestrator.py (output/<project>_validity_gap.json);
                        if fusion's valid-rate is falling far short of the
                        language's own baseline, ask the LLM to narrow the
                        current mapping's false positives rather than
                        regenerating blind. This is the design's guidance
                        loop: "[the validity gap] provides guidance to
                        refine this mapping for languages where the gap is
                        large."

Reuses core/llmgen.py's LLMGenerator for the actual provider call
(gemini/openai/vllm/ollama/deepseek) instead of a new HTTP client.
"""

import json
import logging
import os
import re
import sqlite3
from typing import Dict, List, Optional

from .llmgen import LLMGenerator
from .state_analysis import DEFAULT_PATTERNS, LANGUAGE_ALIASES

logger = logging.getLogger("FFL.LLMMapping")

_CATEGORIES = ("resource_release", "type_conversion", "exception_boundary")

# Validity-gap threshold (percentage points) above which refine_mapping()
# considers the current mapping worth revising. Below this, the gap is
# assumed to reflect the language's own baseline rejection rate rather
# than a bad state-of-interest mapping.
GAP_REFINE_THRESHOLD = 15.0


def _sample_seed_snippets(project_root: str, n: int = 4, max_chars: int = 600) -> List[str]:
    """A few short seed snippets from the project's own corpus, so the LLM
    grounds its answer in this language's actual idioms instead of
    guessing generic syntax."""
    db_path = os.path.join(project_root, "corpus.db")
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT content FROM seeds ORDER BY RANDOM() LIMIT ?", (n,)).fetchall()
        conn.close()
        return [r[0][:max_chars] for r in rows if r[0]]
    except Exception as e:
        logger.debug(f"Could not sample seeds from {db_path}: {e}")
        return []


def _build_prompt(language: str, examples: List[str],
                   current: Optional[Dict[str, List[str]]] = None,
                   refine_reason: Optional[str] = None) -> str:
    example_block = "\n\n".join(f"--- example {i + 1} ---\n{e}" for i, e in enumerate(examples))
    example_block = example_block or "(no examples available — use your own knowledge of the language)"
    task = "generate" if current is None else "refine"

    prompt = f"""You are helping build a fuzzer for the {language} language processor (compiler/interpreter).

The fuzzer looks for "state-of-interest" program points in a seed: places right after a statement where grafting another program's continuation is most likely to trigger an interesting interaction. Three categories:

1. resource_release — right after releasing/freeing/closing a resource (e.g. `drop()` in Rust, `unset()` in PHP, `.close()` in Python).
2. type_conversion — right after an explicit type coercion/cast.
3. exception_boundary — at a try/catch/except/panic/recover boundary.

Task: {task} a JSON object mapping each category to a list of Python regular expressions, each matching a single source line containing that kind of {language} statement. Base every pattern on {language}'s actual syntax.

Example {language} programs from this project's own seed corpus (for grounding — study their idioms):
{example_block}
"""
    if current is not None:
        prompt += f"""
Current patterns (JSON):
{json.dumps(current, indent=2)}

These patterns are producing too many invalid fused programs in this language relative to how often the language's own *unfused* seeds are valid ({refine_reason}). Narrow or correct the regexes so they anchor more precisely on real {language} state-of-interest points and stop matching ordinary code — false positives are what cause a syntactically-broken splice.
"""

    prompt += """
Output ONLY a raw JSON object of this exact shape, nothing else:
{"resource_release": ["regex1", "..."], "type_conversion": ["..."], "exception_boundary": ["..."]}
No markdown, no explanation, no comments, no trailing text.
"""
    return prompt


def _validate_patterns(raw: str) -> Optional[Dict[str, List[str]]]:
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM mapping response wasn't valid JSON: {e}\nRaw: {raw[:300]}")
        return None
    if not isinstance(data, dict):
        return None

    out: Dict[str, List[str]] = {}
    for cat in _CATEGORIES:
        pats = data.get(cat, [])
        if not isinstance(pats, list):
            continue
        valid = []
        for p in pats:
            if not isinstance(p, str):
                continue
            try:
                re.compile(p)
                valid.append(p)
            except re.error:
                logger.debug(f"Dropping uncompilable pattern from LLM mapping: {p!r}")
        if valid:
            out[cat] = valid
    return out or None


def generate_mapping(project_name: str, language: str, config: dict,
                      project_root: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Propose a fresh state-of-interest pattern set for `language`."""
    project_root = project_root or os.path.join("projects", project_name)
    gen = LLMGenerator({**config, "project_name": project_name})
    examples = _sample_seed_snippets(project_root)
    prompt = _build_prompt(language, examples)
    raw = gen._call_api(prompt)
    if not raw:
        logger.warning(f"LLM mapping generation returned nothing for {project_name}/{language}")
        return None
    return _validate_patterns(raw)


def refine_mapping(project_name: str, language: str, config: dict,
                    project_root: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Refine the current mapping using the validity-gap signal written by
    core/orchestrator.py. Returns None if there's no gap data yet, or the
    gap doesn't clear GAP_REFINE_THRESHOLD (nothing worth refining)."""
    project_root = project_root or os.path.join("projects", project_name)
    gap_path = os.path.join("output", f"{project_name}_validity_gap.json")
    if not os.path.exists(gap_path):
        logger.info(f"No validity-gap data at {gap_path} yet — run fuzzing with --dry-run first.")
        return None
    try:
        with open(gap_path, "r", encoding="utf-8") as f:
            gap_data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {gap_path}: {e}")
        return None

    gap = gap_data.get("validity_gap")
    if gap is None or gap < GAP_REFINE_THRESHOLD:
        logger.info(f"Validity gap ({gap}) below refine threshold ({GAP_REFINE_THRESHOLD}pp) — nothing to refine.")
        return None

    lang = LANGUAGE_ALIASES.get(language, language)
    current_path = os.path.join(project_root, "state_patterns.json")
    current_all = {}
    if os.path.exists(current_path):
        try:
            with open(current_path, "r", encoding="utf-8") as f:
                current_all = json.load(f)
        except Exception:
            current_all = {}
    current = current_all.get(lang) or DEFAULT_PATTERNS.get(lang, {})

    gen = LLMGenerator({**config, "project_name": project_name})
    examples = _sample_seed_snippets(project_root)
    reason = (f"baseline={gap_data.get('baseline_valid_rate')}%, "
              f"fused={gap_data.get('fused_valid_rate')}%, gap={gap:+.1f}pp")
    prompt = _build_prompt(language, examples, current=current, refine_reason=reason)
    raw = gen._call_api(prompt)
    if not raw:
        logger.warning(f"LLM mapping refinement returned nothing for {project_name}/{language}")
        return None
    return _validate_patterns(raw)


def save_mapping(project_name: str, language: str, patterns: Dict[str, List[str]],
                  project_root: Optional[str] = None) -> str:
    """Write `patterns` into projects/<name>/state_patterns.json under the
    resolved language key, backing up whatever was there before."""
    project_root = project_root or os.path.join("projects", project_name)
    path = os.path.join(project_root, "state_patterns.json")
    lang = LANGUAGE_ALIASES.get(language, language)

    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            with open(path + ".bak", "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            existing = {}

    existing[lang] = patterns
    os.makedirs(project_root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Wrote state-of-interest mapping for '{lang}' ({sum(len(v) for v in patterns.values())} patterns) to {path}")
    return path


def _main():
    import argparse
    from .config_loader import load_project_config

    ap = argparse.ArgumentParser(description="LLM-assisted state-of-interest mapping generation/refinement")
    ap.add_argument("--project", required=True, help="Project folder name under projects/")
    ap.add_argument("--language", default=None, help="state_analysis language key (defaults to --project name)")
    ap.add_argument("--refine", action="store_true",
                     help="Refine the existing mapping using the validity-gap signal instead of generating fresh")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    language = args.language or args.project
    config = load_project_config(args.project)

    result = refine_mapping(args.project, language, config) if args.refine \
        else generate_mapping(args.project, language, config)

    if result:
        save_mapping(args.project, language, result)
    else:
        logger.info("No mapping written.")


if __name__ == "__main__":
    _main()
