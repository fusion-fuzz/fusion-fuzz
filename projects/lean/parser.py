"""
projects/lean/parser.py — Lean 4 seed collection for FusionFuzz.

Seeds are .lean files under projects/lean/seeds/ (written by setup.py from
the lean4 test suite). The metadata records what the driver and the
fusion strategies look at, read with core/lean_text.py so it agrees with
how the strategies cut a file:

  imports / is_module   the header (`module`, `import ...`)
  has_main              a `def main` — the driver may also run the program
  declarations          declared names per kind (def/theorem/structure/...)
  commands              number of top-level commands
  by_blocks / do_blocks multi-line tactic proofs / `do` blocks, the sites
                        tactic- and do-level state fusion use
  uses_meta             metaprogramming (`run_meta`, `elab`, `MetaM`, ...):
                        a panic raised by Lean's own API on a seed like
                        this may be the program's misuse, not a Lean bug,
                        and the analyzer labels it as such
  expects_fail          an `elab_fail` test (exit 1 is its expected outcome)
"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.parser import BaseParser
from core import lean_text

_META_RE = re.compile(
    r'\brun_meta\b|\brun_cmd\b|\brun_elab\b|^elab\b|^elab_rules\b|\bMetaM\b|\bTermElabM\b|'
    r'\bCommandElabM\b|\bTacticM\b|\bCoreM\b|\baddDecl\b|\bmkApp|\bLean\.Expr\b|\bExpr\.',
    re.M)


class LeanParser(BaseParser):
    extensions = ['.lean']
    seed_type = 'lean'

    def parse_content(self, content, filename=""):
        header, body = lean_text.split_header(content)
        cmds = lean_text.split_commands(body)
        decls = {}
        for d in lean_text.declarations(body, cmds):
            decls.setdefault(d.kind, []).append(d.fqname)
        blocks = lean_text.blocks(body)
        return {
            "type": "lean",
            "extension": ".lean",
            "imports": [h.strip() for h in header if "import" in h],
            "is_module": any(h.strip() == "module" for h in header),
            "has_main": "main" in decls.get("def", []) or
                        bool(re.search(r'^(?:@\[[^\]]*\]\s*)?(?:unsafe\s+)?def\s+main\b', body, re.M)),
            "declarations": decls,
            "commands": len(cmds),
            "by_blocks": sum(1 for b in blocks if b.kind == "by"),
            "do_blocks": sum(1 for b in blocks if b.kind == "do"),
            "uses_meta": bool(_META_RE.search(lean_text.mask(body))),
            "expects_fail": content.startswith("-- ffl: expect-fail"),
            "line_count": len(content.splitlines()),
        }


_parser = LeanParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
