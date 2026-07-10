import os
import re
from core.parser import BaseParser

# ==========================================
# Coarse-grained Fortran Dataflow Logic
# (same spirit as PHPFastDataflow / CFastDataflow, adapted for Fortran's
#  case-insensitive identifiers and '!' comments)
# ==========================================

_KEYWORDS = frozenset(w.upper() for w in (
    "program", "end", "module", "submodule", "subroutine", "function",
    "use", "only", "implicit", "none", "integer", "real", "logical",
    "character", "complex", "double", "precision", "type", "class",
    "dimension", "allocatable", "pointer", "target", "intent", "in",
    "out", "inout", "optional", "parameter", "save", "public",
    "private", "protected", "value", "external", "intrinsic",
    "recursive", "pure", "elemental", "impure", "result", "contains",
    "interface", "generic", "operator", "assignment", "if", "then",
    "else", "elseif", "endif", "do", "while", "concurrent", "enddo",
    "exit", "cycle", "select", "case", "selecttype", "default",
    "where", "elsewhere", "endwhere", "forall", "endforall",
    "associate", "endassociate", "block", "endblock", "critical",
    "endcritical", "goto", "continue", "stop", "errorstop", "return",
    "call", "allocate", "deallocate", "nullify", "read", "write",
    "print", "format", "open", "close", "inquire", "rewind",
    "backspace", "endfile", "common", "equivalence", "data",
    "namelist", "entry", "procedure", "abstract", "deferred", "nopass",
    "pass", "bind", "import", "enum", "enumerator", "sequence",
    "volatile", "asynchronous", "codimension", "contiguous", "errmsg",
    "mold", "source", "sync", "lock", "unlock", "team", "event",
    "images", "kind", "len", "blockdata", "final", "extends",
))

_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
_LINE_COMMENT_RE = re.compile(r'!.*$')
_FIXED_FORM_COMMENT_RE = re.compile(r'^[CcDd*].*$')
_STR_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


class FortranFastDataflow:
    """
    Fast, coarse-grained dataflow analysis on Fortran source: identifiers
    that co-occur on the same (non-comment) line are grouped into the same
    dataflow, mirroring PHPFastDataflow/CFastDataflow's line-co-occurrence
    heuristic.
    """

    def analyze(self, code: str, fixed_form: bool = False):
        line_groups = []
        all_vars = []
        for line in code.splitlines():
            if fixed_form and _FIXED_FORM_COMMENT_RE.match(line):
                continue
            line = _LINE_COMMENT_RE.sub('', line)
            line = _STR_RE.sub(' ', line)
            idents = [t.upper() for t in _IDENT_RE.findall(line) if t.upper() not in _KEYWORDS]
            idents = list(dict.fromkeys(idents))
            if not idents:
                continue
            all_vars.extend(idents)
            line_groups.append(idents)

        variables = list(dict.fromkeys(all_vars))
        dataflows = self._merge_dataflows(line_groups)
        return variables, dataflows

    @staticmethod
    def _merge_dataflows(line_groups):
        merged = []
        for group in line_groups:
            joined = False
            for existing in merged:
                if any(v in existing for v in group):
                    for v in group:
                        if v not in existing:
                            existing.append(v)
                    joined = True
                    break
            if not joined:
                merged.append(list(group))
        return merged


class FlangParser(BaseParser):
    extensions = ['.f90', '.F90', '.f95', '.F95', '.f', '.F', '.cuf', '.CUF']
    seed_type = 'fortran'

    # Uppercase-extension files use fixed-form column rules unless free-form
    # is explicit; lowercase historically means free-form for .f90/.f95 and
    # fixed-form for bare .f. Flang actually infers this from the extension
    # itself, so we just record it for the driver/dataflow pass.
    _FIXED_FORM_EXTS = ('.f', '.F')

    def parse_content(self, content, filename=""):
        ext = os.path.splitext(filename)[1]
        # Normalize casing lookup while preserving the real extension.
        fixed_form = ext in self._FIXED_FORM_EXTS

        variables, dataflows = FortranFastDataflow().analyze(content, fixed_form=fixed_form)

        return {
            "type": "fortran",
            "extension": ext,
            "fixed_form": fixed_form,
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = FlangParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "llvm-project", "flang", "test")
    print("Executing Flang parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: seed source not found at {seeds_dir}. Run setup.py first.")
