import re
from core.parser import BaseParser


class LeanParser(BaseParser):
    extensions = ['.lean']
    seed_type = 'lean'

    def parse_content(self, content, filename=""):
        return {
            "imports":    re.findall(r'^\s*import\s+([\w.]+)', content, re.MULTILINE),
            "defs":       re.findall(r'^\s*(?:private\s+|protected\s+|noncomputable\s+)?def\s+(\w+)', content, re.MULTILINE),
            "theorems":   re.findall(r'^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+(\w+)', content, re.MULTILINE),
            "structures": re.findall(r'^\s*(?:private\s+|protected\s+)?(?:structure|class|inductive|abbrev)\s+(\w+)', content, re.MULTILINE),
            "namespaces": re.findall(r'^\s*namespace\s+(\w+)', content, re.MULTILINE),
        }


_parser = LeanParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
