import re
from core.parser import BaseParser


class RustParser(BaseParser):
    extensions = ['.rs']
    seed_type = 'rust'

    def parse_content(self, content, filename=""):
        return {
            "imports":   re.findall(r'^\s*use\s+([^;]+);', content, re.MULTILINE),
            "functions": re.findall(r'fn\s+([a-zA-Z0-9_]+)\s*\(', content),
            "structs":   re.findall(r'struct\s+([a-zA-Z0-9_]+)', content),
        }


_parser = RustParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
