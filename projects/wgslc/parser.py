import re
from core.parser import BaseParser


class WGSLCParser(BaseParser):
    extensions = ['.wgsl']
    seed_type = 'wgsl'

    def parse_content(self, content, filename=""):
        return {
            "functions": re.findall(r'fn\s+([a-zA-Z0-9_]+)\s*\(', content),
            "structs":   re.findall(r'struct\s+([a-zA-Z0-9_]+)', content),
        }


_parser = WGSLCParser(__file__)


def collect_seeds(source_path):
    return _parser.collect_seeds(source_path)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
