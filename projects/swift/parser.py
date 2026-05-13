from core.parser import BaseParser


class SwiftParser(BaseParser):
    extensions = ['.swift']
    seed_type = 'swift'

    def parse_content(self, content, filename=""):
        return {
            "has_frontend_flags": "// RUN:" in content,
        }


_parser = SwiftParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
