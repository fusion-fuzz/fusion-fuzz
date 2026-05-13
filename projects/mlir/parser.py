from core.parser import BaseParser


class MLIRParser(BaseParser):
    extensions = ['.mlir']
    seed_type = 'mlir'

    def parse_content(self, content, filename=""):
        # Extract comment lines as a rough proxy for dialect hints
        dialects = [line for line in content.splitlines() if line.strip().startswith("//")]
        return {"dialects": dialects}


_parser = MLIRParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
