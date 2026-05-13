import os
from core.parser import BaseParser


class GCCParser(BaseParser):
    extensions = ['.c', '.cc', '.cpp', '.i', '.h', '.hpp']
    seed_type = 'c'  # default; overridden per-seed based on extension

    def parse_content(self, content, filename=""):
        ext = os.path.splitext(filename)[1].lower()
        return {
            "type":        "cpp" if ext in ('.cc', '.cpp', '.hpp') else "c",
            "is_dejagnu":  "{ dg-" in content,
            "extension":   ext,
        }


_parser = GCCParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print(f"Executing GCC parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}")
