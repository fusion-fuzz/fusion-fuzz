import re
from core.parser import BaseParser


# At least one Cangjie top-level declaration must be present
_CANGJIE_TOPLEVEL = re.compile(
    r'^\s*(?:'
    r'(?:(?:public|private|protected|open|abstract|override)\s+)*'
    r'(?:func|class|struct|enum|interface|extend)\s+[A-Za-z_]'
    r'|main\s*\(\s*\)'
    r')',
    re.MULTILINE,
)

# Strong markers that indicate this is NOT Cangjie
_OTHER_LANG = re.compile(
    r'(?:'
    r'^\s*\$[a-zA-Z_{]'           # PHP $var / ${
    r'|^\s*<\?php'                 # PHP open tag
    r'|^\s*function\s+\w+\s*\('   # JS/PHP function keyword
    r'|^\s*def\s+\w+\s*[:(]'      # Python def
    r'|^\s*package\s+main\b'      # Go package main
    r'|^\s*import\s*\('           # Go grouped imports
    r'|:='                         # Go short variable declaration
    r'|^\s*module\s*\{'           # Rust/other module block
    r'|^\s*#!'                    # shebang
    r')',
    re.MULTILINE,
)


class CangjieParser(BaseParser):
    extensions = ['.cj']
    seed_type = 'cangjie'

    def _is_likely_cangjie(self, content: str) -> bool:
        if _OTHER_LANG.search(content):
            return False
        if not _CANGJIE_TOPLEVEL.search(content):
            return False
        return True

    def collect_seeds(self, source_path, blacklist=None):
        """Override to filter out non-Cangjie content before saving to corpus."""
        import os, sqlite3, json

        if not os.path.exists(source_path):
            print(f"Error: Seed source path not found: {source_path}")
            return None

        print(f"Scanning for {self.extensions} files in {source_path}...")
        seed_paths = []
        for root, _, files in os.walk(source_path):
            for fname in files:
                if any(fname.endswith(ext) for ext in self.extensions):
                    seed_paths.append(os.path.join(root, fname))

        print(f"Found {len(seed_paths)} seeds. Filtering and processing...")
        seeds_output = []
        skipped_bl = skipped_lang = 0

        for file_path in seed_paths:
            seed_filename = os.path.relpath(file_path, source_path)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if blacklist and any(term in content for term in blacklist):
                    skipped_bl += 1
                    continue
                if not self._is_likely_cangjie(content):
                    skipped_lang += 1
                    continue
                metadata = self.parse_content(content, seed_filename)
                metadata.setdefault("type", self.seed_type)
                metadata["filename"] = seed_filename
                seeds_output.append({
                    "identifier": seed_filename,
                    "content": content,
                    "metadata": metadata,
                })
            except Exception:
                pass

        print(f"Accepted {len(seeds_output)} Cangjie seeds "
              f"(dropped {skipped_lang} non-Cangjie, {skipped_bl} blacklisted).")
        return self._save_to_db(seeds_output)

    def parse_content(self, content, filename=""):
        return {
            "imports":   re.findall(r'^\s*import\s+([\w.]+)', content, re.MULTILINE),
            "functions": re.findall(r'\bfunc\s+([a-zA-Z_]\w*)\s*[(<]', content),
            "classes":   re.findall(r'\bclass\s+([a-zA-Z_]\w*)', content),
            "structs":   re.findall(r'\bstruct\s+([a-zA-Z_]\w*)', content),
            "enums":     re.findall(r'\benum\s+([a-zA-Z_]\w*)', content),
            "vars":      re.findall(r'\b(?:let|var)\s+([a-zA-Z_]\w*)', content),
            "has_generics": bool(re.search(r'<[A-Z][A-Za-z0-9_]*(?:\s*:\s*\w+)?>', content)),
            "has_async":    bool(re.search(r'\basync\b|\bawait\b', content)),
            "has_unsafe":   bool(re.search(r'\bunsafe\b', content)),
        }


_parser = CangjieParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
