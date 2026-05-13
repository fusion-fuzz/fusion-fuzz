import os
import sqlite3
import json
import ast
import concurrent.futures
import multiprocessing

class PythonFastDataflow:
    """
    Analyzes Python code to extract variables and dataflow groups (interactions) using AST.
    """
    def __init__(self):
        self.variables = set()
        self.interactions = []

    def _get_names(self, node):
        """Recursively collect variable names (ids) from a node."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _merge_dataflows(self, groups):
        """
        Merge groups of variables that interact transitively.
        E.g., [a, b] and [b, c] becomes [a, b, c].
        """
        merged = []
        for group in groups:
            group_set = set(group)
            merged_indices = []
            
            # Find all existing groups that overlap with the new group
            for i, m_group in enumerate(merged):
                if not group_set.isdisjoint(m_group):
                    merged_indices.append(i)
            
            if not merged_indices:
                # No overlap, add as new group
                merged.append(group_set)
            else:
                # Overlap found, merge all involved groups
                new_merged_group = group_set
                # Iterate backwards to pop safely
                for i in sorted(merged_indices, reverse=True):
                    new_merged_group.update(merged.pop(i))
                merged.append(new_merged_group)
                
        # Convert sets back to lists for JSON serialization
        return [list(g) for g in merged]

    def analyze(self, code):
        self.variables = set()
        self.interactions = []
        
        try:
            tree = ast.parse(code)
        except Exception:
            # Return empty if syntax error or parse failure
            return [], []

        # Walk the tree to find interactions
        for node in ast.walk(tree):
            # Check statement types where dataflow happens
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, 
                                 ast.Call, ast.Compare, ast.BinOp, ast.BoolOp, 
                                 ast.Return, ast.Yield)):
                names = self._get_names(node)
                if names:
                    self.variables.update(names)
                    if len(names) > 1:
                        self.interactions.append(list(names))
        
        return list(self.variables), self._merge_dataflows(self.interactions)

def _process_seed_file(file_path, source_path):
    """
    Worker function to parse a single seed file.
    """
    seed_filename = os.path.relpath(file_path, source_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Initialize Analyzer per file to be safe in threads
        analyzer = PythonFastDataflow()
        variables, dataflows = analyzer.analyze(content)

        # Basic metadata extraction
        metadata = {
            "type": "python",
            "filename": seed_filename,
            "is_test": "test_" in seed_filename,
            "variables": variables,
            "dataflows": dataflows
        }
        
        return {
            "identifier": seed_filename, # Use filename as identifier for initial corpus
            "content": content,
            "metadata": metadata
        }
    except Exception as e:
        print(f"Failed to read/parse seed {seed_filename}: {e}")
        return None

def collect_seeds(source_path, blacklist=None):
    """
    Iterates over Python files in the CPython Lib/test directory,
    parses them in parallel, and saves them to corpus.db.
    """
    if not os.path.exists(source_path):
        print(f"Error: Seed source path not found: {source_path}")
        return None

    seeds_output = []

    # List files recursively
    print(f"Scanning for .py files in {source_path}...")
    seed_paths = []
    try:
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith('.py'):
                    seed_paths.append(os.path.join(root, file))
    except OSError as e:
        print(f"Error listing directory {source_path}: {e}")
        return None

    total_files = len(seed_paths)
    print(f"Found {total_files} seeds. Processing in parallel...")

    # Use CPU count for AST parsing
    max_workers = os.cpu_count() or 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_file = {executor.submit(_process_seed_file, fp, source_path): fp for fp in seed_paths}

        completed = 0
        skipped = 0
        for future in concurrent.futures.as_completed(future_to_file):
            res = future.result()
            if res:
                if blacklist and any(term in res["content"] for term in blacklist):
                    skipped += 1
                else:
                    seeds_output.append(res)

            completed += 1
            if completed % 100 == 0:
                print(f"  Parsed {completed}/{total_files}...")

    if skipped:
        print(f"Skipped {skipped} seeds matching blacklist.")

    # Save to corpus.db in the project directory
    try:
        # Determine DB path relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "corpus.db")
        
        print(f"Saving corpus to local DB: {db_path}...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Use 'identifier' column instead of 'filename' to match core schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT UNIQUE,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        # Batch insert for speed
        inserts = []
        for seed in seeds_output:
            inserts.append((
                seed['identifier'], 
                seed['content'], 
                json.dumps(seed['metadata'])
            ))
            
        # Use INSERT OR IGNORE to skip duplicates if re-running
        cursor.executemany('''
            INSERT OR IGNORE INTO seeds (identifier, content, metadata) 
            VALUES (?, ?, ?)
        ''', inserts)
        
        count = len(inserts)
        conn.commit()
        conn.close()
        print(f"Saved/Updated {count} seeds to {db_path}")
        return db_path
        
    except Exception as e:
        print(f"Error saving to corpus.db: {e}")
        return None

def load_corpus(db_path):
    """
    Loads seeds from the SQLite corpus.db into a list of dictionaries.
    """
    if not os.path.exists(db_path):
        print(f"Corpus DB not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT identifier, content, metadata FROM seeds")
    rows = cursor.fetchall()
    conn.close()

    seeds = []
    for r in rows:
        seeds.append({
            "filename": r[0], # Map identifier back to filename for compatibility
            "content": r[1],
            "metadata": json.loads(r[2])
        })
    return seeds