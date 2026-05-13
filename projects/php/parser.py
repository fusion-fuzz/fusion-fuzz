import re
import os
import subprocess
import sys
import sqlite3
import json

# ==========================================
# Provided PHP Dataflow Logic
# ==========================================

def get_php_dataflow_groups(php_script_path, dataflow_script_path='dataflow.php'):
    """
    Invokes the PHP dataflow analysis script and collects the dataflow list.
    """
    try:
        # Execute the PHP dataflow analysis script
        result = subprocess.run(
            ['php', dataflow_script_path, php_script_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract the output
        output = result.stdout.strip()

        # Use eval to parse the output
        dataflow_groups = eval(output)

        return dataflow_groups

    except subprocess.CalledProcessError as e:
        print(f"Error executing PHP script: {e.stderr}")
        return []
    except Exception as e:
        print(f"Error parsing output: {e}")
        return []


class PHPFastDataflow:
    """
    A class that performs fast, coarse-grained dataflow analysis on PHP code.
    It does not guarantee completeness but aims for soundness.
    """

    def __init__(self):
        self.variables = []  # List to store extracted variables from PHP code
        self.dataflows = []  # List of lists to store dataflows between variables
        self.phpcode = ""

    def clean(self):
        self.variables = []
        self.dataflows = []

    def extract_variables(self):
        """
        Extracts all PHP variables from the PHP code using a regular expression.
        """
        regex = r"\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*"
        self.variables = re.findall(regex, self.phpcode)
        self.variables = list(set(self.variables))

    def analyze_php_line(self, php_line):
        regex = r"\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*"
        variables = list(set(re.findall(regex, php_line)))
        if len(variables) > 1:
            return (True, variables)
        else:
            return (False, None)

    def merge_dataflows(self):
        list_of_lists = self.variables

        # Convert any single variables to lists to ensure consistent structure
        for i in range(len(list_of_lists)):
            if type(list_of_lists[i]) != list:
                list_of_lists[i] = [list_of_lists[i]]

        merged_lists = []

        for sublist in list_of_lists:
            merged_with_existing = False
            for merged_sublist in merged_lists:
                if any(var in merged_sublist for var in sublist):
                    merged_sublist.extend(var for var in sublist if var not in merged_sublist)
                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged_lists.append(sublist)

        self.variables = merged_lists

    def extract_dataflow(self):
        for eachline in self.phpcode.split('\n'):
            result, variables = self.analyze_php_line(eachline)
            if result:
                for each_var in variables:
                    if each_var in self.variables:
                        self.variables.remove(each_var)
                self.variables.append(variables)

        self.merge_dataflows()

    def analyze(self, phpcode):
        self.phpcode = phpcode
        self.clean()
        self.extract_variables()
        self.vars = []
        for each in self.variables:
            self.vars.append(each)
        self.extract_dataflow()
        return self.vars, self.variables


def remove_php_comments(code):
    result = ''
    i = 0
    in_single_quote = False
    in_double_quote = False
    in_single_line_comment = False
    in_multi_line_comment = False
    escaped = False
    code_length = len(code)

    while i < code_length:
        c = code[i]
        next_c = code[i+1] if i+1 < code_length else ''

        # Handle string literals
        if in_single_quote:
            result += c
            if not escaped and c == '\\':
                escaped = True
            elif escaped:
                escaped = False
            elif c == "'":
                in_single_quote = False
            i += 1
            continue
        elif in_double_quote:
            result += c
            if not escaped and c == '\\':
                escaped = True
            elif escaped:
                escaped = False
            elif c == '"':
                in_double_quote = False
            i += 1
            continue

        # Handle comments
        if in_single_line_comment:
            if c == '\n':
                in_single_line_comment = False
                result += c
            i += 1
            continue
        elif in_multi_line_comment:
            if c == '*' and next_c == '/':
                in_multi_line_comment = False
                i += 2
            else:
                i += 1
            continue

        # Detect start of string literals
        if c == "'" and not in_double_quote:
            in_single_quote = True
            result += c
            i += 1
            continue
        elif c == '"' and not in_single_quote:
            in_double_quote = True
            result += c
            i += 1
            continue

        # Detect start of comments
        if c == '/' and next_c == '/':
            in_single_line_comment = True
            i += 2
            continue
        elif c == '/' and next_c == '*':
            in_multi_line_comment = True
            i += 2
            continue
        elif c == '#' and not in_single_quote and not in_double_quote:
            in_single_line_comment = True
            i += 1
            continue

        # Copy other characters
        result += c
        i += 1

    return result


def extract_sec(test, section):
    if section not in test:
        return ""
    start_idx = test.find(section) + len(section)
    end_match = re.search("--([_A-Z]+)--", test[start_idx:])
    end_idx = end_match.start() if end_match else len(test) - 1
    return test[start_idx:start_idx + end_idx].strip("\n")


# ==========================================
# Framework Integration Point
# ==========================================

def collect_seeds(source_path, blacklist=None):
    """
    Iterates over PHPT files in the source_path, parses them using the 
    PHPFastDataflow logic, and returns a list of seed objects for the FFL framework.
    Also saves the parsed seeds into a local corpus.db SQLite database.
    """
    if not os.path.exists(source_path):
        print(f"Error: Seed source path not found: {source_path}")
        return None

    seeds_output = []
    
    # List files recursively
    seed_paths = []
    try:
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith('.phpt'):
                    seed_paths.append(os.path.join(root, file))
    except OSError as e:
        print(f"Error listing directory {source_path}: {e}")
        return None

    print(f"Parsing {len(seed_paths)} PHPT files from {source_path}...")
    
    for file_path in seed_paths:
        # Use relative path for identifier to maintain structure info and uniqueness
        seed_identifier = os.path.relpath(file_path, source_path)
        
        try:
            # Read file with ISO-8859-1 as in original script
            with open(file_path, "r", encoding="iso_8859_1") as f:
                phpt = f.read()
            
            # Determine priority (Secondary)
            secondary = False
            if "--EXPECTF--" in phpt or "declare(" in phpt or "namespace" in phpt:
                secondary = True
            
            # Extract Sections
            description = extract_sec(phpt, "--TEST--")
            configuration = extract_sec(phpt, "--INI--")
            skipif = extract_sec(phpt, "--SKIPIF--")
            phpcode = extract_sec(phpt, "--FILE--")
            extension = extract_sec(phpt, "--EXTENSION--")
            
            # Clean and Analyze
            clean_code = remove_php_comments(phpcode)
            
            # Write to tmp file for analysis (mirroring original logic)
            # We use a specific tmp name to avoid general collision
            tmp_path = "/tmp/tmp_ffl_parser.php"
            with open(tmp_path, "w", encoding="iso_8859_1") as f:
                f.write(clean_code)
            
            # Perform Analysis
            dataflow = PHPFastDataflow()
            variables, dataflows = dataflow.analyze(clean_code)
            
            # Pack into Framework format
            # The 'content' is the executable code part (FILE section)
            # All other extracted info goes into 'metadata'
            seed_obj = {
                "identifier": seed_identifier,
                "content": phpcode, 
                "metadata": {
                    "type": "phpt",
                    "description": description,
                    "configuration": configuration,
                    "skipif": skipif,
                    "extension": extension,
                    "variables": list(variables),  # Ensure JSON serializable
                    "dataflows": dataflows,        # Ensure JSON serializable
                    "secondary": secondary,
                    "clean_code": clean_code
                }
            }
            
            seeds_output.append(seed_obj)
            
        except Exception as e:
            print(f"Failed to parse seed {seed_identifier}: {e}")

    # Cleanup temp file
    if os.path.exists("/tmp/tmp_ffl_parser.php"):
        os.remove("/tmp/tmp_ffl_parser.php")
    
    # Save to corpus.db in the parent directory of source_path (e.g., projects/php/corpus.db)
    try:
        # Determine DB path relative to this script for reliability
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "corpus.db")
        print(f"Saving corpus to local DB: {db_path}...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table with relevant fields using UNIQUE identifier
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT UNIQUE,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        count = 0
        # Insert seeds
        for seed in seeds_output:
            try:
                cursor.execute('''
                    INSERT INTO seeds (identifier, content, metadata) 
                    VALUES (?, ?, ?)
                ''', (
                    seed['identifier'], 
                    seed['content'], 
                    json.dumps(seed['metadata'])
                ))
                count += 1
            except sqlite3.IntegrityError:
                # Skip duplicate identifiers silently or update if needed
                pass
            
        conn.commit()
        conn.close()
        print(f"Saved {count} seeds to {db_path}")
        return db_path # Return the path, not the list
        
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