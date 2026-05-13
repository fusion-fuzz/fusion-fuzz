import os
import sqlite3
import json
import subprocess
import tempfile
import sys
import concurrent.futures

# Try to import yaml for config reading
try:
    import yaml
except ImportError:
    yaml = None

def _trace_seed(seed_id, content, metadata_str, python_bin):
    """
    Helper function to trace a single seed.
    Returns (updated_metadata_json, seed_id) if successful, else None.
    Run inside a temporary directory to contain side effects.
    """
    try:
        metadata = json.loads(metadata_str)
    except:
        metadata = {}

    result_tuple = None

    # Use a temporary directory context manager for auto-cleanup
    # This ensures trash files created by seed execution are removed
    with tempfile.TemporaryDirectory(prefix=f'ffl_proc_{seed_id}_') as temp_dir:
        seed_path = os.path.join(temp_dir, "seed.py")
        runner_path = os.path.join(temp_dir, "runner.py")
        
        try:
            with open(seed_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            return None

        # Create the tracer wrapper script
        runner_code = f"""
import sys
import json
import collections
import os

# Map: variable_name -> list of type_names
types_seen = collections.defaultdict(set)

def tracer(frame, event, arg):
    if event == 'line':
        # We only want to trace the specific seed file
        if frame.f_code.co_filename == r"{seed_path}":
            # Snapshot types of local variables
            for k, v in frame.f_locals.items():
                if k.startswith("__"): continue
                try:
                    t_name = type(v).__name__
                    types_seen[k].add(t_name)
                except: 
                    pass
    return tracer

sys.settrace(tracer)

try:
    # Execute the seed file in a dedicated namespace
    with open(r"{seed_path}") as f:
        code = compile(f.read(), r"{seed_path}", 'exec')
        exec(code, {{'__name__': '__main__'}})
except Exception:
    pass # Ignore runtime errors in seeds during preprocessing
except SystemExit:
    pass
finally:
    sys.settrace(None)
    # Output captured types as JSON markers
    output = {{k: list(v) for k, v in types_seen.items()}}
    print("__FFL_TYPES_START__")
    print(json.dumps(output))
    print("__FFL_TYPES_END__")
"""
        
        try:
            with open(runner_path, 'w', encoding='utf-8') as f:
                f.write(runner_code)

            # Execute the runner with a timeout
            # cwd=temp_dir ensures any files created by the seed stay in this temp dir
            res = subprocess.run(
                [python_bin, runner_path],
                capture_output=True,
                text=True,
                timeout=2.0,
                cwd=temp_dir 
            )
            
            # Parse output
            stdout = res.stdout
            if "__FFL_TYPES_START__" in stdout:
                try:
                    parts = stdout.split("__FFL_TYPES_START__")
                    if len(parts) > 1:
                        json_part = parts[1].split("__FFL_TYPES_END__")[0].strip()
                        dynamic_types = json.loads(json_part)
                        
                        if dynamic_types:
                            metadata['dynamic_types'] = dynamic_types
                            result_tuple = (json.dumps(metadata), seed_id)
                except (IndexError, json.JSONDecodeError):
                    pass
                    
        except subprocess.TimeoutExpired:
            pass # Skip slow seeds
        except Exception as e:
            pass
            
    # temp_dir is automatically removed here
    return result_tuple

def preprocess(project_root):
    """
    Linearly executes each seed in the corpus using a tracer to collect 
    dynamic type information for variables. Updates the corpus.db metadata.
    """
    print(f"Starting CPython Preprocessing (Dynamic Type Collection) in {project_root}...")
    
    # 1. Determine Paths
    db_path = os.path.join(project_root, "corpus.db")
    # We use the built python binary to ensure we are testing against the target environment
    python_bin = os.path.join(project_root, "cpython", "build", "python")
    config_path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(db_path):
        print(f"Error: Corpus DB not found at {db_path}")
        return
    
    if not os.path.exists(python_bin):
        print(f"Warning: Built python not found at {python_bin}. Falling back to sys.executable.")
        python_bin = sys.executable

    # 2. Determine Concurrency from Config
    concurrency = 4
    if yaml and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                conf = yaml.safe_load(f)
                concurrency = conf.get("execution", {}).get("concurrency", 4)
        except Exception as e:
            print(f"Warning: could not read config for concurrency, defaulting to {concurrency}: {e}")
    
    # 3. Connect to DB and Fetch Seeds
    try:
        conn = sqlite3.connect(db_path)
        # We don't use row_factory here to simplify data extraction for threads
        cursor = conn.cursor()
        # Fetch data as tuples: (id, content, metadata)
        seeds = cursor.execute("SELECT id, content, metadata FROM seeds").fetchall()
    except Exception as e:
        print(f"DB Error: {e}")
        return

    print(f"Processing {len(seeds)} seeds with concurrency {concurrency}...")
    
    updates = []
    processed_count = 0
    
    # Prepare arguments for parallel execution
    # Extract data from rows to avoid passing SQLite objects to threads
    tasks = []
    for row in seeds:
        seed_id = row[0]
        content = row[1]
        metadata_str = row[2]
        tasks.append((seed_id, content, metadata_str, python_bin))

    # 4. Execute in Parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_id = {executor.submit(_trace_seed, *task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_id):
            processed_count += 1
            try:
                res = future.result()
                if res:
                    updates.append(res)
            except Exception as e:
                print(f"Worker exception: {e}")

            # Batch update every 100 seeds (DB write in main thread)
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(seeds)}...")
                if updates:
                    cursor.executemany("UPDATE seeds SET metadata = ? WHERE id = ?", updates)
                    conn.commit()
                    updates = [] # Clear buffer after commit

    # 5. Save Remaining Updates
    if updates:
        print(f"Updating database with dynamic types for remaining {len(updates)} seeds...")
        cursor.executemany("UPDATE seeds SET metadata = ? WHERE id = ?", updates)
        conn.commit()
    
    conn.close()
    print("Preprocessing complete.")