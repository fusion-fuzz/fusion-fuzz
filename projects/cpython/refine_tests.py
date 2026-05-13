import os
import sys
import subprocess
import glob
import re
from openai import OpenAI

# --- CONFIGURATION ---
# Path to your custom CPython build
CPYTHON_BIN = os.path.abspath("./cpython/build/python")

# Input and Output Directories
INPUT_DIR = "./cpython-fuzzing-corpus/corpus"
OUTPUT_DIR = "./final_seeds"

# LLM API Configuration
# Emulating llm_conf.get("api_base", ...)
API_BASE = "http://localhost:8008/v1"
API_KEY = "EMPTY"  # vLLM usually requires a placeholder key
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Retry Logic
MAX_RETRIES = 3
# ---------------------

def verify_runtime(file_path):
    """
    Runs the script with custom CPython.
    Returns (success, output).
    """
    try:
        # 5-second timeout to prevent infinite loops in bad test cases
        res = subprocess.run(
            [CPYTHON_BIN, file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        return (res.returncode == 0, res.stdout + res.stderr)
    except subprocess.TimeoutExpired:
        return (False, "Timed out execution.")
    except Exception as e:
        return (False, str(e))

def extract_code(text):
    """Extracts python code from markdown blocks."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Connecting to LLM at {API_BASE}...")
    try:
        client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    files = glob.glob(os.path.join(INPUT_DIR, "*.py"))
    print(f"Found {len(files)} candidate files.")

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        print(f"[{i+1}/{len(files)}] Processing {fname}...", end=" ", flush=True)

        # 1. Read Original Source
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
        except Exception as e:
            print(f"Skipping (Read Error): {e}")
            continue

        # 2. Initial Check
        success_init, log_init = verify_runtime(fpath)
        
        # State variables for the retry loop
        current_code = original_code
        current_error = log_init if not success_init else "None (Just minimize)"
        
        # 3. Retry Loop
        resolved = False
        for attempt in range(1, MAX_RETRIES + 1):
            
            # Determine prompt strategy based on attempt number
            task_desc = "Flatten and minimize" if attempt == 1 else "Fix the error in the previous attempt"
            
            prompt_content = (
                "You are a Python Core Developer.\n"
                f"TASK: {task_desc}.\n"
                "1. Remove all 'class' and 'def' wrappers. Code must run at top level.\n"
                "2. Remove unused imports and unnecessary data.\n"
                "3. Remove all assert statements.\n"
                "4. Fix any errors so it returns exit code 0.\n\n"
                "CODE:\n"
                "```python\n"
                f"{current_code}\n"
                "```\n\n"
                "ERROR LOG:\n"
                f"{current_error}\n\n"
                "OUTPUT:\n"
                "Provide ONLY the valid Python code inside a ```python``` block.\n"
                "\\nothink"
            )

            try:
                # Call the API
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=2048,
                    stop=["```\n"] # Stop generation when code block ends
                )
                
                generated_text = response.choices[0].message.content
                new_code = extract_code(generated_text)

                # Write to temp file
                temp_path = os.path.join(OUTPUT_DIR, f"temp_{fname}")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                
                # Verify Runtime
                success_new, log_new = verify_runtime(temp_path)
                
                if success_new:
                    # Success! Rename and break loop
                    final_path = os.path.join(OUTPUT_DIR, fname)
                    os.rename(temp_path, final_path)
                    print(f"SUCCESS (Attempt {attempt}) -> Saved.")
                    resolved = True
                    break
                else:
                    # Failed. Feed error back into the loop
                    current_code = new_code
                    current_error = log_new
            
            except Exception as e:
                print(f"API/Processing Error during attempt {attempt}: {e}")
                break

        # Cleanup if we gave up
        temp_path = os.path.join(OUTPUT_DIR, f"temp_{fname}")
        if not resolved:
            print(f"FAILED (Gave up after {MAX_RETRIES} tries).")
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
