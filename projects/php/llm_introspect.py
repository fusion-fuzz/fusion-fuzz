import sqlite3
import os
import subprocess
import random
import tempfile
import re
import sys
from pathlib import Path
from core.fusion import Seed

# Try to import OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

MAX_RETRIES = 3

# Updated template for plain PHP
PHP_TEMPLATE = """<?php
// Unit test for {symbol_type} {symbol_name}
// TODO: Implement test logic
"""

def safe_filename_from_symbol(symbol: str) -> str:
    bad_chars = ['<', '>', ':', '/', '\\', '*', '?', '"', '|', ' ', '::', '\\']
    fname = symbol.replace("::", "_").replace("\\", "_")
    for ch in bad_chars:
        fname = fname.replace(ch, "_")
    return fname

class CustomLLMGenerator:
    def __init__(self, config):
        self.config = config
        self.project_root = os.path.abspath(os.path.join("projects", "php"))
        self.php_bin = self._find_php_binary()
        
        # Queue for symbols to generate tests for
        self.symbol_queue = []
        self._load_symbols()
        
        # --- LLM Configuration (Shared Logic) ---
        llm_conf = config.get("llm", {})
        self.provider = llm_conf.get("provider", "openai")
        self.model = llm_conf.get("model", "gpt-4o-mini")
        
        env_key = os.getenv("LLM_API_KEY")
        conf_key = llm_conf.get("api_key")

        # Provider setup
        if self.provider == "vllm":
            self.api_key = conf_key or env_key or "EMPTY"
            self.api_base = llm_conf.get("api_base", "http://localhost:8008/v1") 
        elif self.provider == "ollama":
            self.api_key = conf_key or env_key or "ollama"
            self.api_base = llm_conf.get("api_base", "http://localhost:11434/v1")
        else: 
            self.api_key = conf_key or env_key
            self.api_base = llm_conf.get("api_base") 

        self.client = None
        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            except Exception as e:
                print(f"[LLM Init Error] {e}")
        else:
            if not HAS_OPENAI:
                print("[LLM Warning] 'openai' package missing.")
            elif not self.api_key:
                print("[LLM Warning] No API key provided.")

    def _find_php_binary(self):
        # Look for the built CLI binary
        built_bin = os.path.join(self.project_root, "php-src", "sapi", "cli", "php")
        if os.path.exists(built_bin):
            return built_bin
        return "php" # Fallback to system php

    def _load_symbols(self):
        """Loads functions and classes from the SQLite databases."""
        # 1. Load Functions
        apis_db = os.path.join(self.project_root, "apis.db")
        if os.path.exists(apis_db):
            try:
                conn = sqlite3.connect(apis_db)
                cursor = conn.cursor()
                rows = cursor.execute("SELECT name FROM functions").fetchall()
                for r in rows:
                    self.symbol_queue.append(("function", r[0]))
                conn.close()
            except Exception as e:
                print(f"Error loading apis.db: {e}")

        # 2. Load Classes
        class_db = os.path.join(self.project_root, "class.db")
        if os.path.exists(class_db):
            try:
                conn = sqlite3.connect(class_db)
                cursor = conn.cursor()
                rows = cursor.execute("SELECT class_name FROM classes").fetchall()
                for r in rows:
                    self.symbol_queue.append(("class", r[0]))
                conn.close()
            except Exception as e:
                print(f"Error loading class.db: {e}")

        # Shuffle for random coverage
        random.shuffle(self.symbol_queue)
        print(f"Loaded {len(self.symbol_queue)} symbols for PHP generation.")

    def _ask_llm(self, template, symbol_type, symbol_name, previous_error=None):
        if not self.client:
            return None
            
        system_msg = (
            "You are a senior PHP Core developer writing unit tests.\n"
            "Return ONLY valid PHP source code.\n"
            "Do NOT use markdown code blocks.\n"
            "Do NOT output thinking tags or explanations.\n"
            "The generated code must be linear, flat, and concise.\n"
            "Do NOT define new functions or classes inside the test script."
        )

        user_prompt = f"""
Write a standalone PHP script to test the {symbol_type} `{symbol_name}`.

Requirements:
1. It must be valid PHP code (start with <?php).
2. Focus on edge cases and interesting inputs for `{symbol_name}`.
3. Do NOT use assertions (e.g. assert(), expect()). Use simple print statements or var_dump() for verification.
4. Do not use external libraries or frameworks (like PHPUnit).
5. **CRITICAL**: Keep the code linear and flat. Do NOT define helper functions or classes.
6. Avoid branching (if/else/loops) unless necessary for the logic.
7. Keep the test concise and small in size.
8. Output ONLY the code.

Template:
{template}

\\nothink
"""
        if previous_error:
            user_prompt += f"\nThe previous version failed syntax check with:\n{previous_error}\nCorrect it."

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }
            if self.provider != "openai":
                kwargs["extra_body"] = {"include_reasoning": False}

            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            
            # Cleaning
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            
            # Dynamically construct regex to avoid breaking the file generation
            tick = "`"
            # Matches ```optional_lang\n ...content... ```
            pattern = tick * 3 + r"(?:\w+)?\n(.*?)" + tick * 3
            content = re.sub(pattern, r"\1", content, flags=re.DOTALL)
            
            return content.strip()

        except Exception as e:
            print(f"[LLM Request Error] {e}")
            return None

    def _extract_php_code(self, phpt_content):
        """Extracts content of the --FILE-- section."""
        # Simple regex to get content between --FILE-- and the next section header
        match = re.search(r"--FILE--\s*\n(.*?)\n--[A-Z]+--", phpt_content, re.DOTALL)
        if match:
            return match.group(1)
        
        # Fallback: maybe --FILE-- is the last section (unlikely but possible in malformed)
        match = re.search(r"--FILE--\s*\n(.*)", phpt_content, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def generate(self, skip_ids=None):
        if skip_ids is None: skip_ids = set()

        symbol_type, symbol_name = None, None
        
        while self.symbol_queue:
            t, n = self.symbol_queue.pop()
            
            # Check ID uniqueness
            # ID Format: _genai_unit_test_for_php_function_array_push
            safe_name = safe_filename_from_symbol(n)
            uid = f"_genai_unit_test_for_php_{t}_{safe_name}".lower()
            
            if uid in skip_ids:
                continue
                
            symbol_type = t
            symbol_name = n
            break
        
        if not symbol_name:
            return None

        # Prepare Template
        template = PHP_TEMPLATE.format(symbol_type=symbol_type, symbol_name=symbol_name)
        
        previous_error = None
        
        for attempt in range(MAX_RETRIES):
            # 1. Generate
            php_content = self._ask_llm(template, symbol_type, symbol_name, previous_error)
            if not php_content or "<?php" not in php_content:
                continue

            # 2. Validate Syntax (php -l)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as tmp:
                tmp.write(php_content)
                tmp_path = tmp.name
            
            try:
                # Run syntax check
                res = subprocess.run(
                    [self.php_bin, "-l", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if res.returncode == 0 and "No syntax errors" in res.stdout:
                    # Valid!
                    safe_name = safe_filename_from_symbol(symbol_name)
                    final_id = f"_genai_unit_test_for_php_{symbol_type}_{safe_name}".lower()
                    
                    return Seed(
                        id=final_id,
                        content=php_content, 
                        metadata={
                            "type": "php_llm", # changed from phpt_llm
                            "identifier": final_id,
                            "symbol": symbol_name,
                            "symbol_type": symbol_type,
                            "description": f"LLM generated for PHP {symbol_type} {symbol_name}",
                            "extension": "php" # Hints it is a php file
                        }
                    )
                else:
                    previous_error = f"Syntax Error: {res.stdout} {res.stderr}"
                    
            except Exception as e:
                previous_error = str(e)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        return None

    def improve(self, seed_content):
        return None

def get_generator(config):
    return CustomLLMGenerator(config)