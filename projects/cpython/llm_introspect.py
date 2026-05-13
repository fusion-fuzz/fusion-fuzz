import pkgutil
import sys
import inspect
import importlib
import os
import subprocess
import random
import tempfile
import re
from pathlib import Path
from core.fusion import Seed

# Try to import OpenAI, handle missing dependency gracefully
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

MAX_RETRIES = 3

unit_test_template = (
    "#THIS FILE IS UNIT TEST FOR {}\n"
    "#TODO: COMPLETE THE UNIT TEST WITH VALID STATEMENTS\n"
)

def safe_filename_from_symbol(symbol: str) -> str:
    bad_chars = ['<', '>', ':', '/', '\\', '*', '?', '"', '|', ' ']
    fname = symbol.replace(".", "_")
    for ch in bad_chars:
        fname = fname.replace(ch, "_")
    return fname

class CustomLLMGenerator:
    def __init__(self, config):
        self.config = config
        self.modules = set()
        self.symbol_queue = []
        self.python_bin = self._find_python_binary()
        
        # --- LLM Configuration ---
        llm_conf = config.get("llm", {})
        self.provider = llm_conf.get("provider", "openai")
        self.model = llm_conf.get("model", "gpt-4o-mini")
        
        # Determine API Key
        env_key = os.getenv("LLM_API_KEY")
        conf_key = llm_conf.get("api_key")

        # Set defaults based on provider
        if self.provider == "vllm":
            self.api_key = conf_key or env_key or "EMPTY"
            self.api_base = llm_conf.get("api_base", "http://localhost:8008/v1") 
        elif self.provider == "ollama":
            self.api_key = conf_key or env_key or "ollama"
            self.api_base = llm_conf.get("api_base", "http://localhost:11434/v1")
        else: # openai or gemini (via openai compat)
            self.api_key = conf_key or env_key
            self.api_base = llm_conf.get("api_base") # None means default OpenAI URL

        self.client = None
        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            except Exception as e:
                print(f"[LLM Init Error] Could not initialize OpenAI client: {e}")
        else:
            if not HAS_OPENAI:
                print("[LLM Warning] 'openai' package not installed. LLM features disabled.")
            elif not self.api_key:
                print("[LLM Warning] No API key provided. LLM features disabled.")

        # Populate initial module list
        self._list_all_importable_modules()
        self.module_list = list(self.modules)
        # to order the list
        self.module_list.sort()
        
    def _find_python_binary(self):
        # Try to find the built CPython binary
        project_root = os.path.abspath(os.path.join("projects", "cpython"))
        built_bin = os.path.join(project_root, "cpython", "build", "python")
        if os.path.exists(built_bin):
            return built_bin
        return sys.executable

    def _list_all_importable_modules(self):
        """
        Lists only standard library and built-in modules, excluding 
        pip-installed packages (site-packages/dist-packages).
        """
        # 1. Add built-in modules (no file path)
        for name in sys.builtin_module_names:
            if name not in self.modules and not name.startswith("_"):
                self.modules.add(name)

        # 2. Add standard library modules (file-based)
        for finder, name, ispkg in pkgutil.iter_modules(sys.path):
            if name in self.modules or name.startswith("_"):
                continue
            
            # Filter out external packages based on path
            # finder usually has a 'path' attribute representing the directory being scanned
            try:
                path = getattr(finder, "path", "")
                if "site-packages" in path or "dist-packages" in path:
                    continue
            except AttributeError:
                pass

            self.modules.add(name)

    def _dump_module_info(self, mod):
        apis = []
        classes = []
        internals = []
        name = mod.__name__
        all_attrs = mod.__dict__

        public_names = list(getattr(mod, "__all__", []))
        if public_names:
            for n in sorted(public_names):
                apis.append(f"{name}.{n}")

        internal_attrs = {k: v for k, v in all_attrs.items() if k not in public_names}

        for k, v in internal_attrs.items():
            full_name = f"{name}.{k}"
            if inspect.isclass(v):
                classes.append(full_name)
            elif inspect.isfunction(v) or inspect.isbuiltin(v):
                internals.append(full_name)
            elif inspect.ismodule(v):
                internals.append(full_name)
            elif not callable(v) and not inspect.ismodule(v):
                if not (k.startswith("__") and k.endswith("__")):
                    internals.append(full_name)

        return apis + classes + internals

    def _ask_llm(self, template, module_name, symbol_name, previous_error=None):
        if not self.client:
            return None
            
        system_msg = (
            "You are a senior Python engineer writing small Python scripts "
            "that exercise a given symbol without using any test framework.\n"
            "Return ONLY valid Python source code for a single file.\n"
            "Do NOT use markdown, backticks, or explanations.\n"
            "Do NOT output thinking process or reasoning traces (e.g. <think> tags).\n"
            "Do NOT import or use 'unittest', 'pytest', or any other testing framework.\n"
            "Do NOT use 'assert' statements or any assertion mechanism.\n"
            "Prefer a flat script structure: top-level code only if possible."
        )

        user_prompt = f"""
You are given a partial Python file.
- The code should focus on the symbol `{symbol_name}` from module `{module_name}`.
- Keep the existing header comments and import statements.
- Do NOT import or use the `unittest` module or any other test framework.
- Do NOT use any `assert` statements.
- Use only standard-library APIs.
- Simple print statements and benign exception handling are fine.
- Do NOT add an `if __name__ == "__main__":` block.
- Return the COMPLETE Python file.

Partial file:
{template}

\\nothink
"""
        if previous_error:
            user_prompt += f"\nThe previous version failed with:\n{previous_error}\nCorrect it."

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.5,
            }
            
            # For non-OpenAI providers (like vLLM/Ollama), explicitly request no reasoning
            # if supported by the backend via extra_body
            if self.provider != "openai":
                kwargs["extra_body"] = {"include_reasoning": False}

            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            
            # Post-processing to remove <think> tags if any leaked
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()

        except Exception as e:
            print(f"[LLM Request Error] {e}")
            return None

    def _refill_queue(self):
        """Populate symbol queue from a random module."""
        if not self.module_list:
            self._list_all_importable_modules()
            self.module_list = list(self.modules)
            # no random shuffle here to preserve order
            
        if not self.module_list:
            return # No modules found?

        mod_name = self.module_list.pop()
        try:
            # print(f"Introspecting module: {mod_name}")
            mod = importlib.import_module(mod_name)
            symbols = self._dump_module_info(mod)
            self.symbol_queue.extend([(mod_name, s) for s in symbols])
            # Randomize symbols within the module to avoid sequential bias
            random.shuffle(self.symbol_queue)
        except Exception as e:
            pass # Skip module if import fails

    def generate(self, skip_ids=None):
        """
        Generates and validates a single seed.
        Returns a Seed object or None.
        """
        if skip_ids is None:
            skip_ids = set()

        if not self.symbol_queue:
            self._refill_queue()
            
        if not self.symbol_queue:
            return None

        # Loop until we find a symbol that hasn't been generated or queue is empty
        module_name, symbol = None, None
        while self.symbol_queue:
            m_name, sym = self.symbol_queue.pop()

            # check identifier uniqueness
            unit_test_identifier = f"_genai_unit_test_for_{safe_filename_from_symbol(sym)}"
            unit_test_identifier = unit_test_identifier.replace("__", "_")
            unit_test_identifier = unit_test_identifier.replace("__", "_").lower()

            if unit_test_identifier in skip_ids:
                continue
            
            # Found new one
            module_name = m_name
            symbol = sym
            break
        
        if not module_name:
            return None

        base_template = (
            unit_test_template.format(symbol)
            + f"import {module_name}\n"
            + f"#{symbol}\n"
        )

        previous_error = None
        
        for attempt in range(MAX_RETRIES):
            # 1. Generate
            code = self._ask_llm(base_template, module_name, symbol, previous_error)
            if not code:
                continue
                
            # 2. Validate
            if "def " in code:
                # Avoid generating functions, we want top-level scripts
                previous_error = "Generated code contains function definitions."
                continue

            # Write to temp file to execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            try:
                result = subprocess.run(
                    [self.python_bin, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    # Re-calculate ID for saving
                    unit_test_identifier = f"_genai_unit_test_for_{safe_filename_from_symbol(symbol)}"
                    unit_test_identifier = unit_test_identifier.replace("__", "_").lower()
                    
                    return Seed(
                        id=unit_test_identifier,
                        content=code,
                        metadata={
                            "type": "cpython_llm",
                            "identifier": unit_test_identifier,
                            "description": f"The unit test generated by LLM for module {module_name}.{symbol} in CPython"
                        }
                    )
                else:
                    previous_error = f"RC={result.returncode}\nSTDERR:\n{result.stderr[:500]}"
            except Exception as e:
                previous_error = str(e)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        return None

    def improve(self, seed_content):
        # Stub for improvement, can implement similar logic using LLM
        return None

def get_generator(config):
    return CustomLLMGenerator(config)