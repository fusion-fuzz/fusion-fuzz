import logging
import os
import requests
import json
import re
from .fusion import Seed

# Try to import OpenAI library
try:
    from openai import OpenAI
    HAS_OPENAI_LIB = True
except ImportError:
    HAS_OPENAI_LIB = False

logger = logging.getLogger("FFL.LLMGen")

class LLMGenerator:
    def __init__(self, config):
        self.config = config
        self.project_name = config.get("project_name", "unknown")
        
        # Configuration defaults
        llm_config = config.get("llm", {})
        self.provider = llm_config.get("provider", "gemini")  # Options: 'gemini', 'vllm', 'openai', 'ollama', 'deepseek'
        self.model = llm_config.get("model", "gemini-2.5-flash-preview-09-2025")
        self.api_key = llm_config.get("api_key", "")

        # Set default API base depending on provider
        if self.provider == "ollama":
            default_base = "http://localhost:11434"
        elif self.provider == "openai":
            default_base = "https://api.openai.com/v1"
        elif self.provider == "deepseek":
            default_base = "https://api.deepseek.com"
        else:
            default_base = "http://localhost:8000/v1"

        self.api_base = llm_config.get("api_base", default_base)
        self.timeout = llm_config.get("timeout", 60) # Increased timeout for reasoning models

        # Initialize OpenAI Client if applicable (DeepSeek is OpenAI API-compatible)
        # The last transport-level failure, so a caller can tell a rate
        # limit apart from a malformed request. Every _call_* below returns
        # None on failure, which on its own says nothing about whether
        # retrying would help.
        self.last_error = None
        self.last_status = None
        # Token usage of the last call, taken from the provider's own
        # accounting rather than estimated: a daily quota is counted by the
        # provider, so anything we compute ourselves would drift from the
        # number that actually matters.
        self.last_usage = None

        self.openai_client = None
        if self.provider in ("openai", "deepseek"):
            if HAS_OPENAI_LIB and self.api_key:
                try:
                    self.openai_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
            elif not HAS_OPENAI_LIB:
                logger.warning(f"{self.provider} provider selected but 'openai' library not found. Falling back to requests.")

    def _record_usage(self, data):
        """Pull token counts out of a response, whatever shape it has."""
        u = None
        if data is None:
            u = None
        elif isinstance(data, dict):
            u = data.get("usage") or data.get("usageMetadata")
        else:
            u = getattr(data, "usage", None)
        if u is None:
            self.last_usage = None
            return
        get = (u.get if isinstance(u, dict) else lambda k, d=0: getattr(u, k, d))
        prompt = (get("prompt_tokens", 0) or get("promptTokenCount", 0) or 0)
        completion = (get("completion_tokens", 0) or get("candidatesTokenCount", 0) or 0)
        total = (get("total_tokens", 0) or get("totalTokenCount", 0) or 0)
        self.last_usage = {
            "prompt": int(prompt or 0),
            "completion": int(completion or 0),
            "total": int(total or 0) or int(prompt or 0) + int(completion or 0),
        }

    def _record_error(self, exc):
        """Remember what went wrong, and the HTTP status when there is one.

        Providers surface a status in three different places depending on
        the client: a `requests` HTTPError carries `response.status_code`,
        the openai library raises objects with `status_code`, and some
        wrappers only put the number in the message."""
        self.last_error = exc
        status = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
        if status is None:
            status = getattr(exc, "status_code", None)
        if status is None:
            m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
            if m:
                status = int(m.group(1))
        self.last_status = status
        # Honour Retry-After when the server sends one.
        self.retry_after = None
        if resp is not None:
            try:
                ra = resp.headers.get("Retry-After")
                if ra:
                    self.retry_after = float(ra)
            except Exception:
                pass

    def _clean_response(self, text):
        """
        Removes Markdown code blocks if present.
        """
        tick = "`"
        pattern = tick * 3 + r"(?:\w+)?\n(.*?)" + tick * 3
        
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _call_gemini(self, prompt):
        if not self.api_key:
            logger.warning("Gemini API Key missing.")
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 4096
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self._record_usage(data)
            
            candidates = data.get("candidates", [])
            if not candidates:
                return None
                
            raw_text = candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
            return raw_text

        except Exception as e:
            self._record_error(e)
            logger.error(f"Gemini API Error: {e}")
            return None

    def _call_vllm(self, prompt):
        """
        Adapter for vLLM or OpenAI-compatible local APIs.
        """
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key if self.api_key else 'EMPTY'}"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
            "max_tokens": 4096
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self._record_usage(data)
            
            choices = data.get("choices", [])
            if not choices:
                return None
            
            raw_text = choices[0].get("message", {}).get("content", "")
            return raw_text

        except Exception as e:
            self._record_error(e)
            logger.error(f"vLLM API Error: {e}")
            return None

    def _call_ollama(self, prompt):
        url = f"{self.api_base.rstrip('/')}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,
                "num_predict": 4096
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self._record_usage(data)
            return data.get("response", "")

        except Exception as e:
            self._record_error(e)
            logger.error(f"Ollama API Error: {e}")
            return None

    def _call_openai(self, prompt):
        """
        Adapter for official OpenAI API, and any OpenAI-compatible API
        (e.g. DeepSeek) reachable via the 'openai' provider name.
        Handles O-series reasoning models (o1, o3) which have specific constraints.
        """
        provider_label = self.provider.capitalize()
        if not self.api_key:
            logger.error(f"{provider_label} API Key is missing. Please set LLM_API_KEY environment variable.")
            return None

        # Prefer using the official library if available
        if self.openai_client:
            # o-series reasoning models (o1, o1-mini, o3, o3-mini, o4-mini, …)
            is_reasoning = bool(re.match(r"^o\d", self.model)) or self.model == "deepseek-reasoner"

            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }

            if is_reasoning:
                # Reasoning models use max_completion_tokens and generally do not support temperature
                kwargs["max_completion_tokens"] = 8096
            else:
                kwargs["temperature"] = 0.5
                # kwargs["max_tokens"] = 4096

            try:
                response = self.openai_client.chat.completions.create(**kwargs)
                self._record_usage(response)
                return response.choices[0].message.content
            except Exception as e:
                self._record_error(e)
                logger.error(f"{provider_label} API Error (Library): {e}")
                return None

        # Fallback to manual requests if library is missing
        url = f"{self.api_base.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        is_reasoning = bool(re.match(r"^o\d", self.model)) or self.model == "deepseek-reasoner"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if is_reasoning:
            payload["max_completion_tokens"] = 8096
        else:
            payload["temperature"] = 0.9
            payload["max_tokens"] = 4096

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self._record_usage(data)

            choices = data.get("choices", [])
            if not choices:
                return None

            raw_text = choices[0].get("message", {}).get("content", "")
            return raw_text

        except Exception as e:
            self._record_error(e)
            logger.error(f"{provider_label} API Error (Requests): {e}")
            return None

    def _call_api(self, prompt):
        if self.provider == "gemini":
            raw_text = self._call_gemini(prompt)
        elif self.provider == "vllm":
            raw_text = self._call_vllm(prompt)
        elif self.provider == "ollama":
            raw_text = self._call_ollama(prompt)
        elif self.provider in ("openai", "deepseek"):
            raw_text = self._call_openai(prompt)
        else:
            logger.error(f"Unknown LLM provider: {self.provider}")
            return None

        if raw_text:
            return self._clean_response(raw_text)
        return None

    def refine(self, code: str, lang: str, avoid: list[str], extra_constraints: str = "") -> "Seed | None":
        """
        Rewrite *code* in *lang* to avoid specific patterns (e.g. 'ctypes')
        while preserving the original bug / behaviour pattern.

        Parameters
        ----------
        code              : existing (bad) translation to rewrite
        lang              : target language (e.g. "Python")
        avoid             : list of module/API names the output must not use
        extra_constraints : any additional free-text instruction appended to the prompt
        """
        avoid_str = ", ".join(f"`{a}`" for a in avoid)
        prompt = (
            f"You are refining a {lang} fuzzing seed that currently uses {avoid_str}.\n"
            f"These APIs are unsuitable for fuzzing the {lang} runtime and must be removed.\n\n"
            "Task: rewrite the program so that it:\n"
            f"1. Does NOT import or call any of: {avoid_str}.\n"
            "2. Preserves the *same bug pattern* — the specific runtime operation, "
            "edge case, or boundary condition that makes this test interesting — "
            f"using only the {lang} standard library.\n"
            "3. Is a complete, self-contained, runnable script.\n"
            "4. Contains no assertions, test-framework calls, or print statements "
            "that would cause it to exit non-zero on a correct implementation.\n"
            "5. Is as concise as possible; prefer sequential, linear code.\n"
        )
        if extra_constraints:
            prompt += f"\nAdditional constraints:\n{extra_constraints}\n"
        prompt += f"\nOutput ONLY the raw {lang} code — no markdown, no explanation.\n\n"
        prompt += f"Current code:\n{code}"

        refined = self._call_api(prompt)
        if refined:
            return Seed(
                content=refined,
                metadata={
                    "type": "llm_refined",
                    "model": self.model,
                    "provider": self.provider,
                    "avoided": avoid,
                    "description": f"Refined {lang} seed (removed {', '.join(avoid)})",
                },
            )
        return None

    def translate(self, code, source_lang, target_lang, previous_error=None):
        """
        Translates code from source language to target language.
        """
        prompt = (
            f"You are porting a bug-triggering test from {source_lang} to {target_lang}.\n"
            "Goal: transplant the *bug pattern* — the specific operation, edge case, or "
            "language-runtime interaction that makes this test interesting — into idiomatic "
            f"{target_lang}, rather than doing a literal line-by-line translation.\n\n"
            "Rules:\n"
            f"1. Identify the core bug pattern (e.g. integer overflow, type coercion, "
            "memory aliasing, JIT deopt trigger, boundary condition) and re-express it "
            f"using {target_lang} constructs that exercise the same class of behavior.\n"
            f"2. Use only {target_lang} standard library; remove any imports or APIs that "
            "have no equivalent.\n"
            "3. Produce a complete, self-contained, runnable script with a main entry point.\n"
            "4. Remove all assertions and test-framework calls — the script should simply "
            "execute the pattern and terminate without errors under a correct implementation.\n"
            "5. If a language feature has no direct analogue, choose the closest equivalent "
            "that stresses the same runtime subsystem.\n"
            f"6. Output ONLY the raw {target_lang} code — no markdown, no explanation, "
            "no comments about the translation.\n\n"
            f"7. If translate to CPython, avoid using `ctypes` or any FFI that would make the test unsuitable for fuzzing.\n\n"
            f"8. Last but not least, ensure the translated code is as concise as possible. "
            "Don't use function calls if not necessary, and prefer sequential code and linear control flow.\n\n"
            f"Source ({source_lang}):\n{code}"
        )
        
        if previous_error:
            prompt += f"\n\nThe previous translation failed validation with the following error:\n{previous_error}\nPlease fix the code to resolve this error."

        translated = self._call_api(prompt)

        if translated:
            return Seed(
                content=translated,
                metadata={
                    "type": "llm_translated",
                    "model": self.model,
                    "provider": self.provider,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "description": f"Translated from {source_lang}"
                }
            )
        return None
