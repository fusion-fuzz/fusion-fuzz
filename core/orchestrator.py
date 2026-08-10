import logging
import random
import time
import re
import os
import hashlib
import shutil
import tempfile
import glob
import subprocess
import sys
import datetime
import stat
import gc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Tuple, Optional
from .driver import get_driver, DockerDriver, cleanup_stale_processes
from .fusion import Seed
from .utils import smart_truncate
from .coverage import PairwiseCoverageMatrix

logger = logging.getLogger("FFL.Orchestrator")


def _extract_display_code(content: str, ext: str) -> tuple[str, str]:
    """
    Return (code_for_display, markdown_lang_hint) for README / HTML rendering.

    .phpt files  — extracts the ``--FILE--`` section (the actual PHP code) and
                   returns ``"php"`` as the language hint so syntax highlighters
                   work correctly.  Falls back to the full content when the
                   section is absent.
    All others   — returns content unchanged; lang hint is the extension without
                   the leading dot (e.g. ".py" → "python" is NOT done here —
                   callers that need a highlight.js name should remap separately).
    """
    if ext == ".phpt":
        in_file = False
        lines = []
        for line in content.splitlines(keepends=True):
            tag = line.strip()
            if tag.startswith("--") and tag.endswith("--"):
                section = tag[2:-2]
                if section == "FILE":
                    in_file = True
                    continue
                elif in_file:
                    break
            if in_file:
                lines.append(line)
        extracted = "".join(lines).strip()
        return (extracted if extracted else content), "php"
    return content, ext.lstrip(".")

class FusionFuzzLoop:
    def __init__(self, config, strategies, initial_corpus, pre_analysis_enabled=True,
                 guided_fusion_enabled=False, fusion_rate=0.8):
        self.config = config
        self.strategies = strategies
        self.fusion_rate = fusion_rate

        # Sanitize corpus: Ensure everything is a Seed object
        self.corpus = []
        if initial_corpus:
            for item in initial_corpus:
                if isinstance(item, Seed):
                    self.corpus.append(item)
                elif isinstance(item, str):
                    # Auto-wrap raw strings
                    self.corpus.append(Seed(content=item, metadata={"type": "raw"}))
                elif isinstance(item, dict):
                    # Auto-convert dictionaries (e.g. from JSON/DB loaders)
                    self.corpus.append(Seed(
                        content=item.get("content", ""), 
                        metadata=item.get("metadata", {})
                    ))
                else:
                    logger.warning(f"Skipping unknown corpus item type: {type(item)}")
        
        # Replay mode (--execute) constructs the loop with no corpus and no
        # strategies on purpose — only warn when fusion actually needs one.
        if not self.corpus and strategies:
            logger.warning("Corpus is empty after initialization!")

        # Initialize the specific driver (e.g., PHPDriver) via factory
        self.driver = get_driver(config)
        self.iterations = 0
        self.project_name = config.get("project_name", "default_project")

        # Track unique crash signatures to avoid duplicates
        self.unique_crashes = set()
        self._load_existing_crashes()

        # Pairwise conjunction coverage matrix — producer-consumer guided
        # parent selection (--guided-fusion) only takes effect when
        # --pre-analysis is also enabled (see PairwiseCoverageMatrix's
        # docstring).
        self.coverage = PairwiseCoverageMatrix(pre_analysis_enabled=pre_analysis_enabled,
                                                guided_fusion_enabled=guided_fusion_enabled)
        
        self.original_cwd = os.getcwd()
        self.current_workspace = None
        
        # Stats tracking initialized here to avoid AttributeErrors
        self.start_time = time.time()
        self.syntax_error_count = 0
        self.sample_count = 0
        self.last_status_print = 0

        # Sample log (set in run())
        self.sample_log_path = None
        self._sample_log_file = None

    def _load_existing_crashes(self):
        """
        Scans the output directory for existing crash reports and loads their signatures
        to prevent duplicate reporting across runs.
        """
        bugs_dir = os.path.join("output", "bugs", self.project_name)
        if not os.path.exists(bugs_dir):
            return

        # Matches **Signature:** `…` in both the new README.md inline format
        # and the old report.md bullet-list format.
        sig_pattern = re.compile(r"\*\*Signature:\*\*\s*`([^`]+)`")
        
        loaded_count = 0
        # Check all markdown files in the bugs directory (recursive to handle bundle folders)
        for root, _, files in os.walk(bugs_dir):
            for file in files:
                if file.endswith(".md"):
                    report_file = os.path.join(root, file)
                    try:
                        with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
                            # Read the first 1KB which usually contains the metadata
                            content = f.read(1024)
                            match = sig_pattern.search(content)
                            if match:
                                signature = match.group(1).strip()
                                self.unique_crashes.add(signature)
                                loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to read existing crash report {report_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} existing crash signatures from disk.")

    def select_parents(self):
        """Select two parents from the corpus, preferring unfused pairs."""
        return self.coverage.select_parents(self.corpus)

    def _pick_strategy_chain(self, pool: List) -> List:
        """
        Choose a subset of `pool` to combine on the same parent pair: each
        strategy is included independently with probability
        self.fusion_rate (--fusion-rate, default 0.8). If every draw comes
        up empty, one strategy is picked uniformly at random so every
        iteration still applies at least one technique. The resulting
        chain is shuffled — any included strategy can end up last (only
        the last strategy in the chain runs bidirectionally; see
        process_iteration). With a pool of size 1 (e.g. --dataflow-fusion
        passed alone) this always degenerates to a chain of length 1,
        matching the old single-strategy behavior, regardless of
        fusion_rate.
        """
        chain = [s for s in pool if random.random() < self.fusion_rate]
        if not chain:
            chain = [random.choice(pool)]
        random.shuffle(chain)
        return chain

    def _fuse_once(self) -> Tuple[List[Seed], List, Tuple[Optional[Seed], Optional[Seed]]]:
        """
        Selection + fusion for one iteration, without execution.

        Returns (children, strategy_chain, (parent_a, parent_b)); children
        is empty when the selected pair had no viable strategy or fusion
        raised. The parents returned are the two corpus seeds picked for
        this iteration — a child's own metadata["parents"] may instead
        name an intermediate seed produced earlier in the chain.

        Fusion combines techniques rather than picking a single one:
        _pick_strategy_chain layers 1-3 of the pool's strategies onto the
        same parent pair. Every technique before the last in the chain
        contributes one single-direction intermediate fuse (host <- donor)
        that becomes the next technique's host; only the last technique
        runs bidirectionally (producing both ab/ba children), if it
        supports that.
        """
        parent_a, parent_b = self.select_parents()

        if not parent_a or not parent_b:
            return [], [], (parent_a, parent_b)

        # Drop strategies that would just be a no-op on this specific pair
        # (e.g. clang declaration fusion when one side has nothing to
        # donate — see FusionStrategy.is_viable_pair) instead of spending a
        # full compile on a syntactically-unchanged child. If that leaves
        # nothing usable (e.g. the pool is declaration-fusion-only and this
        # pair truly has no declarations on either side), skip the
        # iteration rather than force an unviable technique.
        usable = [s for s in self.strategies if s.is_viable_pair(parent_a, parent_b)]
        if not usable:
            return [], [], (parent_a, parent_b)

        try:
            chain = self._pick_strategy_chain(usable)
            host = parent_a
            for strategy in chain[:-1]:
                host = strategy.fuse(host, parent_b)

            final_strategy = chain[-1]
            if hasattr(final_strategy, 'fuse_bidirectional'):
                children = final_strategy.fuse_bidirectional(host, parent_b)
            else:
                children = [final_strategy.fuse(host, parent_b)]
        except Exception as e:
            logger.warning(f"Fusion error: {e}")
            return [], [], (parent_a, parent_b)

        return [c for c in children if c], chain, (parent_a, parent_b)

    def process_iteration(self) -> List[Tuple[Optional[Seed], Optional[object]]]:
        """
        Worker function to handle Selection, Fusion, and Execution.
        Executed in a separate thread.
        """
        children, _chain, _parents = self._fuse_once()
        if not children:
            return [(None, None)]

        # 3. Execute
        pairs = []
        for child in children:
            try:
                result = self.driver.execute(child)
                pairs.append((child, result))
            except Exception as e:
                logger.error(f"Execution driver error: {e}")
                pairs.append((None, None))
        return pairs

    def _extract_crash_signature(self, result) -> Optional[str]:
        """
        Extracts a unique, stable signature for a crash to aid in deduplication.
        Prioritizes the most specific location info available.
        Returns None only if truly nothing can be extracted.
        """
        combined = (result.stderr or "") + "\n" + (result.stdout or "")

        # 1. Driver-provided signature (project specific)
        if hasattr(result, 'signature') and result.signature:
            return result.signature

        # 2. AddressSanitizer SUMMARY line
        # "SUMMARY: AddressSanitizer: heap-use-after-free ... in foo /path/file.cpp:123"
        m = re.search(r"SUMMARY: AddressSanitizer:\s+(\S+).*?in (\S+)\s+(\S+:\d+)", combined)
        if m:
            return f"ASAN:{m.group(1)}_in_{m.group(2)}_at_{m.group(3)}"
        m = re.search(r"SUMMARY: AddressSanitizer:\s+(.*)", combined)
        if m:
            return f"ASAN:{m.group(1).strip()[:120]}"

        # 3. UndefinedBehaviorSanitizer
        m = re.search(r"SUMMARY: UndefinedBehaviorSanitizer:\s+(.*)", combined)
        if m:
            return f"UBSAN:{m.group(1).strip()[:120]}"

        # 3b. Rust internal compiler error (ICE) — keep the whole diagnostic
        # line verbatim, e.g.:
        #   "internal compiler error: /rustc-dev/<rev>/compiler/.../layout.rs:192:13: layout_of: unexpected const: {const error}"
        # This already carries the exact source file:line:col of the failing
        # compiler assertion plus the actual bug detail, so it's more
        # specific than the generic "panicked at ...: Box<dyn Any>" line
        # that follows it in stderr — check this first so that one wins.
        m = re.search(r"internal compiler error:.*", combined)
        if m:
            return m.group(0).strip()[:200]

        # 4. Rust panic — extract file:line (stable across duplicate triggers)
        # "thread 'main' panicked at src/foo.rs:123:45:\nmessage"
        m = re.search(r"panicked at ([^:]+\.rs:\d+:\d+):\s*\n?(.*)", combined)
        if m:
            location = m.group(1).strip()
            msg = m.group(2).strip()[:80]
            return f"RustPanic:{location}:{msg}"

        # 5. C/C++ assertion failure
        m = re.search(r"Assertion `(.{1,120})' failed", combined)
        if m:
            return f"Assertion:{m.group(1).strip()}"

        # 6. LLVM/MLIR assertion  (e.g. "Assertion failed: (cond), function ...")
        m = re.search(r"Assertion failed: \((.{1,120})\)", combined)
        if m:
            return f"LLVMAssert:{m.group(1).strip()}"

        # 7. MLIR / LLVM error with file:line
        m = re.search(r"llvm_unreachable executed at ([^:]+:\d+)", combined)
        if m:
            return f"Unreachable:{m.group(1).strip()}"
        m = re.search(r"fatal error: error in backend: (.{1,100})", combined)
        if m:
            return f"BackendError:{m.group(1).strip()}"

        # 8. naga / wgslc WGSL parse / validation errors — use first error line
        # e.g. "error: could not parse WGSL\n  --> file:10:5\n  |\n10| bad code\n     ^^^^ message"
        m = re.search(r"error:\s+(.{4,120})", combined, re.IGNORECASE)
        if m:
            msg = m.group(1).strip()
            # Normalize "Cannot redeclare function/class/method X" — strip specific name.
            redecl = re.match(r"(Cannot redeclare \w+)", msg, re.IGNORECASE)
            if redecl:
                return f"Error:{redecl.group(1)}"
            # Strip absolute paths so /tmp/abc123/foo.php doesn't create unique dirs.
            msg = re.sub(r'/\S+', '', msg).strip()
            msg = msg[:100]
            # Try to also grab file:line for better specificity
            loc = re.search(r"(?:-->|at)\s+[^:]+:(\d+):\d+", combined)
            if loc:
                return f"Error:{msg}_L{loc.group(1)}"
            return f"Error:{msg}"

        # 9. Signal / segfault (generic)
        m = re.search(r"signal:\s+(\w+)", combined, re.IGNORECASE)
        if m:
            return f"Signal:{m.group(1)}"
        if "Segmentation fault" in combined or "SIGSEGV" in combined:
            return "SIGSEGV"
        if "Aborted" in combined or "SIGABRT" in combined:
            return "SIGABRT"

        # 10. Fallback — hash of first non-empty stderr line for minimal grouping
        for line in (result.stderr or "").splitlines():
            line = line.strip()
            if line and not line.startswith("="):
                return f"Stderr:{line[:120]}"

        return None




    def _seed_extension(self, seed=None) -> str:
        """File extension to use when writing `seed` out for this project."""
        meta = getattr(seed, "metadata", None) or {}
        ext = ".txt"
        if "php" in self.project_name: ext = ".phpt"
        elif "cpython" in self.project_name or "python" in self.project_name: ext = ".py"
        elif "mlir" in self.project_name: ext = ".mlir"
        elif "swift" in self.project_name: ext = ".swift"
        elif "rust" in self.project_name: ext = ".rs"
        elif "clang" in self.project_name or "gcc" in self.project_name:
            # C/C++/Obj-C are all valid here — a blanket ".c" mislabels C++
            # seeds (test.sh still records the real "clang++ ... -std=c++20"
            # invocation against a file that no longer exists under that
            # name, so the saved reproducer silently can't be re-run).
            ext = meta.get("extension") or ".c"
        elif "flang" in self.project_name or "fortran" in self.project_name:
            ext = meta.get("extension") or ".f90"
        elif "haskell" in self.project_name: ext = ".hs"
        return ext

    def _fuse_pair(self, host_seed, donor_seed):
        """
        Fuse one ordered pair (host <- donor) with a strategy chain, no
        execution. Returns (child, chain); child is None when no strategy
        is viable for the pair or fusion raised. Single-direction only —
        the caller enumerates the reverse pair separately.
        """
        usable = [s for s in self.strategies
                  if s.is_viable_pair(host_seed, donor_seed)]
        if not usable:
            return None, []
        chain = self._pick_strategy_chain(usable)
        try:
            host = host_seed
            for strategy in chain:
                host = strategy.fuse(host, donor_seed)
        except Exception as e:
            logger.warning(f"Fusion error: {e}")
            return None, chain
        return host, chain

    def generate_only(self, output_dir, max_programs=-1, self_fusion=True):
        """
        Fusion-only mode (--save-to): run the same strategy-chain fusion as
        fuzzing, but write each fused program to `output_dir` instead of
        executing it. No driver, no workspace, no crash triage.

        Every ordered pair of corpus seeds is fused exactly once — (a, b)
        and (b, a) are separate programs, and (a, a) too unless
        self_fusion is False — so an N-seed corpus yields exactly N*N
        files (N*(N-1) without self-fusion), with no duplicated or missing
        pairs. Pair order is shuffled so a run cut short by max_programs
        (--iterations) or Ctrl-C still samples the whole corpus rather
        than just its first seeds.

        max_programs caps how many programs are written (-1 = all pairs).
        Alongside the programs a manifest.jsonl records, per file, the
        child seed id, the ordered (host, donor) parents and the fusion
        strategies applied.
        """
        import json

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.jsonl")

        n = len(self.corpus)
        if n == 0:
            logger.error("Corpus is empty — nothing to fuse.")
            return 0

        total_pairs = n * n if self_fusion else n * (n - 1)
        planned = total_pairs if max_programs == -1 else min(max_programs, total_pairs)

        logger.info(
            f"Fusion-only mode: {n} seeds -> {total_pairs} ordered pair(s)"
            + ("" if self_fusion else " (self-fusion excluded)")
            + f"; writing {planned} fused program(s) for {self.project_name} "
              f"to {output_dir}"
        )
        if max_programs != -1 and max_programs > total_pairs:
            logger.warning(
                f"--iterations {max_programs} exceeds the {total_pairs} "
                f"available ordered pairs; writing {total_pairs}."
            )

        # Materializing the pair list costs ~16 bytes/pair; at 100 seeds
        # that is nothing, but a full project corpus (tens of thousands of
        # seeds) would be hundreds of millions of pairs. Refuse rather than
        # exhaust memory — the user can narrow with --corpus-size.
        if total_pairs > 20_000_000:
            logger.error(
                f"{total_pairs} ordered pairs from {n} seeds is too many to "
                "enumerate. Narrow the corpus with --corpus-size (e.g. "
                "--corpus-size 100 --diverse) and re-run."
            )
            return 0

        pairs = [(i, j) for i in range(n) for j in range(n)
                 if self_fusion or i != j]
        random.shuffle(pairs)
        if max_programs != -1:
            pairs = pairs[:max_programs]

        # fused_* files are written from index 0 every run, so a previous
        # run's manifest would no longer describe what's on disk — start it
        # fresh and say so when we're overwriting an earlier batch.
        stale = glob.glob(os.path.join(output_dir, "fused_*"))
        if stale:
            logger.warning(
                f"{len(stale)} fused_* file(s) already in {output_dir} — "
                "they will be overwritten and manifest.jsonl reset."
            )

        self.start_time = time.time()
        saved = 0
        failed = 0
        manifest = open(manifest_path, "w", encoding="utf-8")

        try:
            for idx, (i, j) in enumerate(pairs):
                host_seed, donor_seed = self.corpus[i], self.corpus[j]
                self.iterations += 1
                self.coverage.record(host_seed.id, donor_seed.id)

                child, chain = self._fuse_pair(host_seed, donor_seed)
                if child is None or not (child.content or "").strip():
                    failed += 1
                    manifest.write(json.dumps({
                        "file": None,
                        "error": "no viable strategy or fusion failed",
                        "strategies": [type(s).__name__ for s in chain],
                        "parents": [
                            {"id": host_seed.id,
                             "filename": host_seed.metadata.get("filename"),
                             "role": "host"},
                            {"id": donor_seed.id,
                             "filename": donor_seed.metadata.get("filename"),
                             "role": "donor"},
                        ],
                    }) + "\n")
                    continue

                ext = self._seed_extension(child)
                filename = f"fused_{idx:06d}{ext}"
                try:
                    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                        f.write(child.content)
                except Exception as e:
                    logger.error(f"Failed to write {filename}: {e}")
                    failed += 1
                    continue

                manifest.write(json.dumps({
                    "file": filename,
                    "id": child.id,
                    "strategies": [type(s).__name__ for s in chain],
                    # Ordered: the first parent hosts, the second donates.
                    "parents": [
                        {"id": host_seed.id,
                         "filename": host_seed.metadata.get("filename"),
                         "role": "host"},
                        {"id": donor_seed.id,
                         "filename": donor_seed.metadata.get("filename"),
                         "role": "donor"},
                    ],
                    "self_fusion": i == j,
                }) + "\n")
                saved += 1

                if saved % 100 == 0:
                    manifest.flush()
                    elapsed = time.time() - self.start_time
                    sys.stdout.write(
                        f"\rGenerated: {saved}/{len(pairs)} | failed: {failed} | "
                        f"{elapsed:.0f}s"
                    )
                    sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            logger.info("Generation interrupted by user.")
        finally:
            manifest.close()

        sys.stdout.write("\n")
        logger.info(
            f"Wrote {saved} fused program(s) to {output_dir} "
            f"(of {len(pairs)} ordered pairs attempted"
            + (f", {failed} produced no program" if failed else "")
            + f"); manifest: {manifest_path}"
        )
        return saved

    def _save_crash_bundle(self, seed, result, signature):
        """
        Creates a dedicated folder for the crash and saves:
          test.<ext>     — original reproducer
          min.<ext>      — minimized reproducer (placeholder, same content until minimizer runs)
          test.out       — combined stdout + stderr
          test.sh        — reproducing shell command
          parent_a.<ext> — parent A program (if available)
          parent_b.<ext> — parent B program (if available)
          README.md      — human-readable bug report
        """
        # 1. Sanitize signature for folder name
        safe_sig = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', signature)
        if len(safe_sig) > 300:
            # Truncating alone can make two distinct signatures collide on the
            # same folder name (their differing suffix falls past the cut),
            # silently overwriting one crash's saved bundle with another's.
            # Append a short hash of the full signature to keep them distinct.
            sig_hash = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:8]
            safe_sig = f"{safe_sig[:291]}_{sig_hash}"

        # Use the sanitized signature as the folder name — no seed-ID suffix.
        # The signature is already unique per distinct crash; appending the ID
        # would create a separate folder for every instance of the same bug.
        folder_name = safe_sig
        if signature.startswith("ASAN:memory-leak"):
            crash_dir = os.path.join(self.original_cwd, "output", "bugs", self.project_name, "leaks", folder_name)
        else:
            crash_dir = os.path.join(self.original_cwd, "output", "bugs", self.project_name, folder_name)

        if not os.path.exists(crash_dir):
            os.makedirs(crash_dir, exist_ok=True)

        # 2. Determine file extension
        ext = self._seed_extension(seed)

        test_filename = f"test{ext}"

        # 3. Write test.<ext> — original reproducer
        try:
            with open(os.path.join(crash_dir, test_filename), "w", encoding="utf-8") as f:
                f.write(seed.content)
        except Exception as e:
            logger.error(f"Failed to write {test_filename}: {e}")

        # 3b. For .phpt files also write test.php (the extracted --FILE-- section)
        if ext == ".phpt":
            php_code, _ = _extract_display_code(seed.content, ext)
            try:
                with open(os.path.join(crash_dir, "test.php"), "w", encoding="utf-8") as f:
                    f.write(php_code)
            except Exception as e:
                logger.error(f"Failed to write test.php: {e}")

        # 4. Write min.<ext> — minimized reproducer (initially same as test.<ext>)
        try:
            with open(os.path.join(crash_dir, f"min{ext}"), "w", encoding="utf-8") as f:
                f.write(seed.content)
        except Exception as e:
            logger.error(f"Failed to write min{ext}: {e}")

        # 4b. For .phpt files also write min.php (initially same as test.php)
        if ext == ".phpt":
            php_code, _ = _extract_display_code(seed.content, ext)
            try:
                with open(os.path.join(crash_dir, "min.php"), "w", encoding="utf-8") as f:
                    f.write(php_code)
            except Exception as e:
                logger.error(f"Failed to write min.php: {e}")

        # 5. Write test.out — combined stdout + stderr, truncated around the
        #    crash signature so infinite-loop output doesn't create gigabyte files.
        combined_output = ""
        if result.stderr:
            combined_output += result.stderr.rstrip("\n") + "\n"
        if result.stdout:
            combined_output += result.stdout.rstrip("\n") + "\n"
        try:
            with open(os.path.join(crash_dir, "test.out"), "w", encoding="utf-8") as f:
                f.write(smart_truncate(combined_output))
        except Exception as e:
            logger.error(f"Failed to write test.out: {e}")

        # 6. Look up and save parent programs (parent_a.<ext>, parent_b.<ext>, …)
        parent_ids = seed.metadata.get("parents", [])
        corpus_index = {s.id: s for s in self.corpus}
        parent_labels = ["a", "b", "c", "d"]
        parent_seeds = []
        for i, pid in enumerate(parent_ids):
            parent = corpus_index.get(pid)
            if parent is None:
                continue
            label = parent_labels[i] if i < len(parent_labels) else str(i)
            parent_filename = f"parent_{label}{ext}"
            try:
                with open(os.path.join(crash_dir, parent_filename), "w", encoding="utf-8") as f:
                    f.write(parent.content)
            except Exception as e:
                logger.error(f"Failed to write {parent_filename}: {e}")
            parent_seeds.append((label, pid, parent))

        # 7. Write test.sh — reproducing shell command
        cmd_template = self.config.get('execution', {}).get('command', 'unknown_cmd {seed_path}')

        sh_lines = ["#!/bin/bash", 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"']

        if hasattr(result, 'command') and result.command:
            src_cmd = result.command
            if hasattr(result, 'full_command') and result.full_command:
                src_cmd = result.full_command

            if isinstance(self.driver, DockerDriver):
                # Find the container seed path: /workspace/.fused/<seed_id>.<container_ext>
                m = re.search(rf'/workspace/\.fused/{re.escape(seed.id)}(\.[^\s/]+)', src_cmd)
                if m:
                    container_ext = m.group(1)                      # e.g. ".php"
                    container_repro = f'/workspace/.fused/ffl_repro{container_ext}'
                    host_file = f'test{container_ext}'              # matches the extracted file saved above
                    exec_cmd = src_cmd.replace(
                        f'/workspace/.fused/{seed.id}{container_ext}', container_repro)
                else:
                    container_repro = f'/workspace/.fused/ffl_repro{ext}'
                    host_file = test_filename
                    exec_cmd = src_cmd.replace(seed.id, 'ffl_repro')
                sh_lines.append(
                    f'docker cp "$SCRIPT_DIR/{host_file}" {self.driver.container_name}:{container_repro}')
                sh_lines.append(exec_cmd)
            else:
                # Non-Docker driver: replace the full absolute seed path.
                # Use the extension from result.seed_file (the file actually executed)
                # rather than the project-level ext, so e.g. PHP writes test.php not test.phpt.
                if hasattr(result, 'seed_file') and result.seed_file:
                    actual_ext = os.path.splitext(result.seed_file)[1]
                    sh_fname = f"test{actual_ext}"
                    quoted = '"$SCRIPT_DIR/' + sh_fname.replace('"', '\\"') + '"'
                    sh_lines.append(src_cmd.replace(result.seed_file, quoted))
                else:
                    quoted = '"$SCRIPT_DIR/' + test_filename.replace('"', '\\"') + '"'
                    sh_lines.append(src_cmd.replace(seed.id, quoted))
        else:
            # Fallback: expand the config command template with a portable host path
            sh_lines.append(cmd_template.replace('{seed_path}', '"$SCRIPT_DIR/' + test_filename + '"'))

        repro_cmd = "\n".join(sh_lines)

        try:
            with open(os.path.join(crash_dir, "test.sh"), "w", encoding="utf-8") as f:
                f.write(repro_cmd + "\n")
            os.chmod(os.path.join(crash_dir, "test.sh"), 0o755)
        except Exception as e:
            logger.error(f"Failed to write test.sh: {e}")

        # 8. Write README.md — human-readable bug report
        # For phpt files, extract just the --FILE-- section so the README shows
        # clean PHP code rather than the full test harness.
        display_code, lang_hint = _extract_display_code(seed.content, ext)
        readme_path = os.path.join(crash_dir, "README.md")
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write("*Fusion-Fuzz Bug Report*\n\n")
                f.write(f"**ID:** `{seed.id}` &nbsp;·&nbsp; "
                        f"**Signature:** `{signature}` &nbsp;·&nbsp; "
                        f"**RC:** `{result.return_code}`\n\n")

                f.write("The following code:\n\n")
                f.write(f"```{lang_hint}\n{display_code}\n```\n\n")

                f.write("Resulted in this output:\n\n")
                f.write(f"```\n{combined_output.rstrip()}\n```\n\n")

                f.write("To reproduce:\n\n")
                f.write(f"```\n{repro_cmd}\n```\n\n")

                # Parent provenance
                if parent_seeds:
                    f.write("### Parents\n\n")
                    f.write("| Label | ID | Source |\n")
                    f.write("|-------|----|--------|\n")
                    for label, pid, parent in parent_seeds:
                        meta = parent.metadata or {}
                        ptype = meta.get("type", "")
                        if ptype == "bug_corpus":
                            src_proj = meta.get("source_project", "?")
                            src_name = meta.get("source_name", "?")
                            source_desc = f"Bug corpus (project: `{src_proj}`, name: `{src_name}`)"
                        else:
                            identifier = meta.get("identifier", meta.get("description", ""))
                            source_desc = (f"Project seed (`{identifier}`)"
                                           if identifier else "Project seed")
                        f.write(f"| `{label}` | `{pid}` | {source_desc} |\n")
                    f.write("\n")
                elif parent_ids:
                    f.write("### Parents\n\n")
                    f.write(f"Parent IDs (not in current corpus): "
                            f"{', '.join(f'`{p}`' for p in parent_ids)}\n\n")

                f.write("*This report is automatically generated by "
                        "[Fusion-Fuzz](https://github.com/0599jiangyc/FusionFuzzLoop)*\n")
        except Exception as e:
            logger.error(f"Failed to write README.md: {e}")

    def _cleanup_stale_processes(self):
        """Kill potential zombie processes from the target project to free
        resources — see core/driver.py's cleanup_stale_processes (shared
        with core/dryrun.py's --dry-run/--pre-analysis pass)."""
        cleanup_stale_processes(self.project_name)


    # Strip ANSI/VT100 escape codes so pattern matching works even when the
    # compiler colorises its output (e.g. cjc wraps "error" in \x1b[31m…\x1b[0m).
    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    def _strip_ansi(self, text: str) -> str:
        return self._ANSI_RE.sub('', text)

    def _is_syntax_error(self, result):
        """Checks stderr/stdout for common syntax error patterns."""
        err = self._strip_ansi(result.stderr or "").lower()
        # Some compilers (e.g. cjc) write diagnostics to stdout, not stderr
        combined = err + "\n" + self._strip_ansi(result.stdout or "").lower()

        # Generic language syntax errors
        if "error:" in err:
            return True
        
        if "syntaxerror" in err or "syntax error" in err or "indentationerror" in err:
            return True

        # WGSL / Naga
        if "could not parse wgsl" in err or "validation error" in err:
            return True

        # Return code 0 with no syntax patterns matched = genuinely valid output
        if result.return_code == 0:
            return False

        # Project-specific error return codes treated as expected/syntax errors
        error_codes = self.config.get("analysis", {}).get("error_return_codes", [])
        if error_codes and result.return_code in error_codes:
            return True

        # rc=1 → normal compile/elaboration failure; rc=139 → SIGSEGV (handled as crash)
        # Neither should be blindly counted as a "syntax error"
        if result.return_code not in (1, 139):
            return True

        return False

    def _fused_valid_rate(self):
        """% of fused samples accepted without a syntax/parse error."""
        if self.sample_count == 0:
            return 100.0
        return 100.0 * (1.0 - (self.syntax_error_count / self.sample_count))

    def _print_status(self):
        """Prints a dynamic status bar to stdout."""
        current_time = time.time()
        # Update UI max 2 times per second
        if current_time - self.last_status_print < 0.5:
            return

        self.last_status_print = current_time
        elapsed = current_time - self.start_time
        if elapsed <= 0: elapsed = 1e-9

        throughput = self.sample_count / elapsed

        valid_rate = self._fused_valid_rate()

        n = len(self.corpus)
        total_pairs = n * (n - 1) // 2
        covered = self.coverage.covered_count()
        cov_pct = (covered / total_pairs * 100.0) if total_pairs > 0 else 100.0
        pairs_per_sec = covered / elapsed
        status = (
            f"\r[ {str(datetime.timedelta(seconds=int(elapsed)))} ] "
            f"Throughput: {throughput:.1f} tests/s | "
            f"Bugs: {len(self.unique_crashes)} | "
            f"FuseValidRate: {valid_rate:.1f}% | "
            f"PairCov: {covered}/{total_pairs} ({cov_pct:.1f}%, {pairs_per_sec:.1f} pairs/s)"
        )

        # Write carriage return to overwrite line
        sys.stdout.write(status)
        sys.stdout.flush()

    def _force_remove(self, action, name, exc):
        """
        Force remove helper for shutil.rmtree.
        Handles read-only files generated by fuzz targets.
        """
        try:
            os.chmod(name, stat.S_IWRITE)
            action(name)
        except Exception as e:
            # logger.warning(f"Force remove failed for {name}: {e}")
            pass

    def _setup_workspace(self):
        """Creates a fresh temporary workspace with dependencies."""
        workspace = tempfile.mkdtemp(prefix=f"ffl_{self.project_name}_")
        # logger.info(f"Initialized temporary workspace: {workspace}")

        try:
            # Symlink 'projects' and 'output' to the workspace
            for item in ["projects", "output"]:
                src = os.path.join(self.original_cwd, item)
                dst = os.path.join(workspace, item)
                if os.path.exists(src):
                    os.symlink(src, dst)

            # Copy project-specific runtime dependencies
            deps_dir_name = f"{self.project_name}_deps" # Generic guess
            if "php" in self.project_name: 
                deps_dir_name = "phpt_deps" # Specific for PHP
            
            deps_src = os.path.join(self.original_cwd, "projects", self.project_name, deps_dir_name)
            
            if os.path.exists(deps_src):
                # logger.info(f"Copying runtime dependencies from {deps_src}...")
                for item in os.listdir(deps_src):
                    s = os.path.join(deps_src, item)
                    d = os.path.join(workspace, item)
                    try:
                        if os.path.isdir(s):
                            shutil.copytree(s, d, symlinks=True, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s, d)
                    except Exception as e:
                        logger.warning(f"Failed to copy dependency {item}: {e}")
            
            return workspace
        except Exception as e:
            logger.error(f"Workspace setup failed: {e}")
            return workspace # Attempt to continue?

    def _log_sample(self, child, result):
        """Append a single sample's execution details to the sample log file."""
        if self._sample_log_file is None:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"\n{'='*60}",
                f"[{ts}] Iteration #{self.iterations}",
                f"{'='*60}",
                f"[SEED ID]     {child.id}",
                f"[RETURN CODE] {result.return_code}",
                f"[CRASHED]     {result.crashed}",
                f"[EXEC TIME]   {result.execution_time:.3f}s",
            ]
            if hasattr(result, 'command') and result.command:
                lines.append(f"[COMMAND]     {result.command}")
            lines += [
                "",
                "--- SEED CONTENT ---",
                child.content or "(empty)",
                "",
                "--- STDOUT ---",
                result.stdout.strip() if result.stdout.strip() else "(empty)",
                "",
                "--- STDERR ---",
                result.stderr.strip() if result.stderr.strip() else "(empty)",
            ]
            self._sample_log_file.write("\n".join(lines) + "\n")
            self._sample_log_file.flush()
        except Exception as e:
            logger.warning(f"Failed to write sample log: {e}")

    def _process_result(self, child, result) -> bool:
        """
        Post-execution handling shared by fuzzing (run) and replay
        (execute_folder): sample logging, syntax/validity stats, crash
        deduplication + bundle saving, corpus feedback, output truncation.

        Returns True when this result was a newly-seen crash.
        """
        # Log sample details
        self._log_sample(child, result)

        # Check Syntax Stats
        self.sample_count += 1
        if self._is_syntax_error(result):
            self.syntax_error_count += 1

        new_crash = False
        if result.crashed:
            # Deduplication logic
            signature = self._extract_crash_signature(result)

            if signature:
                if signature not in self.unique_crashes:
                    self.unique_crashes.add(signature)
                    sys.stdout.write("\n")
                    logger.error(f"CRASH FOUND! ID: {child.id} (Sig: {signature})")
                    self._save_crash_bundle(child, result, signature)
                    new_crash = True
                else:
                    # logger.info(f"Duplicate crash ignored.")
                    pass
            else:
                fallback_sig = f"NoSig_{int(time.time())}"
                self.unique_crashes.add(fallback_sig)
                sys.stdout.write("\n")
                logger.error(f"CRASH FOUND! ID: {child.id} (No Sig - Saving)")
                self._save_crash_bundle(child, result, fallback_sig)
                new_crash = True

        if self.is_interesting(result):
            self.corpus.append(child)

        # Truncate large outputs to save memory, keeping content around
        # the crash signature.
        if len(result.stdout or "") > 50000:
            result.stdout = smart_truncate(result.stdout or "", max_chars=50000)
        if len(result.stderr or "") > 50000:
            result.stderr = smart_truncate(result.stderr or "", max_chars=50000)

        return new_crash

    def execute_folder(self, folder, sample_log=None):
        """
        Replay mode (--execute): run every program in `folder` through the
        project driver and report bugs. No fusion, no generation, no
        corpus — just execution plus the same crash triage, deduplication
        (against output/bugs/<project>/ from previous runs too) and bug
        bundle saving that fuzzing does.

        Programs are read recursively; manifest.jsonl and hidden files are
        skipped. Each program keeps its own file extension, so a folder
        written by --save-to can be replayed as-is.
        """
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            logger.error(f"--execute: {folder} is not a directory")
            return 0

        skip_names = {"manifest.jsonl"}
        files = []
        for root, dirs, names in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(names):
                if name.startswith(".") or name in skip_names:
                    continue
                files.append(os.path.join(root, name))

        if not files:
            logger.error(f"--execute: no programs found in {folder}")
            return 0

        programs = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
                continue
            if not content.strip():
                continue
            rel = os.path.relpath(path, folder)
            programs.append(Seed(content=content, metadata={
                "type": "replay",
                "filename": rel,
                "source_path": path,
                "extension": os.path.splitext(path)[1] or None,
            }))

        if not programs:
            logger.error(f"--execute: no readable, non-empty programs in {folder}")
            return 0

        logger.info(f"Replay mode: executing {len(programs)} program(s) from {folder}")
        self.start_time = time.time()
        self.driver.prepare_environment()

        # Open sample log file
        if sample_log:
            self.sample_log_path = sample_log
            try:
                os.makedirs(os.path.dirname(os.path.abspath(sample_log)), exist_ok=True)
                self._sample_log_file = open(sample_log, "a", encoding="utf-8")
                start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._sample_log_file.write(
                    f"\n{'#'*60}\n# FFL Replay Log | Project: {self.project_name} | "
                    f"Started: {start_ts}\n{'#'*60}\n")
                self._sample_log_file.flush()
                logger.info(f"Sample log: {sample_log}")
            except Exception as e:
                logger.warning(f"Could not open sample log file {sample_log}: {e}")

        self.original_cwd = os.getcwd()
        self.current_workspace = self._setup_workspace()
        os.chdir(self.current_workspace)

        max_workers = self.config.get("execution", {}).get("concurrency", 4)
        logger.info(f"ThreadPoolExecutor initialized with {max_workers} workers.")

        crashes_before = len(self.unique_crashes)
        executed = 0
        executor = ThreadPoolExecutor(max_workers=max_workers)

        def _exec(seed):
            try:
                return seed, self.driver.execute(seed)
            except Exception as e:
                logger.error(f"Execution driver error on {seed.metadata.get('filename')}: {e}")
                return seed, None

        try:
            pending = iter(programs)
            active = set()
            while True:
                while len(active) < max_workers:
                    nxt = next(pending, None)
                    if nxt is None:
                        break
                    active.add(executor.submit(_exec, nxt))
                if not active:
                    break

                done, active = wait(active, return_when=FIRST_COMPLETED, timeout=30)
                for future in done:
                    try:
                        seed, result = future.result()
                    except Exception as e:
                        logger.error(f"Error extracting result: {e}")
                        continue
                    if result is None:
                        continue
                    executed += 1
                    self._process_result(seed, result)

                    if executed % 10 == 0 or executed == len(programs):
                        elapsed = time.time() - self.start_time
                        rate = executed / elapsed if elapsed > 0 else 0.0
                        sys.stdout.write(
                            f"\r[ {int(elapsed)//60:d}:{int(elapsed)%60:02d} ] "
                            f"Executed: {executed}/{len(programs)} | "
                            f"{rate:.1f} tests/s | "
                            f"Bugs: {len(self.unique_crashes)} | "
                            f"ValidRate: {self._fused_valid_rate():.1f}%"
                        )
                        sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            logger.info(f"Replay interrupted by user after {executed} program(s).")
        finally:
            executor.shutdown(wait=False)
            if self._sample_log_file:
                try:
                    self._sample_log_file.close()
                except Exception:
                    pass
            self._cleanup_stale_processes()
            os.chdir(self.original_cwd)
            try:
                shutil.rmtree(self.current_workspace, onerror=self._force_remove)
            except Exception:
                pass

        sys.stdout.write("\n")
        new_crashes = len(self.unique_crashes) - crashes_before
        logger.info(
            f"Replay finished: {executed}/{len(programs)} program(s) executed, "
            f"valid rate {self._fused_valid_rate():.1f}%, "
            f"{new_crashes} new unique bug(s) "
            f"(saved under output/bugs/{self.project_name}/)."
        )
        return new_crashes

    def run(self, max_iterations, sample_log=None):
        logger.info(f"Starting FFL for {self.project_name} with parallel execution...")
        self.start_time = time.time() # Reset start time
        self.driver.prepare_environment()

        # Open sample log file
        if sample_log:
            self.sample_log_path = sample_log
            try:
                os.makedirs(os.path.dirname(os.path.abspath(sample_log)), exist_ok=True)
                self._sample_log_file = open(sample_log, "a", encoding="utf-8")
                start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._sample_log_file.write(f"\n{'#'*60}\n# FFL Sample Log | Project: {self.project_name} | Started: {start_ts}\n{'#'*60}\n")
                self._sample_log_file.flush()
                logger.info(f"Sample log: {sample_log}")
            except Exception as e:
                logger.warning(f"Could not open sample log file {sample_log}: {e}")

        # 1. Create Initial Temporary Workspace
        self.original_cwd = os.getcwd()
        self.current_workspace = self._setup_workspace()
        os.chdir(self.current_workspace)

        # 2. Setup ThreadPool
        max_workers = self.config.get("execution", {}).get("concurrency", 4)
        logger.info(f"ThreadPoolExecutor initialized with {max_workers} workers.")
        
        if max_iterations == -1:
            logger.info("Mode: Continuous Fuzzing (Press Ctrl-C to stop)")

        submitted_count = 0
        active_futures = set()
        rotate_pending = False
        
        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Helper to check if we should continue submitting tasks
            def should_submit():
                if self.coverage.is_saturated(self.corpus):
                    return False
                if max_iterations == -1: return True
                return submitted_count < max_iterations

            # Loop until finished
            while active_futures or should_submit():
                
                # REPLENISH PHASE
                # Submit tasks only if not rotating
                while len(active_futures) < max_workers and should_submit() and not rotate_pending:
                    f = executor.submit(self.process_iteration)
                    active_futures.add(f)
                    submitted_count += 1
                
                # Check if we should rotate but queue is empty (ready to rotate)
                if rotate_pending and not active_futures:
                    sys.stdout.write("\n")
                    logger.info("Maintenance: Rotating workspace to clean disk...")
                    
                    # 0. Clean up any lingering processes first
                    self._cleanup_stale_processes()
                    # Short sleep to let OS release file handles
                    time.sleep(0.2)

                    # 1. Restore CWD
                    os.chdir(self.original_cwd)
                    
                    # 2. Delete old workspace with force handler
                    try:
                        shutil.rmtree(self.current_workspace, onerror=self._force_remove)
                    except Exception as e:
                        logger.warning(f"Cleanup warning: {e}")
                        
                    # 3. Create new
                    gc.collect() # Force Python GC
                    self.current_workspace = self._setup_workspace()
                    os.chdir(self.current_workspace)
                    
                    rotate_pending = False
                    continue # Loop back to replenish

                # WAIT PHASE
                if active_futures:
                    done, not_done = wait(active_futures, return_when=FIRST_COMPLETED, timeout=10)
                    
                    if done:
                        # Reset activity timer
                        last_activity_time = time.time()
                        
                        for future in done:
                            # PROCESS RESULTS
                            try:
                                self.iterations += 1

                                # Check for rotation triggers
                                if self.iterations % 2000 == 0:
                                    rotate_pending = True

                                # Periodic Process Cleanup
                                if self.iterations % 2000 == 0:
                                    self._cleanup_stale_processes()

                                # Periodic GC
                                if self.iterations % 2000 == 0:
                                    gc.collect()

                                pairs = future.result()

                                for child, result in pairs:
                                    if not child or not result:
                                        continue
                                    self._process_result(child, result)

                            except Exception as e:
                                logger.error(f"Error extracting future result: {e}")
                            
                            # Explicitly delete future to free memory
                            del future

                        self._print_status()
                        active_futures = not_done
                        del done
                        
                    else:
                        # Timeout/Stall Detection
                        if not rotate_pending: # Don't warn if we are intentionally draining
                            # We can implement stall restart here if needed, but wait() usually returns
                            pass
                elif not should_submit():
                    # No active futures and shouldn't submit -> Done
                    if self.coverage.is_saturated(self.corpus):
                        sys.stdout.write("\n")
                        logger.info(
                            f"All {self.coverage.covered_count()} pairs covered — stopping fuzzing."
                        )
                    break

        except KeyboardInterrupt:
            sys.stdout.write("\n")
            logger.info(f"Fuzzing interrupted by user. Stopping after {self.iterations} iterations.")
        finally:
            logger.info("Shutting down executor...")
            executor.shutdown(wait=False)
            if self._sample_log_file:
                try:
                    self._sample_log_file.close()
                except Exception:
                    pass
            if self.sample_count > 0:
                logger.info(f"Fused valid rate: {self._fused_valid_rate():.1f}%")

        # 3. Cleanup Workspace
        os.chdir(self.original_cwd)
        try:
            shutil.rmtree(self.current_workspace, onerror=self._force_remove)
            logger.info("Final workspace cleaned up.")
        except Exception as e:
            pass

    def is_interesting(self, result):
        if result.crashed: return False
        # if result.execution_time > 1.0: return True 
        return False
