"""
Reducer: minimizes a crash-inducing reproducer using delta debugging.

Usage (via main.py):
    python3 main.py --reduce ./output/bugs/php/Assertion__0_2f95bf0e

Reads test.<ext> (or legacy reproduce.<ext>), runs DeltaDebugger line-by-line,
writes min.<ext>, and updates README.md (or report.md) with minimization stats.
"""

import os
import re
import sys
import signal
import subprocess
import time
import logging
from datetime import datetime, timezone

from core.driver import ExecutionResult

logger = logging.getLogger("FFL.Reducer")


# ---------------------------------------------------------------------------
# Shell-based test driver — replays the exact command from test.sh
# ---------------------------------------------------------------------------

class _ShellTestDriver:
    """
    Minimal driver-compatible wrapper that runs the command extracted from
    test.sh.  Used by the minimizer so it exercises the exact same binary and
    flags that originally triggered the crash, bypassing the project driver.
    """

    _CRASH_KEYWORDS = (
        "SUMMARY:",
        "Assertion failed",
        ": Assertion `",
        "INTERNAL PANIC:",
        "Segmentation fault",
        "SIGABRT",
        "core dumped",
        "AddressSanitizer",
        "UndefinedBehaviorSanitizer",
        "Bus error",
        "Fatal Python error:",
        "LLVM ERROR:",
        "internal compiler error:",
    )

    def __init__(self, bug_dir: str, test_sh_path: str, test_fname: str, timeout: int = 30):
        self.bug_dir = bug_dir
        self.test_fname = test_fname        # e.g. "test.php"
        self.ext = os.path.splitext(test_fname)[1]   # e.g. ".php"
        self.timeout = timeout
        self._cmd_template = self._parse_test_sh(test_sh_path)
        if not self._cmd_template:
            raise ValueError(f"Could not extract execution command from {test_sh_path}")
        print(f"  [ShellTestDriver] command template: {self._cmd_template[:120]}")

    def _parse_test_sh(self, path: str) -> str:
        """Return the last substantive line of test.sh (the actual binary invocation)."""
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        exec_lines = [
            l for l in lines
            if l.strip()
            and not l.strip().startswith("#")
            and "SCRIPT_DIR=" not in l
            and l.strip() != "#!/bin/bash"
        ]
        return exec_lines[-1].strip() if exec_lines else ""

    def _build_cmd(self, seed_file: str) -> str:
        """Replace all $SCRIPT_DIR/test.EXT references with the actual seed file path."""
        cmd = self._cmd_template
        for variant in (
            f'"$SCRIPT_DIR/{self.test_fname}"',
            f"$SCRIPT_DIR/{self.test_fname}",
            f'"$SCRIPT_DIR/test{self.ext}"',
            f"$SCRIPT_DIR/test{self.ext}",
        ):
            cmd = cmd.replace(variant, f'"{seed_file}"')
        cmd = cmd.replace("$SCRIPT_DIR", self.bug_dir)
        return cmd

    def execute(self, seed):
        tmp = os.path.join(self.bug_dir, f"_min_tmp_{seed.id}{self.ext}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(seed.content)

            cmd = self._build_cmd(tmp)
            start = time.time()
            try:
                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    raw_out, raw_err = proc.communicate(timeout=self.timeout)
                    rc = proc.returncode
                    stdout = raw_out.decode("utf-8", errors="replace")
                    stderr = raw_err.decode("utf-8", errors="replace")
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait()
                    rc, stdout, stderr = 124, "", "TIMEOUT"
            except Exception as e:
                rc, stdout, stderr = 1, "", str(e)
            duration = time.time() - start

            combined = stdout + stderr
            crashed = any(kw in combined for kw in self._CRASH_KEYWORDS)
            sig = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
            res = ExecutionResult(rc, stdout, stderr, duration, crashed, sig)
            res.command = cmd
            res.seed_file = tmp
            return res
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def extract_crash_signature(self, stdout, stderr, return_code):
        for text in (stderr, stdout):
            m = re.search(r"(SUMMARY: .*)", text)
            if m:
                return m.group(1).strip()
        for text in (stderr, stdout):
            m = re.search(r"(Assertion[^:]*:.*)", text)
            if m:
                return m.group(1).strip()
        for text in (stderr, stdout):
            m = re.search(r"SUMMARY: (\S+Sanitizer):\s+(.*)", text)
            if m:
                return f"SUMMARY: {m.group(1)}: {m.group(2).strip()}"
        return None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_test_file(bug_dir: str) -> tuple[str | None, str | None]:
    """Return (filename, path) for the primary reproducer (test.* or reproduce.*).

    Extension priority: .php is preferred over .phpt so that --reduce operates
    on the plain PHP file rather than the full test-harness wrapper.
    """
    _EXT_PRIORITY = {".php": 0, ".phpt": 1}

    def _priority(fname):
        ext = os.path.splitext(fname)[1].lower()
        return _EXT_PRIORITY.get(ext, 2)

    for prefix in ("test", "reproduce"):
        candidates = [
            f for f in os.listdir(bug_dir)
            if f.startswith(prefix) and not f.endswith((".sh", ".out"))
        ]
        candidates.sort(key=_priority)
        if candidates:
            fname = candidates[0]
            return fname, os.path.join(bug_dir, fname)
    return None, None


def _read_signature(bug_dir: str) -> str | None:
    """Read the crash signature from README.md or report.md."""
    for name in ("README.md", "report.md"):
        path = os.path.join(bug_dir, name)
        if os.path.exists(path):
            text = open(path, encoding="utf-8", errors="replace").read()
            m = re.search(r"\*\*Signature:\*\*\s*`([^`]+)`", text)
            if m:
                return m.group(1)
    return None


def _update_report(bug_dir: str, min_fname: str, original_lines: int, minimized_lines: int) -> None:
    """Replace the reproducer code block in README.md with the minimized content."""
    report_path = None
    for name in ("README.md", "report.md"):
        candidate = os.path.join(bug_dir, name)
        if os.path.exists(candidate):
            report_path = candidate
            break
    if not report_path:
        logger.warning("No README.md / report.md found — skipping report update.")
        return

    min_path = os.path.join(bug_dir, min_fname)
    if not os.path.exists(min_path):
        logger.warning(f"Minimized file {min_path} not found — skipping report update.")
        return

    minimized_content = open(min_path, encoding="utf-8", errors="replace").read()
    text = open(report_path, encoding="utf-8", errors="replace").read()

    reduction_pct = (1.0 - minimized_lines / original_lines) * 100 if original_lines > 0 else 0.0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Replace the code block after "The following code:" with the minimized content.
    # The block looks like:  The following code:\n\n```lang\n<code>\n```\n
    def _replace_code_block(m):
        opening = m.group(1)   # "The following code:\n\n```lang\n"
        closing = m.group(3)   # "```"
        note = f"*(minimized {original_lines}→{minimized_lines} lines, {reduction_pct:.1f}% reduction, {now})*\n\n"
        return opening + minimized_content.rstrip('\n') + "\n" + closing + "\n\n" + note

    new_text, count = re.subn(
        r'(The following code:\n\n```[^\n]*\n)(.*?)(```)',
        _replace_code_block,
        text,
        count=1,
        flags=re.DOTALL,
    )

    if count == 0:
        logger.warning("Could not find code block in report — report not updated.")
        return

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    logger.info(f"Updated {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def reduce_command(bug_dir: str, override_sig: str | None = None) -> None:
    """Minimize the crash reproducer in *bug_dir* using delta debugging."""
    from core.config_loader import load_project_config
    from core.driver import get_driver
    from core.minimizer import DeltaDebugger

    bug_dir = os.path.normpath(bug_dir)
    if not os.path.isdir(bug_dir):
        print(f"Error: Not a directory: {bug_dir}", file=sys.stderr)
        sys.exit(1)

    # Infer project from path: …/output/bugs/<project>/<dir_name>
    parts = bug_dir.replace("\\", "/").split("/")
    project = None
    for i, part in enumerate(parts):
        if part == "bugs" and i + 1 < len(parts):
            project = parts[i + 1]
            break
    if not project:
        print(
            "Error: Cannot infer project from path. Expected .../output/bugs/<project>/...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Project:    {project}")
    print(f"Bug dir:    {bug_dir}")

    try:
        config = load_project_config(project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    test_fname, test_path = _find_test_file(bug_dir)
    if not test_fname:
        print("Error: No test.* or reproduce.* file found in bug directory.", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(test_fname)[1]
    min_fname = f"min{ext}"
    min_path = os.path.join(bug_dir, min_fname)

    content = open(test_path, encoding="utf-8", errors="replace").read()
    original_lines = len(content.splitlines())

    sig = override_sig if override_sig is not None else _read_signature(bug_dir)

    print(f"Reproducer: {test_fname}  ({original_lines} lines)")
    print(f"Output:     {min_fname}")
    print(f"Signature:  {sig or '(none — any crash counts)'}"
          + ("  [override]" if override_sig is not None else ""))
    print()

    # Prefer test.sh for reproduction — it uses the exact command that found the bug.
    test_sh = os.path.join(bug_dir, "test.sh")
    timeout = config.get("execution", {}).get("timeout", 30)
    if os.path.exists(test_sh):
        print("Using test.sh command for reproduction (bypasses project driver).")
        try:
            driver = _ShellTestDriver(bug_dir, test_sh, test_fname, timeout=timeout)
        except Exception as e:
            print(f"Warning: could not parse test.sh ({e}), falling back to project driver.")
            driver = get_driver(config)
    else:
        driver = get_driver(config)

    debugger = DeltaDebugger(driver)

    # Pre-flight: confirm the original reproducer still triggers the crash
    print("Verifying bug reproduces...")
    if not debugger._test(content.splitlines(keepends=True), sig):
        print(
            f"Error: original reproducer does NOT reproduce the crash"
            + (f" with signature '{sig}'." if sig else " (no crash detected)."),
            file=sys.stderr,
        )
        sys.exit(1)
    print("Confirmed. Starting delta debugging...")
    print()

    minimized = debugger.minimize(content, expected_sig=sig)
    minimized_lines = len(minimized.splitlines())

    with open(min_path, "w", encoding="utf-8") as f:
        f.write(minimized)

    reduction_pct = (1.0 - minimized_lines / original_lines) * 100 if original_lines > 0 else 0.0
    print()
    print(f"Done.  {original_lines} → {minimized_lines} lines  ({reduction_pct:.1f}% reduction)")
    print(f"Written: {min_path}")

    _update_report(bug_dir, min_fname, original_lines, minimized_lines)
    print("Report updated.")
