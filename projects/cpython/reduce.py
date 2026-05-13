#!/usr/bin/env python3
"""
reduce.py — delta-debug a crashing Python test into a minimal PoC, preserving cwd/__file__.

Why your earlier run timed out
------------------------------
Many test scripts use relative paths or Path(__file__).parent. My previous tool
executed candidates from /tmp, breaking those assumptions and causing hangs/timeouts.
This version:
  • Verifies the actual seed file (no temp file).
  • Executes all candidates from the seed directory, writing them as <as-name>
    (default: _min_cand.py) so __file__ and sibling lookups keep working.

Quick start (your case)
-----------------------
python3 reduce.py \
  --python ./cpython/python \
  --seed ./bugs/1/test.py \
  --out ./bugs/1/min.py \
  --timeout 30 \
  --pattern "AddressSanitizer|SUMMARY:" \
  --retries 1

If your test needs specific ASan/Python env, add:
  --env ASAN_OPTIONS=detect_leaks=0:allocator_may_return_null=1:symbolize=1
Deterministic runs (optional): --env PYTHONHASHSEED=0 --env PYTHONDONTWRITEBYTECODE=1

Extras you may need:
  --stdin-file input.txt          # if the test reads from stdin()
  --interpreter-args "-X dev"     # flags for CPython
  --script-args "--flag foo"      # flags for your script
  --cwd /path/to/dir              # override working directory (defaults to seed dir)
  --as-name test.py               # pretend the candidate *is* test.py (rarely needed)

Retry semantics
---------------
• This build treats retries as "ANY run matching counts as a hit" (useful for flaky crashes).

License: MIT
"""

from __future__ import annotations
import argparse, ast, hashlib, os, re, shlex, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Delta-debug a crashing CPython script while preserving cwd and __file__."
    )
    p.add_argument("--python", required=True, help="Path to CPython binary (e.g., ./cpython/python).")
    p.add_argument("--seed", required=True, help="Path to the initial crashing test script.")
    p.add_argument("--out", default="minimized.py", help="Output path for the minimal PoC.")
    p.add_argument("--work", default=None,
                   help="Work dir for logs/steps; default: <seed_dir>/.minwork")
    p.add_argument("--timeout", type=int, default=10, help="Per-run timeout (seconds).")
    p.add_argument("--retries", type=int, default=1,
                   help="Reruns to confirm the oracle (ANY run matching counts as a hit).")
    p.add_argument("--pattern", default=r"AddressSanitizer|heap-use-after-free|SUMMARY:",
                   help="Regex that must appear in stdout/stderr to count as a crash.")
    p.add_argument("--env", action="append", default=[],
                   help="Extra env KEY=VALUE (repeatable).")
    p.add_argument("--stdin-file", default=None,
                   help="Path to a file whose contents are piped to stdin for each run.")
    p.add_argument("--interpreter-args", default="",
                   help="Args for the Python interpreter (e.g., '-X dev -W error').")
    p.add_argument("--script-args", default="",
                   help="Args passed to the script (e.g., '--flag foo').")
    p.add_argument("--cwd", default=None,
                   help="Working directory for running candidates. Default: seed's directory.")
    p.add_argument("--as-name", default="_min_cand.py",
                   help="Filename to use for candidate in cwd (keeps __file__ stable).")
    p.add_argument("--quiet", action="store_true", help="Less verbose output.")
    return p.parse_args()

# ------------------------- Utils -------------------------

def now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def split_keepends(s: str) -> List[str]:
    return s.splitlines(keepends=True)

def join_lines(lines: List[str]) -> str:
    return "".join(lines)

def sha1(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8", errors="replace")).hexdigest()

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def strip_ws(lines: List[str]) -> List[str]:
    import re as _re
    return [_re.sub(r"[ \t]+(?=\r?\n$)", "", ln) for ln in lines]

def strip_blank_and_comment_only(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if s.startswith("#"): continue
        out.append(ln)
    return out

def parse_env(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for kv in pairs:
        if "=" not in kv:
            raise SystemExit(f"--env expects KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        out[k] = v
    return out

# ------------------------- Runner / Oracle -------------------------

class Runner:
    def __init__(self, pybin: str, cwd: Path, pat: str, timeout: int,
                 extra_env: Dict[str, str], interp_argv: List[str], script_argv: List[str],
                 stdin_data: Optional[str], quiet: bool):
        self.pybin = pybin
        self.cwd = cwd
        self.pat = re.compile(pat, re.IGNORECASE | re.MULTILINE)
        self.timeout = timeout
        self.env = os.environ.copy()
        self.env.update(extra_env)
        # make runs deterministic & avoid pyc litter
        self.env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        self.interp_argv = interp_argv
        self.script_argv = script_argv
        self.stdin_data = stdin_data
        self.quiet = quiet

    def _run_path(self, path: Path) -> Tuple[str, str, int, bool]:
        """Run an existing file at `path`. Returns (stdout, stderr, rc, timed_out)."""
        argv = [self.pybin] + self.interp_argv + [str(path)] + self.script_argv
        try:
            cp = subprocess.run(
                argv,
                cwd=str(self.cwd),
                env=self.env,
                input=self.stdin_data,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
            return cp.stdout or "", cp.stderr or "", cp.returncode, False
        except subprocess.TimeoutExpired as te:
            so = te.stdout or ""
            if isinstance(te.stderr, (bytes, bytearray, memoryview)):
                te.stderr = bytes(te.stderr).decode("utf-8", errors="replace")
            se = (te.stderr or "") + ("TIMEOUT" if "TIMEOUT" not in (te.stderr or "") else "")
            return so, se, -9, True

    def verify_seed_file(self, seed_path: Path, retries: int) -> Tuple[bool, str, str]:
        """Verify the real seed file is interesting. ANY run matching the pattern counts as a hit."""
        out, err = "", ""
        for _ in range(max(1, retries)):
            out, err, _rc, _to = self._run_path(seed_path)
            if self.pat.search((out or "") + "\n" + (err or "")):
                return True, out, err
        return False, out, err

    def _to_text(self, x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        # bytes/bytearray -> str
        try:
            return x.decode("utf-8", "replace")
        except Exception:
            return str(x)

    def run_candidate_code(self, code: str, candidate_filename: str, retries: int) -> Tuple[bool, str, str, int]:
        """
        Write `code` to cwd/candidate_filename, run it N times, delete it.
        Returns (interesting, last_stdout, last_stderr, last_rc).
        """
        cpath = self.cwd / candidate_filename
        last_out = last_err = ""
        last_rc = 0
        interesting_any = False

        # Write candidate
        write_text(cpath, code)
        try:
            for _ in range(max(1, retries)):
                out, err, rc, _to = self._run_path(cpath)
                last_out, last_err, last_rc = out, err, rc

                haystack = self._to_text(out) + "\n" + self._to_text(err)
                if self.pat.search(haystack):
                    interesting_any = True
                    break
        finally:
            try:
                cpath.unlink()
            except OSError:
                pass

        return interesting_any, last_out, last_err, last_rc

# ------------------------- Caching -------------------------

class OracleCache:
    def __init__(self):
        self.yes: Set[str] = set()
        self.no: Set[str] = set()
    def has(self, d: str) -> Optional[bool]:
        if d in self.yes: return True
        if d in self.no: return False
        return None
    def put(self, d: str, val: bool) -> None:
        (self.yes if val else self.no).add(d)

# ------------------------- ddmin + greedy -------------------------

def ddmin(lines: List[str], is_interesting, record_step) -> List[str]:
    n = 2
    if not lines: return lines
    while True:
        if n > len(lines): break
        chunk = max(1, len(lines) // n)
        changed = False
        i = 0
        while i < len(lines):
            cand = lines[:i] + lines[i+chunk:]
            if is_interesting(cand):
                record_step(cand, f"ddmin: removed lines[{i}:{i+chunk}) @ n={n}")
                lines = cand
                n = max(2, n - 1)
                changed = True
                i = 0
                continue
            i += chunk
        if not changed:
            if n >= len(lines): break
            n = min(len(lines), n * 2)
    return lines

def greedy_single_line(lines: List[str], is_interesting, record_step) -> List[str]:
    progress = True
    while progress:
        progress = False
        i = 0
        while i < len(lines):
            cand = lines[:i] + lines[i+1:]
            if is_interesting(cand):
                record_step(cand, f"greedy: removed line {i}")
                lines = cand
                progress = True
                i = 0
                continue
            i += 1
    return lines

# ------------------------- Driver -------------------------

def minimize(seed_code: str, runner: Runner, retries: int, workdir: Path,
             candidate_filename: str, quiet: bool) -> str:
    cache = OracleCache()

    def test_candidate(lines: List[str]) -> bool:
        code = join_lines(lines)
        d = sha1(code)
        memo = cache.has(d)
        if memo is not None:
            return memo
        if not is_valid_python(code):
            cache.put(d, False)
            return False
        ok, _o, _e, _rc = runner.run_candidate_code(code, candidate_filename, retries)
        cache.put(d, ok)
        return ok

    step = [0]
    def record_step(lines: List[str], msg: str):
        step[0] += 1
        code = join_lines(lines)
        write_text(workdir / f"step_{step[0]:04d}.py", code)
        if not quiet:
            print(f"[{now()}] {msg} -> {workdir}/step_{step[0]:04d}.py ({len(lines)} lines)")

    # Light cleanup
    L0 = split_keepends(seed_code)
    L1 = strip_ws(L0)
    L2 = strip_blank_and_comment_only(L1)
    if L2 != L1 and test_candidate(L2):
        record_step(L2, "pre: stripped blank/comment-only lines")
        L1 = L2

    # ddmin + greedy
    Lr = ddmin(L1, test_candidate, record_step)
    Lr = greedy_single_line(Lr, test_candidate, record_step)
    return join_lines(Lr)

def main():
    a = parse_args()

    pybin = shutil.which(a.python) if os.path.sep not in a.python else a.python
    if not pybin or not Path(pybin).exists():
        sys.exit(f"Cannot find CPython binary: {a.python}")

    seed_path = Path(a.seed).resolve()
    if not seed_path.exists():
        sys.exit(f"Seed not found: {seed_path}")

    seed_dir = Path(a.cwd).resolve() if a.cwd else seed_path.parent
    workdir = Path(a.work).resolve() if a.work else (seed_dir / ".minwork")
    workdir.mkdir(parents=True, exist_ok=True)

    seed_code = read_text(seed_path)
    if not seed_code.strip():
        sys.exit("Seed is empty.")

    extra_env = parse_env(a.env)
    interp_argv = shlex.split(a.interpreter_args) if a.interpreter_args else []
    script_argv = shlex.split(a.script_args) if a.script_args else []
    stdin_data = None
    if a.stdin_file:
        stdin_data = read_text(Path(a.stdin_file))

    runner = Runner(
        pybin=pybin,
        cwd=seed_dir,
        pat=a.pattern,
        timeout=a.timeout,
        extra_env=extra_env,
        interp_argv=interp_argv,
        script_argv=script_argv,
        stdin_data=stdin_data,
        quiet=a.quiet,
    )

    # 1) Verify the *actual* seed file (not a temp copy)
    if not a.quiet:
        print(f"[{now()}] Verifying seed file reproduces in cwd={seed_dir} ...")
    ok, so, se = runner.verify_seed_file(seed_path, retries=max(1, a.retries))
    if not ok:
        sys.stderr.write("Seed did not match the crash oracle. Here is the output:\n")
        sys.stderr.write("---- stdout ----\n" + (so or "") + "\n")
        sys.stderr.write("---- stderr ----\n" + (se or "") + "\n")
        sys.stderr.write(
            "\nHints:\n"
            " • If the script needs stdin, provide --stdin-file.\n"
            " • If it needs specific env (ASAN_OPTIONS, PYTHONHASHSEED, etc.), pass --env KEY=VAL.\n"
            " • If it only crashes with certain interpreter flags, use --interpreter-args.\n"
            " • If it requires script args, use --script-args.\n"
            " • If it must keep the original filename, try --as-name test.py.\n"
        )
        sys.exit(2)

    # 2) Start minimization
    if not a.quiet:
        print(f"[{now()}] Seed verified. Starting minimization ...")
        write_text(workdir / "seed.py", seed_code)

    minimized = minimize(
        seed_code=seed_code,
        runner=runner,
        retries=max(1, a.retries),
        workdir=workdir,
        candidate_filename=a.as_name,
        quiet=a.quiet,
    )

    # 3) Write final PoC
    out_path = Path(a.out).resolve()
    write_text(out_path, minimized)

    # 4) Re-validate final PoC (as a file at out_path, preserving cwd)
    if not a.quiet:
        print(f"[{now()}] Minimization complete. Wrote minimal PoC to: {out_path}")
        print(f"[{now()}] Re-validating final PoC ...")
    # Run the OUT file itself so __file__ points to the final location
    out_runner = Runner(
        pybin=pybin,
        cwd=out_path.parent,
        pat=a.pattern,
        timeout=a.timeout,
        extra_env=extra_env,
        interp_argv=interp_argv,
        script_argv=script_argv,
        stdin_data=stdin_data,
        quiet=a.quiet,
    )
    ok2, so2, se2 = out_runner.verify_seed_file(out_path, retries=max(1, a.retries))
    write_text(workdir / "final_stdout.txt", so2 or "")
    write_text(workdir / "final_stderr.txt", se2 or "")
    if not ok2:
        sys.stderr.write("WARNING: Final PoC did not reproduce on re-run. Consider --retries 2..3.\n")

if __name__ == "__main__":
    main()