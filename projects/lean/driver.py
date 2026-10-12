"""
projects/lean/driver.py — elaborate (and sometimes run, or compile and
run) a fused Lean file and say whether what came back is a bug.

Toolchains (setup.py records them in projects/lean/toolchain.json): the
from-source `relwithassert` stage1 (LEAN_DEBUG assertions, -O3) for most
runs, and the `asan` stage1 (the same plus ASan/UBSan) for ASAN_SHARE of
them when it was built.

What varies per run:

  tool     elab  `lean file.lean` — parse, elaborate, kernel-check and
                 compile every declaration (the compiler runs on every
                 `def`, not only on `#eval`), most of the weight;
           run   `lean --run file.lean` on a program with `main`: the IR
                 interpreter executes it after elaboration;
           diff  `lean -c` + `leanc` + the native binary, against
                 `lean --run` of the same file: a miscompile shows up as
                 different output or exit status. Only for programs whose
                 output is deterministic.
  outputs  `-o x.olean` (module serialisation), `-i x.ilean`, `-c x.c`
           (C emission), `--json`, `--stats`, `--profile`.
  checks   the frontend's own debug checks: `compiler.check` (LCNF type
           checking after every compiler pass), `grind.debug`,
           `grind.debug.proofs`, `sym.debug`, `debug.simp.check.have`,
           `debug.synthInstance.checkCacheHits`.
  modes    threads (`-j`), `Elab.async`, `compiler.postponeCompile`,
           `compiler.reuse`/`extract_closed`, one `backward.*` switch
           flipped to its legacy algorithm (the legacy `do` elaborator,
           the old match compiler layout, ...), a low `maxHeartbeats`
           (cancellation and error recovery), and printing options that
           route every message through different delaborator paths
           (`pp.all`, `pp.explicit`, `pp.analyze`, ...).

stdin is closed: after a C++ `lean_assert` fails, Lean asks a debugger
prompt on stdin, and on EOF it aborts — otherwise it would wait out the
timeout.
"""

import json
import os
import random
import re
import shlex
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ffl_lean_analyzer", os.path.join(_here, "analyzer.py"))
_an = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_an)
classify = _an.classify

_MAIN_RE = re.compile(r'^(?:@\[[^\]]*\]\s*)?(?:(?:private|public|unsafe|partial)\s+)*def\s+main\b', re.M)
_META_RE = re.compile(
    r'\brun_meta\b|\brun_cmd\b|\brun_elab\b|^elab\b|^elab_rules\b|\bMetaM\b|\bTermElabM\b|'
    r'\bCommandElabM\b|\bTacticM\b|\bCoreM\b|\baddDecl\b|\bmkApp|\bLean\.Expr\b|\bExpr\.', re.M)
_UNSAFE_RE = re.compile(r'\bunsafe\b|unsafeCast|ptrAddrUnsafe|unsafeBaseIO|unsafeIO|'
                        r'implemented_by|@\[extern|withPtr|unsafeEIO')
# Output that is not a function of the program text.
_NONDET_RE = re.compile(
    r'monoMsNow|monoNanosNow|IO\.rand|stdGenRef|randNat|getRandomBytes|IO\.Process|'
    r'getEnv|IO\.FS|FilePath|readFile|writeFile|IO\.sleep|Task\.|IO\.asTask|BaseIO\.asTask|'
    r'Thread|ptrAddr|getNumHeartbeats|IO\.getStdin|getLine|IO\.appPath|currentDir|'
    r'Std\.Time|IO\.getTID|\bdbgTrace|panic!|unreachable!|assert!|IO\.Ref|ST\.Ref', re.M)


class LeanDriver(BaseDriver):
    ASAN_SHARE = 0.3
    RUN_SHARE = 0.35          # of programs that define `main`
    DIFF_SHARE = 0.10         # of deterministic programs that define `main`

    # `--memory` is Lean's own heap limit (MB): a runaway `decide` or
    # `#reduce` fails with an out-of-memory error instead of swapping.
    BASE = ["-DElab.inServer=true", "--memory=6144"]
    LINTER = [(["-Dlinter.all=false"], 70), ([], 20), (["-Dlinter.all=true"], 10)]
    OUTPUTS = [("-o", "out.olean", 0.25), ("-i", "out.ilean", 0.10), ("-c", "out.c", 0.25)]
    REPORT = [("--json", 0.12), ("--stats", 0.04), ("--profile", 0.05)]
    CHECKS = [
        ("-Dcompiler.check=true", 0.5),
        ("-Dgrind.debug=true", 0.25),
        ("-Dgrind.debug.proofs=true", 0.2),
        ("-Dsym.debug=true", 0.2),
        ("-Ddebug.simp.check.have=true", 0.2),
        ("-Ddebug.synthInstance.checkCacheHits=true", 0.15),
    ]
    THREADS = [([], 40), (["-j1"], 25), (["-j2"], 15), (["-j4"], 20)]
    ASYNC = [([], 50), (["-DElab.async=true"], 30), (["-DElab.async=false"], 20)]
    COMPILER = [
        ("-Dcompiler.postponeCompile=false", 0.6),
        ("-Dcompiler.reuse=false", 0.1),
        ("-Dcompiler.extract_closed=false", 0.1),
    ]
    # One legacy-algorithm switch (each flips its option's default).
    BACKWARD = [
        "-Dbackward.do.legacy=true", "-Dbackward.match.sparseCases=false",
        "-Dbackward.match.rowMajor=false", "-Dbackward.split=false",
        "-Dbackward.isDefEq.lazyWhnfCore=false", "-Dbackward.isDefEq.lazyProjDelta=false",
        "-Dbackward.isDefEq.respectTransparency=false", "-Dbackward.isDefEq.implicitBump=false",
        "-Dbackward.eqns.nonrecursive=false", "-Dbackward.eqns.deepRecursiveSplit=false",
        "-Dbackward.linearNoConfusionType=false", "-Dbackward.synthInstance.canonInstances=false",
        "-Dbackward.dsimp.proofs=true", "-Dbackward.dsimp.instances=true",
        "-Dbackward.dsimp.useDefEqAttr=false", "-Dbackward.whnf.reducibleClassField=false",
        "-Dbackward.grind.inferPattern=true", "-Dbackward.grind.normalizer=false",
        "-Dbackward.inferInstanceAs.wrap=false", "-Dbackward.isDefEq.throwOnStuckAfterApp=true",
    ]
    BACKWARD_SHARE = 0.3
    PRINTING = [
        "-Dpp.all=true", "-Dpp.explicit=true", "-Dpp.universes=true", "-Dpp.proofs=true",
        "-Dpp.funBinderTypes=true", "-Dpp.analyze=true", "-Dpp.deepTerms=true",
        "-Dpp.oneline=true", "-Dpp.mvars=false", "-Dpp.fieldNotation=false",
        "-Dpp.notation=false", "-Dpp.coercions=false", "-Dpp.letVarTypes=true",
        "-Dpp.structureInstances=false", "-Dpp.match=false", "-Dpp.beta=true",
        "-Dpp.instantiateMVars=false", "-Dpp.raw=true", "-Dpp.unicode.fun=true",
    ]
    PRINTING_SHARE = 0.35
    HEARTBEATS = ["-DmaxHeartbeats=2000", "-DmaxHeartbeats=20000", "-DsynthInstance.maxHeartbeats=500"]
    HEARTBEATS_SHARE = 0.08

    def __init__(self, config):
        super().__init__(config)
        root = os.path.join(self.ffl_root, "projects", "lean")
        info = {}
        try:
            with open(os.path.join(root, "toolchain.json")) as f:
                info = json.load(f)
        except Exception:
            pass
        self.toolchains = {t["variant"]: t for t in info.get("toolchains", [])
                           if os.path.exists(t.get("lean", ""))}
        if not self.toolchains:
            lean = shutil.which("lean")
            if lean:
                self.toolchains["path"] = {"variant": "path", "lean": lean,
                                           "leanc": shutil.which("leanc"), "env": {}}
        self.mem_limit_mb = int(config.get("execution", {}).get("mem_limit_mb", 0) or 0)
        self.timeout = int(config.get("execution", {}).get("timeout", 60) or 60)

    # ── command construction ──────────────────────────────────────────

    def _toolchain(self, dryrun):
        if not dryrun and "asan" in self.toolchains and random.random() < self.ASAN_SHARE:
            return self.toolchains["asan"]
        return self.toolchains.get("relwithassert") or next(iter(self.toolchains.values()))

    def _prefix(self, tc, workdir):
        env = dict(tc.get("env", {}))
        env.update({"HOME": workdir, "TMPDIR": workdir, "LEAN_BACKTRACE": "1"})
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        # ASan reserves terabytes of address space; a `ulimit -v` would make
        # every sanitized run fail at startup.
        cap = f"ulimit -v {self.mem_limit_mb * 1024}; " \
            if self.mem_limit_mb and tc.get("variant") != "asan" else ""
        return f"export {exports}; {cap}"

    @staticmethod
    def _weighted(options):
        vals, weights = zip(*options)
        return list(random.choices(vals, weights=weights, k=1)[0])

    def _flags(self, workdir):
        flags = list(self.BASE) + self._weighted(self.LINTER)
        flags += self._weighted(self.THREADS) + self._weighted(self.ASYNC)
        for flag, p in self.CHECKS + self.COMPILER:
            if random.random() < p:
                flags.append(flag)
        for opt, fname, p in self.OUTPUTS:
            if random.random() < p:
                flags.append(f"{opt}={os.path.join(workdir, fname)}")
        for flag, p in self.REPORT:
            if random.random() < p:
                flags.append(flag)
        if random.random() < self.BACKWARD_SHARE:
            flags.append(random.choice(self.BACKWARD))
        if random.random() < self.PRINTING_SHARE:
            flags += random.sample(self.PRINTING, random.randint(1, 3))
        if random.random() < self.HEARTBEATS_SHARE:
            flags.append(random.choice(self.HEARTBEATS))
        return flags

    def _build_command(self, seed_file, workdir, content, dryrun=False):
        tc = self._toolchain(dryrun)
        pre = self._prefix(tc, workdir)
        lean = tc["lean"]
        if dryrun:
            flags = self.BASE + ["-Dlinter.all=false", "-Dcompiler.postponeCompile=false"]
            return f"{pre}{lean} {' '.join(flags)} {seed_file} </dev/null", "elab", tc
        has_main = bool(_MAIN_RE.search(content))
        flags = self._flags(workdir)
        if has_main and tc.get("leanc") and not _NONDET_RE.search(content) and \
                random.random() < self.DIFF_SHARE:
            return self._diff_command(tc, pre, seed_file, workdir), "diff", tc
        if has_main and random.random() < self.RUN_SHARE:
            # `--run` must be the last option: everything after the file is
            # passed to `main`.
            flags = [f for f in flags if not f.startswith(("-o=", "-i=", "-c="))]
            run_t = max(10, self.timeout // 2)
            return (f"{pre}timeout -s KILL {run_t} {lean} {' '.join(flags)} --run {seed_file} "
                    f"</dev/null"), "run", tc
        return f"{pre}{lean} {' '.join(flags)} {seed_file} </dev/null", "elab", tc

    def _diff_command(self, tc, pre, seed_file, workdir):
        """Native vs interpreted. Prints FFL markers the driver parses:
        the elaboration's own output (`lean -c`) is a prefix of `lean
        --run`'s, and what follows it is `main`'s output."""
        lean, leanc = tc["lean"], tc["leanc"]
        c, exe = os.path.join(workdir, "out.c"), os.path.join(workdir, "prog")
        e1, e2, e3 = (os.path.join(workdir, n) for n in ("elab.out", "native.out", "interp.out"))
        opts = "-DElab.inServer=true -Dlinter.all=false -Dcompiler.postponeCompile=false"
        t = max(10, self.timeout // 4)
        return (
            f"{pre}{lean} {opts} -c={c} {seed_file} </dev/null >{e1} 2>&1; ELAB_RC=$?; "
            f"cat {e1}; "
            f"if [ $ELAB_RC -eq 0 ]; then "
            f"  {leanc} -O3 -o {exe} {c} >/dev/null 2>&1 || {{ echo FFL_LEANC_FAILED; exit 0; }}; "
            f"  timeout -s KILL {t} {exe} </dev/null >{e2} 2>&1; NAT_RC=$?; "
            f"  timeout -s KILL {t} {lean} {opts} --run {seed_file} </dev/null >{e3} 2>&1; INT_RC=$?; "
            f"  echo FFL_NATIVE_RC=$NAT_RC FFL_INTERP_RC=$INT_RC; "
            f"  echo FFL_NATIVE_BEGIN; cat {e2}; echo FFL_NATIVE_END; "
            f"  echo FFL_INTERP_BEGIN; cat {e3}; echo FFL_INTERP_END; "
            f"fi; exit $ELAB_RC")

    # ── differential verdict ──────────────────────────────────────────

    @staticmethod
    def _section(text, name):
        m = re.search(rf"FFL_{name}_BEGIN\n(.*?)FFL_{name}_END", text, re.S)
        return m.group(1) if m else None

    def _diff_verdict(self, stdout):
        m = re.search(r"FFL_NATIVE_RC=(\d+) FFL_INTERP_RC=(\d+)", stdout)
        if not m:
            return None
        nat_rc, int_rc = int(m.group(1)), int(m.group(2))
        native = self._section(stdout, "NATIVE") or ""
        interp = self._section(stdout, "INTERP") or ""
        elab = stdout[:stdout.find("FFL_NATIVE_RC")] if "FFL_NATIVE_RC" in stdout else ""
        elab = elab.replace("FFL_LEANC_FAILED\n", "")
        if 137 in (nat_rc, int_rc):
            return None                   # one side ran out of time
        if not interp.startswith(elab):
            return None                   # the elaboration output differed; not comparable
        interp_main = interp[len(elab):]
        if nat_rc != int_rc and (nat_rc >= 128 or nat_rc < 0):
            v = classify(native, nat_rc, tool="run")
            if v["is_bug"] or nat_rc in (134, 139, 132, 136, 135):
                return {"kind": "miscompile",
                        "signature": f"native binary dies (rc {nat_rc}), interpreter rc {int_rc}",
                        "is_bug": True, "is_valid": True}
        if nat_rc != int_rc or native.strip() != interp_main.strip():
            if "PANIC" in native or "PANIC" in interp_main:
                return None               # both report a panic differently: not a value
            return {"kind": "miscompile",
                    "signature": "native and interpreted output differ"
                                 + (f" (rc {nat_rc} vs {int_rc})" if nat_rc != int_rc else ""),
                    "is_bug": True, "is_valid": True}
        return None

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd, tool, tc = "unknown", "elab", {}
        rc, stdout, stderr = 1, "", ""
        seed_file = None
        content = seed.content
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.lean")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(content)
            cmd, tool, tc = self._build_command(seed_file, workdir, content,
                                                dryrun=getattr(self, "dryrun_mode", False))
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        duration = time.time() - start
        uses_meta = bool(_META_RE.search(content))
        unsafe = bool(_UNSAFE_RE.search(content))
        out = (stdout or "") + "\n" + (stderr or "")
        if tool == "diff":
            verdict = classify(out.replace("FFL_NATIVE_BEGIN", "").replace("FFL_INTERP_BEGIN", ""),
                               rc, content, uses_meta, "run", unsafe)
            if not verdict["is_bug"] and not unsafe:
                verdict = self._diff_verdict(stdout or "") or verdict
        else:
            verdict = classify(out, rc, content, uses_meta, tool, unsafe)
        crashed = bool(verdict["is_bug"])
        if verdict["kind"] in ("resource", "timeout", "rejected") and rc != 0 and \
                not re.search(r"error", out):
            stderr = f"error: lean {verdict['kind']}\n" + (stderr or "")
        sig = verdict["signature"] if crashed else None
        res = ExecutionResult(rc, stdout, stderr, duration, crashed, sig)
        res.command = cmd
        res.seed_file = seed_file
        res.tool = tool
        res.verdict = verdict["kind"]
        return res

    # ── oracle hooks (BaseDriver) ─────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        return bool(classify((stdout or "") + "\n" + (stderr or ""), return_code)["is_bug"])

    def extract_crash_signature(self, stdout, stderr, return_code):
        return classify((stdout or "") + "\n" + (stderr or ""), return_code)["signature"]
