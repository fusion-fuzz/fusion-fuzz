"""
projects/typescript/driver.py — run a fused TypeScript program through the
Go-native compiler (projects/typescript/setup.py) and report whether what
came back is a bug.

tsc is a *checker*, not an interpreter: nothing is executed, so "invalid"
means a type or syntax diagnostic (`error TS<n>`), which a fused program
usually produces and which says nothing about the compiler. What does:

  * `panic:` — a Go panic. `Debug failure.` and `Unexpected node.` are the
    compiler's own assertions (tsc/internal/debug); a nil dereference or
    an index out of range is the runtime catching it later.
  * `fatal error:` — the Go runtime: stack overflow (deep recursion in
    the checker), concurrent map read/write (the checker runs in
    parallel), out of memory.

Exit statuses come from tsc/internal/execute/tsc/compile.go: 0 clean,
1/2 diagnostics, 3 invalid project, 5 not implemented. A panic exits 2
through the Go runtime, so the return code alone cannot decide — the
oracle is pattern-based.

Options
-------
Each test case carries its own `// @option: value` directives, which the
upstream test harness turns into compiler options; the driver does the
same, and adds random ones. The important axes: `--target` (the whole
downlevel-emit machinery per level), `--strict` and the individual
strictness flags (different checker paths), `--module`, `--declaration`
(the declaration emitter, a separate printer), `--checkers N` (the
parallel checker).
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult


class TypeScriptDriver(BaseDriver):

    # The compiler's own option table, read from the binary under test
    # (`tsc --all`) rather than hard-coded: this is a moving target and a
    # stale table is worse than none. TypeScript 7 removed `--target es5`
    # and `es3`, `--module amd/system/umd`, `--moduleResolution classic`
    # and a dozen flags outright, and the corpus is full of test cases
    # that ask for them. Passing one makes tsc exit on a *configuration*
    # error (TS5108/TS6046) before it reads a line of the program, so the
    # child is recorded invalid for a reason fusion had nothing to do
    # with — 12% of the first measurement's failures.
    _OPTION_TABLE = None

    # Options that require a companion (TS5069/TS5051) and options that
    # exclude one another (TS5053), verified against the built binary.
    _REQUIRES = {
        # `tsc --all` still lists this one, and it is still parsed, but
        # it is refused without its companion (TS5052).
        "emitdecoratormetadata": ("experimentaldecorators",),
        "declarationmap": ("declaration", "composite"),
        "emitdeclarationonly": ("declaration", "composite"),
        "isolateddeclarations": ("declaration", "composite"),
        "inlinesources": ("sourcemap", "inlinesourcemap"),
        "sourceroot": ("sourcemap", "inlinesourcemap"),
        "maproot": ("sourcemap", "declarationmap"),
    }
    _CONFLICTS = {
        "sourcemap": ("inlinesourcemap",),
        "inlinesourcemap": ("sourcemap",),
    }
    # Refused on a command line (TS6230) or harness-only.
    _NEVER = {"composite", "tsbuildinfofile", "incremental", "watch", "project",
              "build", "all", "help", "init", "version", "locale", "pprofdir",
              "generatecpuprofile", "generatetrace", "plugins",
              # Listed by `tsc --all` and still parsed, then refused as
              # removed (TS5102). Downlevel iteration went with the ES5
              # target it existed for.
              "downleveliteration"}

    def _option_table(self):
        """{lowercase option name: set of accepted values, or None when
        the option takes no enumerated value} from the compiler under
        test. Queried once per process."""
        cls = type(self)
        if cls._OPTION_TABLE is not None:
            return cls._OPTION_TABLE
        table = {}
        try:
            _rc, out, err = self._run_command(f"{self.tsc_bin} --all")
            text = re.sub(r'\x1b\[[0-9;]*m', '', (out or "") + (err or ""))
            for block in re.split(r'\n(?=--\w)', text):
                m = re.match(r'--([A-Za-z]+)', block)
                if not m:
                    continue
                vals = re.search(r'^one of: (.+)$', block, re.M) or \
                    re.search(r'^one or more: (.+)$', block, re.M)
                accepted = None
                if vals:
                    accepted = set()
                    for v in vals.group(1).split(","):
                        # `es6/es2015` is one value under two spellings.
                        accepted.update(x.strip().lower() for x in v.split("/"))
                table[m.group(1).lower()] = accepted
        except Exception:
            table = {}
        cls._OPTION_TABLE = table
        return table

    # Options that change *which* code the compiler runs, drawn per
    # execution on top of whatever the seed asked for.
    FUZZ_OPTIONS = [
        "--target es2015", "--target es2017", "--target es2020",
        "--target es2022", "--target esnext",
        "--module commonjs", "--module es2015", "--module esnext",
        "--module node16", "--module nodenext", "--module preserve",
        "--moduleResolution node10", "--moduleResolution bundler",
        "--strict", "--strictNullChecks", "--strictFunctionTypes",
        "--strictBindCallApply", "--strictPropertyInitialization",
        "--noImplicitAny", "--noImplicitThis", "--useUnknownInCatchVariables",
        "--exactOptionalPropertyTypes", "--noUncheckedIndexedAccess",
        "--useDefineForClassFields", "--experimentalDecorators",
        # --emitDecoratorMetadata needs --experimentalDecorators beside
        # it, so it is drawn as the pair or not at all.
        "--experimentalDecorators --emitDecoratorMetadata",
        # Deliberately absent: --noUnusedLocals / --noUnusedParameters.
        # A fused program almost always leaves one half's binding unread,
        # so those fail nearly every child for a reason that is a
        # property of concatenation rather than of the compiler.
        "--noImplicitReturns",
        "--verbatimModuleSyntax", "--isolatedModules", "--erasableSyntaxOnly",
        "--jsx preserve", "--jsx react", "--jsx react-jsx",
        "--skipLibCheck", "--noResolve", "--stableTypeOrdering",
    ]
    # Emit paths. `--noEmit` is the cheap default; emitting exercises the
    # transformers and the declaration printer, which are as large a
    # surface as the checker and have no other way in.
    EMIT_MODES = [
        ("--noEmit", 0.55),
        ("--declaration --emitDeclarationOnly", 0.2),
        ("--declaration --declarationMap --sourceMap", 0.15),
        ("--sourceMap --inlineSources", 0.1),
    ]

    _DIRECTIVE_RE = re.compile(r'^\s*//\s*@([A-Za-z]+)\s*:\s*([^\r\n]*)$', re.M)
    _PANIC_RE = re.compile(r'^(panic|fatal error):\s*(.+)$', re.M)
    # The first frame inside the compiler itself; runtime/debug frames
    # above it are the same for every panic.
    _FRAME_RE = re.compile(
        r'^(github\.com/microsoft/TypeScript/tsc/internal/[\w./]+\.[\w.()*]+)\(', re.M)
    _SANITIZER_RE = re.compile(r'SUMMARY: (\w+Sanitizer): ([\w-]+)')

    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.memory_limit_mb = int(exec_cfg.get("mem_limit_mb", self.DEFAULT_MEM_LIMIT_MB) or 0)
        self.tsc_bin = os.path.join(self.ffl_root, "projects", "typescript", "tsc-bin")

    def _seed_options(self, content):
        """The seed's own `// @option: value` directives as command-line
        options. Unknown names (harness-only directives such as
        `@noTypesAndSymbols`), removed options and values this compiler
        no longer accepts are dropped rather than passed on."""
        table = self._option_table()
        opts = []
        for name, value in self._DIRECTIVE_RE.findall(content):
            key = name.lower()
            if key in self._NEVER or (table and key not in table):
                continue
            value = value.strip()
            accepted = table.get(key)
            if accepted:
                # A list-valued option (`--lib es2015,dom`) is passed only
                # when every element is still accepted.
                parts = [v.strip().lower() for v in value.split(",") if v.strip()]
                if not parts or any(v not in accepted for v in parts):
                    continue
                opts.append(f"--{name} {value}")
            elif value.lower() in ("true", ""):
                opts.append(f"--{name}")
            elif value.lower() == "false":
                continue                       # the default; `--x false` is an error
            else:
                opts.append(f"--{name} {value}")
        return self._resolve_deps(opts)

    def _resolve_deps(self, opts):
        """Drop an option whose companion is missing (TS5069/TS5051) or
        that excludes one already present (TS5053). Both are refusals to
        start, so the child would be recorded invalid without ever having
        been checked."""
        # Two passes: a companion may be spelled *after* the option that
        # needs it (directives appear in the order the test author wrote
        # them), so membership is decided against the whole set, not
        # against what has been emitted so far.
        present = {k.lstrip("-").lower()
                   for o in opts for k in o.split() if k.startswith("-")}
        out = []
        for opt in opts:
            key = opt.split()[0].lstrip("-").lower()
            need = self._REQUIRES.get(key)
            if need and not (present & set(need)):
                continue
            emitted = {k.lstrip("-").lower()
                       for o in out for k in o.split() if k.startswith("-")}
            if emitted & set(self._CONFLICTS.get(key, ())):
                continue
            out.append(opt)
        return out

    # How often an execution adds options of its own at all. The corpus
    # is a decade of test cases written before the strictness flags
    # existed, so `--strict` and its family reject a large share of them
    # outright ("Property has no initializer" was 7% of the second
    # measurement's failures). They stay in the pool — strictNullChecks
    # is one of the checker's largest subsystems and nothing else reaches
    # it — but most executions now run with what the seed itself asked
    # for, the same rule the gcc/clang drivers follow for -std.
    _EXTRA_OPTION_SHARE = 0.45

    def _random_options(self, seed_opts):
        """Extra options, never contradicting one the seed asked for: a
        second `--target` wins over the first and silently discards the
        level the test was written for."""
        if random.random() >= self._EXTRA_OPTION_SHARE:
            return []
        given = {o.split()[0].lstrip("-").lower() for o in seed_opts}
        table = self._option_table()

        def usable(o):
            parts = o.split()
            key = parts[0].lstrip("-").lower()
            if key in given or (table and key not in table):
                return False
            accepted = table.get(key)
            return not (accepted and parts[1].lower() not in accepted)

        pool = [o for o in self.FUZZ_OPTIONS if usable(o)]
        return self._resolve_deps(random.sample(pool, min(len(pool), random.randint(1, 2))))

    def _emit_mode(self, seed_opts):
        given = {o.split()[0].lstrip("-").lower() for o in seed_opts}
        if given & {"noemit", "declaration", "emitdeclarationonly", "outfile", "outdir"}:
            return []
        modes, weights = zip(*self.EMIT_MODES)
        chosen = random.choices(modes, weights=weights)[0]
        return self._resolve_deps(["--" + part.strip()
                                   for part in chosen.split("--") if part.strip()])

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        seed_file = None
        rc, stdout, stderr = 1, "", ""
        try:
            seed_file = os.path.join(workdir, f"{seed.id}.ts")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(seed.content)
            seed_opts = self._seed_options(seed.content)
            opts = seed_opts + self._random_options(seed_opts) + self._emit_mode(seed_opts)
            if random.random() < 0.2:
                opts.append(f"--checkers {random.choice((2, 4, 8))}")
            env = ""
            if self.memory_limit_mb:
                # Go's own soft limit, so the runtime reports it rather
                # than being OOM-killed with no output.
                env = f"GOMEMLIMIT={self.memory_limit_mb}MiB GOGC=100 "
            out_dir = os.path.join(workdir, "out")
            cmd = (f"ulimit -c 0; {env}{self.tsc_bin} {' '.join(opts)} "
                   f"--outDir {out_dir} {seed_file}").strip()
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
            emit = self._emit_digest(out_dir)
            if emit:
                stdout = (stdout or "") + f"\ntsc-emit digest={emit}\n"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        crashed = self._check_crash(stdout, stderr, rc)
        signature = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
        res = ExecutionResult(rc, stdout, stderr, time.time() - start, crashed, signature)
        res.command = cmd
        res.seed_file = seed_file
        return res

    @staticmethod
    def _emit_digest(out_dir):
        """A digest of what the compiler *produced*, appended to stdout.

        A clean `tsc --noEmit` prints nothing at all, so every child that
        checks cleanly looked identical to the harness's diversity
        measure — 1 distinct output among 280 valid children. The emitted
        JavaScript and declaration files are the compiler's real product
        and they differ whenever the two fused halves differ, so hashing
        them gives the measure something to see.

        Encoded in the letters g..z on purpose: the harness masks digits
        and long hex runs out of a program's output before fingerprinting
        it, so a decimal or hex digest would collapse to the same "#".
        """
        if not os.path.isdir(out_dir):
            return ""
        h = 0
        for dirpath, _dirs, files in os.walk(out_dir):
            for name in sorted(files):
                try:
                    with open(os.path.join(dirpath, name), "rb") as f:
                        blob = f.read(200_000)
                except OSError:
                    continue
                for b in name.encode() + blob:
                    h = (h * 31 + b) & 0xFFFFFFFFFFFF
        if not h:
            return ""
        alphabet = "ghijklmnopqrstuvwxyz"
        enc = ""
        while h:
            enc = alphabet[h % 20] + enc
            h //= 20
        return enc

    # ── crash oracle ──────────────────────────────────────────────────

    def _check_crash(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        # Resource exhaustion is the fused program's, not the compiler's.
        if re.search(r'out of memory|cannot allocate memory|GOMEMLIMIT|'
                     r'runtime: out of memory|too many open files', text):
            return False
        # A stack overflow in the checker is deep *user* recursion — a
        # deliberately deep generic in the corpus reaches it on its own,
        # and `core.ApplyDebugStackLimit` exists precisely because the
        # limit is expected to be hit. Only a panic counts.
        if re.search(r'fatal error: stack overflow', text) and "panic:" not in text:
            return False
        return super()._check_crash(stdout, stderr, return_code)

    @staticmethod
    def _mask(s):
        s = re.sub(r'0x[0-9a-fA-F]+', '0x', s)
        s = re.sub(r'/[\w./\-]+\.ts\b', '<file>', s)
        s = re.sub(r'\b\d+\b', 'N', s)
        return re.sub(r'\s+', ' ', s).strip()

    def extract_crash_signature(self, stdout, stderr, return_code):
        text = (stderr or "") + "\n" + (stdout or "")
        m = self._PANIC_RE.search(text)
        if m:
            kind, msg = m.group(1), self._mask(m.group(2))
            # `Debug failure. <what>` is the compiler's own assertion; the
            # message identifies the site better than the frame does.
            frame = self._FRAME_RE.search(text)
            where = frame.group(1).split("tsc/internal/")[-1] if frame else ""
            return f"{kind}: {msg}" + (f" @ {where}" if where else "")
        m = self._SANITIZER_RE.search(text)
        if m:
            return f"{m.group(1)}: {m.group(2)}"
        return super().extract_crash_signature(stdout, stderr, return_code) or "unknown"
