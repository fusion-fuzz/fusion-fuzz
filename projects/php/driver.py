import os
import random
import re
import shlex
import shutil
import stat
import tempfile
import threading
import time
from core.driver import BaseDriver, ExecutionResult


class PHPDriver(BaseDriver):
    """
    PHP driver: invokes the ASAN-instrumented PHP CLI directly.
    FFL runs inside the ffl-php container where the PHP binary lives under
    {ffl_root}/projects/php/php-src/sapi/cli/php.
    """

    BLOCKED_FUNCTIONS = [
        # Process control
        "pcntl_fork", "pcntl_exec", "pcntl_alarm", "pcntl_wait", "pcntl_waitpid",
        "pcntl_signal", "pcntl_wexitstatus", "pcntl_wifexited", "pcntl_wifsignaled",
        "posix_kill", "posix_mkfifo", "posix_setuid", "posix_setgid", "posix_setsid",
        # Shell execution
        "system", "exec", "shell_exec", "passthru", "proc_open", "popen",
        # Filesystem modification — prevent PHP from corrupting the exec dir or
        # escaping to the host filesystem (root cause of the chmod 000 incident)
        "chmod", "chown", "chgrp", "chdir", "chroot",
        "mkdir", "rmdir", "rename", "unlink", "link", "symlink", "copy",
        # Network — avoids hangs and external side-effects
        "fsockopen", "pfsockopen",
    ]

    # ---------------------------------------------------------------
    # Opcache and JIT
    #
    # PHP's JIT is where a large share of the engine's miscompilation and
    # memory-safety bugs live, and none of it is reachable from the plain
    # CLI: opcache is a zend_extension that has to be loaded, and the JIT
    # is off until it is given a buffer. The build now passes
    # `--enable-opcache=shared` so `modules/opcache.so` exists.
    #
    # The hot counters are lowered alongside the mode. Tracing JIT only
    # compiles a loop or function once it has run `jit_hot_loop` /
    # `jit_hot_func` times, and a fused seed is short, so the intent is to
    # reach the compiler on the first execution rather than never. Measured
    # on this build, a hot loop runs 1615ms with `jit=disable` and 92ms
    # with `jit=tracing`, so the JIT is demonstrably compiling; lowering the
    # thresholds did not change that number either way on a loop that hot.
    # They are kept because a short seed is the case they are meant for,
    # not because the speedup above is evidence for them.
    #
    # `opcache.jit` is a CRTO quadruple: C=CPU-specific opts, R=register
    # allocation, T=trigger, O=optimisation level. The named modes are
    # aliases — `tracing` is 1254 and `function` is 1205 — and the spread
    # below walks the register allocator and the trigger independently,
    # since those are the two digits that change which code the JIT emits.
    JIT_MODES = [
        None,          # opcache not loaded at all: the interpreter path
        "disable",     # opcache loaded, JIT off: exercises the optimiser only
        "tracing",     # 1254
        "function",    # 1205
        "1235",        # tracing trigger, no global register allocation
        "1201",        # compile on first execution, minimal optimisation
        "1254",
        "0254",        # tracing without CPU-specific instruction selection
    ]
    JIT_WEIGHTS = [22, 10, 20, 14, 8, 8, 10, 8]

    # opcache write-protects its shared memory when asked. A bug that
    # writes through a stale pointer into that region becomes an immediate
    # fault with a clean stack instead of silent corruption surfacing
    # somewhere unrelated. It costs an mprotect per request.
    JIT_PROTECT_RATE = 0.30

    # Clean up stale .php files every this many executions.
    _CLEANUP_INTERVAL = 500
    # Stale threshold: a .php file older than this many seconds is considered orphaned.
    _STALE_AGE_SECS = 60

    def __init__(self, config):
        super().__init__(config)
        self.php_bin = os.path.join(
            self.ffl_root, "projects", "php", "php-src", "sapi", "cli", "php"
        )
        self.modules_dir = os.path.join(
            self.ffl_root, "projects", "php", "php-src", "modules"
        )
        self.phpt_deps_dir = os.path.join(
            self.ffl_root, "projects", "php", "phpt_deps"
        )
        # Every execution gets its own directory under here, holding the
        # program plus links to the fixtures of the php-src test
        # directories its parents came from (seed metadata `dep_dirs` /
        # `source_dir`, see projects/php/setup.py's collect_phpt). That is
        # what makes `__DIR__ . '/foo.inc'` and `require 'foo.inc'`
        # resolve the way they do under run-tests.php. The previous shared
        # directory held one flat set of fixtures under mangled names, so
        # no such include ever resolved.
        self._exec_dir = os.path.join(self.fused_base, "php_exec")

        self._exec_count = 0
        self._cleanup_lock = threading.Lock()
        self._dep_listing_cache = {}
        self._dep_listing_lock = threading.Lock()

        self._ensure_exec_dir()

    # ------------------------------------------------------------------
    # Directory lifecycle
    # ------------------------------------------------------------------

    def _ensure_exec_dir(self):
        """
        Create the exec dir if missing and guarantee it has rwx permissions.
        Repairs the mode-000 case that occurs when fuzzed PHP calls chmod().
        Also removes stale .php files and any garbage left by previous runs.
        """
        os.makedirs(self._exec_dir, exist_ok=True)
        current_mode = stat.S_IMODE(os.stat(self._exec_dir).st_mode)
        if current_mode != 0o755:
            os.chmod(self._exec_dir, 0o755)

        # Remove orphaned .php files from previous sessions.
        self._remove_stale_php_files(max_age=0)

    def _remove_stale_php_files(self, max_age=None):
        """
        Delete per-execution directories older than max_age seconds — left
        behind only when a run was killed between mkdtemp and rmtree.
        max_age=0 removes ALL of them (used at startup for a clean slate).
        Also restores exec dir permissions in case PHP chmod'd it during the run.
        """
        if max_age is None:
            max_age = self._STALE_AGE_SECS
        now = time.time()
        try:
            # Heal permissions first so we can actually list the directory.
            current_mode = stat.S_IMODE(os.stat(self._exec_dir).st_mode)
            if current_mode != 0o755:
                os.chmod(self._exec_dir, 0o755)

            for name in os.listdir(self._exec_dir):
                path = os.path.join(self._exec_dir, name)
                try:
                    if max_age == 0 or (now - os.path.getmtime(path)) > max_age:
                        if os.path.isdir(path) and not os.path.islink(path):
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            os.unlink(path)
                except OSError:
                    pass
        except OSError:
            pass

    def _dep_files(self, rel_dir: str):
        """[(name, absolute path)] of the fixtures in one php-src test
        directory, listed once and cached — a hot loop links them on
        every execution."""
        with self._dep_listing_lock:
            cached = self._dep_listing_cache.get(rel_dir)
        if cached is not None:
            return cached
        base = os.path.normpath(os.path.join(self.phpt_deps_dir, rel_dir))
        files = []
        if base.startswith(self.phpt_deps_dir) and os.path.isdir(base):
            for name in os.listdir(base):
                full = os.path.join(base, name)
                if os.path.isfile(full):
                    files.append((name, full))
        with self._dep_listing_lock:
            self._dep_listing_cache[rel_dir] = files
        return files

    def _link_fixtures(self, workdir: str, seed) -> list:
        """Link the parents' fixtures beside the program. Returns the
        directories used (relative to phpt_deps), first parent first, so
        a name present in both keeps the first parent's file — the one
        whose `__DIR__` semantics the host half was written against."""
        meta = getattr(seed, "metadata", None) or {}
        dirs = list(meta.get("dep_dirs") or [])
        if meta.get("source_dir") and meta["source_dir"] not in dirs:
            dirs.insert(0, meta["source_dir"])
        used = []
        for rel in dirs:
            files = self._dep_files(rel)
            if not files:
                continue
            used.append(rel)
            for name, full in files:
                dst = os.path.join(workdir, name)
                if os.path.lexists(dst):
                    continue
                try:
                    os.symlink(full, dst)
                except OSError:
                    pass
        return used

    def _maybe_cleanup(self):
        """Periodically remove stale .php files without blocking the hot path."""
        with self._cleanup_lock:
            self._exec_count += 1
            trigger = (self._exec_count % self._CLEANUP_INTERVAL == 0)
        if trigger:
            self._remove_stale_php_files()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_seed_filename(self, seed_id: str) -> str:
        """
        Derive a filesystem-safe filename from a seed ID.
        Strips path separators and other shell-special characters so that
        os.path.join(exec_dir, filename) cannot escape the exec dir.
        """
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', seed_id)
        return (safe[:64] or "seed") + ".php"

    # A .phpt section header is `--NAME--` and nothing else. The previous
    # test, "starts and ends with --", also matched a line of program text
    # such as `-----END EC PRIVATE KEY-----` inside a PEM string, and cut
    # the --FILE-- section off there: the executed program was truncated
    # mid-string and booked as "Unclosed '('", while `php -l` on the same
    # fused test passed. run-tests.php uses the same anchored form.
    _PHPT_SECTION_RE = re.compile(r'^--([A-Z_]+)--\s*$')

    def _parse_phpt(self, content):
        sections = {}
        current = None
        for line in content.splitlines():
            m = self._PHPT_SECTION_RE.match(line)
            if m:
                current = m.group(1)
                sections[current] = ""
            elif current is not None:
                sections[current] += line + "\n"
        return sections

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = None
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            sections = self._parse_phpt(seed.content)
            php_code = sections.get("FILE", seed.content).strip()
            ini_content = sections.get("INI", "").strip()

            workdir = tempfile.mkdtemp(prefix="x", dir=self._exec_dir)
            dep_dirs = self._link_fixtures(workdir, seed)
            seed_file = os.path.join(workdir, self._safe_seed_filename(seed.id))
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(php_code)

            ini_args = [
                f'-d disable_functions={",".join(self.BLOCKED_FUNCTIONS)}',
                # Restrict PHP file access to the exec dir and deps only.
                # This is the primary sandbox that prevents PHP from touching
                # anything outside its working directory. open_basedir
                # checks the *resolved* path, so the fixture tree the
                # links point into has to be allowed as well.
                f'-d open_basedir="{workdir}:{self.phpt_deps_dir}"',
                # Disable network access to avoid hangs.
                '-d allow_url_fopen=0',
                '-d allow_url_include=0',
            ]

            # `.` is the workdir (cwd), where the fixtures are linked; the
            # parents' own directories follow for a fixture that resolves
            # a further include relative to itself.
            inc = [".", workdir] + [os.path.join(self.phpt_deps_dir, d) for d in dep_dirs]
            ini_args.append("-d " + shlex.quote("include_path=" + ":".join(inc)))

            # A seed that names opcache or jit in its own INI section is
            # asking for it and always gets it; every other seed draws a
            # mode, so the JIT is exercised by the whole corpus rather than
            # only by the handful of .phpt files written to test it.
            seed_wants_jit = ("opcache" in ini_content.lower()
                              or "jit" in ini_content.lower())
            jit_mode = ("tracing" if seed_wants_jit
                        else random.choices(self.JIT_MODES,
                                            weights=self.JIT_WEIGHTS, k=1)[0])
            if jit_mode is not None:
                # opcache has been built by default since PHP 8 and this
                # build has it linked in statically, so there is no
                # opcache.so and nothing to load. Requiring one is what kept
                # the JIT switched off for every execution, including the
                # seeds whose own INI section asked for it. Load the module
                # only when a shared build actually produced one.
                opcache = os.path.join(self.modules_dir, "opcache.so")
                if os.path.exists(opcache):
                    ini_args += [
                        f'-d extension_dir="{self.modules_dir}"',
                        f'-d zend_extension="{opcache}"',
                    ]
                ini_args += [
                    '-d opcache.enable=1',
                    '-d opcache.enable_cli=1',
                    f'-d opcache.jit={jit_mode}',
                    '-d opcache.jit_buffer_size=64M',
                    # Without these the tracing JIT never reaches its
                    # threshold on a seed this short; see JIT_MODES.
                    '-d opcache.jit_hot_loop=1',
                    '-d opcache.jit_hot_func=1',
                    '-d opcache.jit_hot_return=1',
                    '-d opcache.jit_hot_side_exit=1',
                ]
                if random.random() < self.JIT_PROTECT_RATE:
                    ini_args.append('-d opcache.protect_memory=1')

            for line in ini_content.splitlines():
                line = line.strip()
                # Skip blank lines, INI comment lines (;... or #...), lines without =
                if not line or line[0] in (';', '#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                # Do not let seed INI override the security settings we set above.
                if key.lower() in ('open_basedir', 'disable_functions',
                                   'allow_url_fopen', 'allow_url_include'):
                    continue
                # run-tests.php substitutes these before handing the INI
                # to php; without it `{PWD}/x.h` is a literal brace path.
                val = val.replace('{PWD}', workdir).replace('{TMP}', workdir)
                # The command line goes through `sh -c`, and a .phpt INI
                # value is free to contain shell syntax: `error_reporting=
                # E_ALL&~E_NOTICE` backgrounds php at the `&`, `sendmail_
                # path="cat > /tmp/x"` redirects, `session.save_path=
                # "a;b"` ends the command. Every such seed (130 with `;`,
                # 50 with `&` in this corpus) ran a truncated command and
                # was booked as invalid — as "sh: Syntax error" — for a
                # defect in this harness, not in the fused program.
                ini_args.append("-d " + shlex.quote(f"{key}={val}"))

            ini_flags = " ".join(ini_args)
            # Sanitizer runtime options, same spirit as the other adapters:
            #   handle_abort=1     a plain abort() (libgmp's "overflow in
            #                      mpz type", a zend_mm assertion) gets an
            #                      ASan stack trace, so it is grouped by
            #                      the frame that aborted instead of every
            #                      abort in the corpus sharing one
            #                      "SIGABRT" bucket
            #   allocator_may_return_null=1
            #                      a huge allocation returns NULL for PHP's
            #                      own memory_limit machinery to report,
            #                      rather than ASan aborting with a
            #                      "requested allocation size exceeds
            #                      maximum" that is not a PHP bug
            #   detect_leaks=1     kept on: php-src fixes leaks; they are
            #                      filed under output/bugs/php/leaks/ (see
            #                      extract_crash_signature) so they never
            #                      crowd out crashes
            env = ("ASAN_OPTIONS='handle_abort=1:abort_on_error=1:allocator_may_return_null=1:"
                   "detect_leaks=1:symbolize=1:print_stacktrace=1' "
                   "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' ")
            cmd = f"{env}{self.php_bin} {ini_flags} {seed_file}"
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        duration = time.time() - start
        crashed = self._check_crash(stdout, stderr, rc)
        sig = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
        res = ExecutionResult(rc, stdout, stderr, duration, crashed, sig)
        # The recorded command is what goes into a crash bundle's
        # test.sh, which core/orchestrator.py rewrites to run from the
        # bundle directory. The per-execution workdir is gone by then,
        # and with `open_basedir` still naming it the reproducer was
        # refused with "open_basedir restriction in effect" — every
        # bundle from the first verification run failed to reproduce for
        # that reason alone. Point the sandbox at the bundle instead.
        if workdir:
            res.command = cmd.replace(workdir, "$SCRIPT_DIR")
            res.seed_file = seed_file.replace(workdir, "$SCRIPT_DIR") if seed_file else seed_file
        else:
            res.command = cmd
            res.seed_file = seed_file

        self._maybe_cleanup()
        return res

    # First stack frame inside php-src itself, i.e. the allocation or
    # crash site — skipping the allocator/interceptor frames above it.
    _PHP_FRAME_RE = re.compile(
        r"^\s+#\d+\s+0x[0-9a-f]+\s+in\s+([A-Za-z_]\w*)\s+\S*php-src/([^\s:]+)", re.M)

    def _first_php_frame(self, text, skip=("malloc", "calloc", "realloc", "free",
                                             "__zend_malloc", "__zend_calloc", "__zend_realloc",
                                             "_emalloc", "_erealloc", "_ecalloc", "_efree",
                                             "_safe_emalloc", "_estrdup", "_estrndup",
                                             "abort", "raise", "kill", "__interceptor_abort",
                                             "__assert_fail", "__assert_fail_base")):
        for m in self._PHP_FRAME_RE.finditer(text):
            func, path = m.group(1), m.group(2)
            if func in skip or func.startswith(("zend_mm_", "__asan", "__interceptor", "__sanitizer")):
                continue
            return f"{func}_at_{path.rsplit('/', 1)[-1]}"
        return None

    def extract_crash_signature(self, stdout, stderr, return_code):
        combined = (stderr or "") + "\n" + (stdout or "")
        for text in (stderr, stdout):
            m = re.search(r"(Assertion: .*)", text)
            if m:
                return m.group(1).strip()
        # A leak report's SUMMARY line names a byte count, which is
        # different for every run of the same leak, so every occurrence
        # became a new "bug". Group by the php-src frame that allocated
        # instead, under the ASAN:memory-leak prefix core/orchestrator.py
        # files into output/bugs/php/leaks/.
        if "LeakSanitizer" in combined or "leaked in" in combined:
            site = self._first_php_frame(combined)
            return f"ASAN:memory-leak_in_{site or 'unknown'}"
        for text in (stderr, stdout):
            m = re.search(r"(SUMMARY: .*)", text)
            if m:
                sig = m.group(1).strip()
                # "SUMMARY: AddressSanitizer: ABRT on unknown address ..."
                # (handle_abort=1) says nothing about *where*; the first
                # php-src frame does.
                k = re.match(r"SUMMARY: (\w+Sanitizer): (ABRT|SEGV|FPE|ILL|BUS)\b", sig)
                if k:
                    site = self._first_php_frame(combined)
                    if site:
                        # Drop the libc location (pthread_kill.c, the
                        # kill() zend_mm raises on heap corruption): the
                        # php-src frame is the bug's identity.
                        return f"SUMMARY: {k.group(1)}: {k.group(2)} in {site}"
                return sig
        return super().extract_crash_signature(stdout, stderr, return_code)
