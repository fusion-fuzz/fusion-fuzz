import os
import re
import stat
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
        # Shared execution directory: deps symlinked once, PHP scripts written
        # and removed per-execution. Avoids recreating ~964 symlinks every run.
        self._exec_dir = os.path.join(self.fused_base, "php_exec_shared")

        self._exec_count = 0
        self._cleanup_lock = threading.Lock()

        self._ensure_exec_dir()
        self._setup_exec_dir()

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

    def _setup_exec_dir(self):
        """Symlink all phpt_deps into the shared exec dir. Called once at init."""
        if not os.path.isdir(self.phpt_deps_dir):
            return
        for name in os.listdir(self.phpt_deps_dir):
            src = os.path.join(self.phpt_deps_dir, name)
            dst = os.path.join(self._exec_dir, name)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass

    def _remove_stale_php_files(self, max_age=None):
        """
        Delete .php files in the exec dir that are older than max_age seconds.
        max_age=0 removes ALL .php files (used at startup for a clean slate).
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
                if not name.endswith(".php"):
                    continue
                path = os.path.join(self._exec_dir, name)
                try:
                    if max_age == 0 or (now - os.path.getmtime(path)) > max_age:
                        os.unlink(path)
                except OSError:
                    pass
        except OSError:
            pass

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

    def _parse_phpt(self, content):
        sections = {}
        current = None
        for line in content.splitlines():
            if line.startswith("--") and line.endswith("--"):
                current = line.strip("-")
                sections[current] = ""
            elif current is not None:
                sections[current] += line + "\n"
        return sections

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = self._exec_dir
        seed_file = None
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            sections = self._parse_phpt(seed.content)
            php_code = sections.get("FILE", seed.content).strip()
            ini_content = sections.get("INI", "").strip()

            seed_file = os.path.join(workdir, self._safe_seed_filename(seed.id))
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(php_code)

            ini_args = [
                f'-d disable_functions={",".join(self.BLOCKED_FUNCTIONS)}',
                # Restrict PHP file access to the exec dir and deps only.
                # This is the primary sandbox that prevents PHP from touching
                # anything outside its working directory.
                f'-d open_basedir="{workdir}:{self.phpt_deps_dir}"',
                # Disable network access to avoid hangs.
                '-d allow_url_fopen=0',
                '-d allow_url_include=0',
            ]

            # Add phpt_deps to include_path as fallback for includes not in workdir.
            if os.path.isdir(self.phpt_deps_dir):
                ini_args.append(f'-d include_path=".:{self.phpt_deps_dir}"')

            use_jit = "opcache" in ini_content.lower() or "jit" in ini_content.lower()
            if use_jit:
                opcache = os.path.join(self.modules_dir, "opcache.so")
                if os.path.exists(opcache):
                    ini_args.append(f'-d extension_dir="{self.modules_dir}"')
                    ini_args.append(f'-d zend_extension="{opcache}"')

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
                ini_args.append(f"-d {key}={val}")

            ini_flags = " ".join(ini_args)
            cmd = f"{self.php_bin} {ini_flags} {seed_file}"
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            # Only remove the seed script — the shared dir and its symlinks stay.
            if seed_file:
                try:
                    os.unlink(seed_file)
                except OSError:
                    pass

        duration = time.time() - start
        crashed = self._check_crash(stdout, stderr, rc)
        sig = self.extract_crash_signature(stdout, stderr, rc) if crashed else None
        res = ExecutionResult(rc, stdout, stderr, duration, crashed, sig)
        res.command = cmd
        res.seed_file = seed_file

        self._maybe_cleanup()
        return res

    def extract_crash_signature(self, stdout, stderr, return_code):
        for text in (stderr, stdout):
            m = re.search(r"(Assertion: .*)", text)
            if m:
                return m.group(1).strip()
        for text in (stderr, stdout):
            m = re.search(r"(SUMMARY: .*)", text)
            if m:
                return m.group(1).strip()
        return super().extract_crash_signature(stdout, stderr, return_code)
