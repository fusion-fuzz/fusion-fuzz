import subprocess
import os
import signal
import shutil
import sys
import tempfile
import time
import logging
import uuid
import importlib.util
import inspect
import re

logger = logging.getLogger("FFL.Driver")

class ExecutionResult:
    def __init__(self, return_code, stdout, stderr, time, crashed, signature=None):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = time
        self.crashed = crashed
        self.signature = signature
        self.command = None # Optional: Store command used

class BaseDriver:
    """
    Base Driver that executes a command defined in config.yaml on the local system.
    Includes common result analysis for crashes.
    """
    def __init__(self, config):
        self.config = config
        self.project_name = config.get('project_name', 'unknown')
        self.timeout = config.get('execution', {}).get('timeout', 5)
        # Derive FFL root from this file's location (core/driver.py → FusionFuzzLoop/)
        _core_dir = os.path.dirname(os.path.abspath(__file__))
        self.ffl_root = os.path.dirname(_core_dir)
        self.fused_base = os.path.join(self.ffl_root, ".fused")

    def prepare_environment(self):
        pass

    def _run_command(self, cmd, cwd=None):
        """
        Executes a shell command safely, handling binary output and encoding errors.
        Uses start_new_session=True so the entire process group (shell + all children)
        can be killed on timeout, preventing orphaned processes from leaking memory.
        Returns (return_code, stdout, stderr).
        """
        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd, start_new_session=True,
            )
            try:
                raw_out, raw_err = proc.communicate(timeout=self.timeout)
                stdout = raw_out.decode('utf-8', errors='replace')
                stderr = raw_err.decode('utf-8', errors='replace')
                return proc.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                # Kill the entire process group so go run + compiler + binary all die
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                proc.wait()
                return 124, "", "TIMEOUT"
        except Exception as e:
            return 1, "", str(e)

    def _make_workdir(self):
        """Create an isolated per-execution temp directory under .fused/<project>/."""
        base = os.path.join(self.fused_base, self.project_name)
        os.makedirs(base, exist_ok=True)
        return tempfile.mkdtemp(dir=base)

    def execute(self, seed):
        start = time.time()

        # 1. Isolated working directory — keeps the project root clean
        workdir = self._make_workdir()
        seed_file = None
        try:
            # 2. Write seed into workdir
            seed_file = os.path.join(workdir, f"{seed.id}.test")
            try:
                with open(seed_file, "w", encoding="utf-8") as f:
                    f.write(seed.content)
            except Exception as e:
                return ExecutionResult(1, "", f"Failed to write seed file: {e}", 0, False)

            # 3. Construct Harness Command (absolute seed path so cwd doesn't matter)
            cmd_template = self.config['execution']['command']
            cmd = cmd_template.format(seed_path=seed_file)

            # 4. Execute with workdir as CWD
            return_code, stdout, stderr = self._run_command(cmd, cwd=workdir)

        finally:
            # 5. Always wipe the workdir — removes seed + any side-effect files
            shutil.rmtree(workdir, ignore_errors=True)

        duration = time.time() - start

        # 6. Check Crash Patterns
        crashed = self._check_crash(stdout, stderr, return_code)

        # 7. Extract Signature (if crashed)
        signature = None
        if crashed:
            signature = self.extract_crash_signature(stdout, stderr, return_code)

        res = ExecutionResult(return_code, stdout, stderr, duration, crashed, signature)
        res.command = cmd
        res.seed_file = seed_file
        return res

    def _check_crash(self, stdout, stderr, return_code):
        """
        Common result analysis logic to detect crashes.
        """
        # if return_code not in [0, 1, 124]: 
        #     return True
            
        for pattern in self.config.get('analysis', {}).get('crash_patterns', []):
            if pattern in stdout or pattern in stderr:
                return True
        return False

    def extract_crash_signature(self, stdout, stderr, return_code):
        """
        Base signature extraction (ASAN or Return Code).
        """
        # 1. AddressSanitizer (ASAN)
        asan_pattern = r"SUMMARY: AddressSanitizer:\s+(.*)"
        match = re.search(asan_pattern, stderr)
        if match: return match.group(1).strip()
        match = re.search(asan_pattern, stdout)
        if match: return match.group(1).strip()

        return None # Return None to let Orchestrator fallback or handle generic logic

class DockerDriver(BaseDriver):
    """
    Base driver for projects that run inside a persistent Docker container.
    The project root is volume-mounted at /workspace, and seed files are
    exchanged via the host's .fused/ directory (mapped to /workspace/.fused/).

    Subclasses must set `container_name`, `container_image`, and `file_ext`, and
    must implement `_build_exec_cmd(container_path, seed) -> str`.

    Subclasses may also override:
      _container_run_args()          -- customize docker run flags/mounts
      _verify_container_mounts()     -- return False to force a container restart
      _prepare_content(seed)         -- preprocess seed content before writing
      _get_file_ext(seed)            -- dynamic extension (default: self.file_ext)
      _post_cleanup(host_temp, seed) -- remove extra side-effect files after execution
    """

    container_name: str = ""
    container_image: str = ""
    container_workspace: str = "/workspace"  # in-container mount point; override per project
    file_ext: str = ""

    def __init__(self, config):
        super().__init__(config)
        # Derive project root from the subclass file (projects/<name>/driver.py -> FFL root)
        subclass_file = inspect.getfile(self.__class__)
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(subclass_file)))
        )
        self.host_tmp = os.path.join(self.project_root, ".fused")
        os.makedirs(self.host_tmp, exist_ok=True)

    def _container_run_args(self) -> list:
        """Returns the full docker run argument list. Override to add extra mounts/options."""
        return [
            "docker", "run", "-dit",
            "--name", self.container_name,
            "-v", f"{self.project_root}:{self.container_workspace}",
            self.container_image,
        ]

    def _verify_container_mounts(self) -> bool:
        """Return False to trigger a container restart. Override for extra health checks."""
        return True

    def _ensure_container(self):
        """Ensures the container is running. Call from subclass __init__ when needed."""
        needs_restart = False
        try:
            subprocess.run(
                ["docker", "exec", self.container_name, "true"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if not self._verify_container_mounts():
                needs_restart = True
        except subprocess.CalledProcessError:
            needs_restart = True

        if needs_restart:
            print(f"Starting Docker container {self.container_name}...")
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                subprocess.run(self._container_run_args(), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to start Docker container {self.container_name}: {e}")

    def _prepare_content(self, seed) -> str:
        """Override to preprocess seed content before writing to the temp file."""
        return seed.content

    def _get_file_ext(self, seed) -> str:
        """Returns the file extension for this seed. Override for dynamic extensions."""
        return self.file_ext

    def _build_exec_cmd(self, container_path: str, seed) -> str:
        """
        Build and return the full docker exec command string.
        container_path is the in-container path to the seed file.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_exec_cmd")

    def _post_cleanup(self, host_temp: str, seed):
        """Override to remove extra files produced during execution."""
        pass

    def execute(self, seed):
        start = time.time()

        ext = self._get_file_ext(seed)
        host_temp = os.path.join(self.host_tmp, f"{seed.id}.{ext}")
        try:
            with open(host_temp, "w", encoding="utf-8") as f:
                f.write(self._prepare_content(seed))
        except Exception:
            return ExecutionResult(1, "", "Host Write Failed", 0, False)

        container_path = f"{self.container_workspace}/.fused/{seed.id}.{ext}"
        cmd = self._build_exec_cmd(container_path, seed)
        return_code, stdout, stderr = self._run_command(cmd)
        duration = time.time() - start

        if os.path.exists(host_temp):
            os.remove(host_temp)
        self._post_cleanup(host_temp, seed)

        crashed = self._check_crash(stdout, stderr, return_code)
        signature = None
        if crashed:
            signature = self.extract_crash_signature(stdout, stderr, return_code)

        res = ExecutionResult(return_code, stdout, stderr, duration, crashed, signature)
        res.command = cmd
        return res


#: Extra process names (exact `comm`) a project's frontend may leave
#: behind, beyond the obvious one named after the project itself.
PROJECT_PROCESS_NAMES = {
    "swift": ["swift-frontend"],
    "rust": ["rustc"],
    "clang": ["clang", "clang++"],
    "flang": ["flang", "flang-22", "flang-new"],
    "lfortran": ["lfortran"],
}

#: Command-line substrings identifying a project's frontend where its
#: process name is too generic to match on (cpython's interpreter is just
#: `python3`, which every other tool on the box is also called).
PROJECT_PROCESS_PATTERNS = {
    "cpython": ["build/python"],
}

#: A process younger than this (seconds) is assumed to be a *live*
#: execution, not a leak, and is left alone.
DEFAULT_STALE_AFTER = 120.0


def _process_table():
    """[(pid, ppid, etimes, comm, args)] for every process on this host."""
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etimes=,comm=,args="],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return []
    rows = []
    for line in res.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, etimes = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rows.append((pid, ppid, etimes, parts[3], parts[4] if len(parts) > 4 else ""))
    return rows


def _protected_pids(table):
    """This process and every ancestor of it — killing any of them takes
    the fuzzer (or the watchdog that restarts it) down with the leak."""
    parents = {pid: ppid for pid, ppid, _, _, _ in table}
    protected, pid = set(), os.getpid()
    while pid and pid not in protected:
        protected.add(pid)
        pid = parents.get(pid, 0)
    return protected


def cleanup_stale_processes(project_name: str, min_age: float = DEFAULT_STALE_AFTER):
    """
    Kill leaked compiler-or-interpreter-frontend processes for
    `project_name` to free resources between seed executions.

    Why this exists: BaseDriver._run_command's subprocess timeout only
    guarantees the *host*-side process (and its process group) is killed —
    it does not guarantee that a process a Docker-based driver started
    *inside* its container dies with it (docker exec doesn't reliably
    propagate the host client's termination into the container), nor that
    every child a compiler frontend spawned exits cleanly on SIGKILL of its
    parent. Left unchecked across many seed executions, these can pile up
    and progressively starve CPU/memory, making later executions (in the
    fuzzing loop, or during --dry-run/--pre-analysis) appear to hang even
    though each individual subprocess call still respected its own timeout.

    Only processes older than `min_age` are killed. An unconditional
    `pkill -9 -x flang` also kills the frontends the fuzzer's own worker
    threads are running *right now*: each one comes back as return code
    -9, which the analyzer cannot tell from a rejected test case, so every
    sweep silently books a batch of valid tests as syntax errors. The same
    applies to the by-path sweep below, which additionally used to match
    (and SIGKILL) any other tool invoked with the project's path on its
    command line — `python3 projects/flang/prune_corpus.py`, for instance.

    Shared by core/orchestrator.py's periodic in-loop cleanup and core/
    dryrun.py's --dry-run/--pre-analysis pass.
    """
    table = _process_table()
    if not table:
        return
    protected = _protected_pids(table)
    names = set(PROJECT_PROCESS_NAMES.get(project_name, []))
    names.add(project_name)
    patterns = list(PROJECT_PROCESS_PATTERNS.get(project_name, []))
    # A driver may also run the frontend out of the project's own tree
    # rather than under a well-known binary name.
    tree_marker = f"projects/{project_name}"
    safe_bins = ("vim", "nvim", "nano", "code", "git", "emacs", "less",
                 "tail", "ps", "grep", "docker")
    # The project's own tree also holds the tools that maintain it, and
    # those run under an interpreter — `python3 projects/flang/
    # prune_corpus.py` matches the by-tree sweep but is not a leaked
    # frontend. (A project whose frontend genuinely *is* an interpreter
    # names it in PROJECT_PROCESS_PATTERNS, which is not filtered here.)
    interpreters = ("python", "python3", "bash", "sh", "perl", "ruby")

    killed = 0
    for pid, _ppid, etimes, comm, args in table:
        if pid in protected or etimes < min_age:
            continue
        if any(comm.startswith(s) for s in safe_bins):
            continue
        by_name = comm in names
        by_pattern = any(pat in args for pat in patterns)
        by_tree = tree_marker in args and comm not in interpreters
        if not (by_name or by_pattern or by_tree):
            continue
        try:
            os.kill(pid, 9)
            killed += 1
        except OSError:
            pass

    if killed:
        logger.info(f"Reaped {killed} stale {project_name} process(es)")


def get_driver(config):
    project_name = config.get('project_name', '')
    driver_path = os.path.join("projects", project_name, "driver.py")
    
    if os.path.exists(driver_path):
        try:
            module_name = f"ffl_{project_name}_driver"
            spec = importlib.util.spec_from_file_location(module_name, driver_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module   # required for inspect.getfile to work
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseDriver) and obj not in (BaseDriver, DockerDriver):
                    logger.info(f"Loaded custom driver '{name}' from {driver_path}")
                    return obj(config)
                    
        except Exception as e:
            # Print detailed error to debug import failures (like the one user faced)
            logger.error(f"Failed to load custom driver from {driver_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to generic BaseDriver.")

    logger.info("Using Generic BaseDriver.")
    return BaseDriver(config)