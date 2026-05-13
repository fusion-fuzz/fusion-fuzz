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

    def save_artifact(self, seed, result, type_label, signature="N/A"):
        """
        Saves a beautified Markdown crash report to the output directory.
        """
        folder = os.path.join("output", "bugs", self.project_name)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        filepath = os.path.join(folder, f"{type_label}_{seed.id}.md")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"## THIS BUG REPORT IS GENERATED BY FuzzFusionLoop ##\n\n")
                
                f.write(f"### Metadata\n")
                f.write(f"- **ID:** `{seed.id}`\n")
                f.write(f"- **Signature:** `{signature}`\n")
                f.write(f"- **Return Code:** `{result.return_code}`\n")
                f.write(f"- **Execution Time:** `{result.execution_time:.4f}s`\n")
                if result.command:
                    f.write(f"- **Command:** `{result.command}`\n\n")
                
                f.write(f"### STDERR\n")
                f.write("```bash\n")
                err = result.stderr if result.stderr else "(Empty)"
                f.write(err)
                f.write("\n```\n\n")

                f.write(f"### STDOUT\n")
                f.write("```bash\n")
                out = result.stdout if result.stdout else "(Empty)"
                f.write(out)
                f.write("\n```\n\n")
                
                f.write(f"### CONTENT\n")
                f.write("```\n") 
                f.write(seed.content)
                f.write("\n```\n")
        except Exception as e:
            logger.error(f"Failed to write artifact {filepath}: {e}")

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