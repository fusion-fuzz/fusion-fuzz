import logging
import os
import re
import shlex
import struct
import subprocess
import sys
import threading
import time

logger = logging.getLogger("FFL.BPCov")
HERE = os.path.dirname(os.path.abspath(__file__))


class CoverageManager:
    """Fuzzer-side driver for breakpoint coverage.

    modules: executables to instrument (all processes running one of them are
             covered, including ones exec'd by a compiler driver)
    mode:    "preload" (LD_PRELOAD runtime; dynamic executables) or
             "ptrace" (bptrace; also static executables)
    work_dir: where tables/images/config live (one per run)
    rebuild_secs: how often to drop covered blocks from the probe tables
    stats_path: CSV written every `stats_secs` with coverage/corpus counters
    """

    MOD_SHIFT = 56
    COUNTER_COLS = ["execs", "valid", "crashes", "unique_crashes", "corpus", "added_seeds",
                    "evicted", "rej_novelty", "rej_size", "rej_generation", "rej_budget", "novelty_thr"]

    def __init__(self, modules, mode="preload", work_dir=".bpcov", rebuild_secs=300,
                 stats_path=None, stats_secs=60):
        if mode not in ("preload", "ptrace"):
            raise ValueError(f"unknown coverage mode {mode!r}")
        self.mode = mode
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.rebuild_secs = rebuild_secs
        self.cfg_path = os.path.join(self.work_dir, "current.cfg")
        self._lock = threading.Lock()
        self._tls = threading.local()
        self.version = 0
        self.total_new = 0
        self._new_since_rebuild = 0
        self.counters = {}              # filled in by the orchestrator
        self.t0 = time.time()

        self.so_path = os.path.join(self.work_dir, "bpcov.so")
        self.bptrace_path = os.path.join(self.work_dir, "bptrace")
        self._compile()

        self.modules = []
        for i, exe in enumerate(modules):
            real = os.path.realpath(exe)
            if not os.path.isfile(real):
                raise FileNotFoundError(f"coverage module not found: {exe}")
            st = os.stat(real)
            # x3: extraction also excludes sanitizer aliases and helpers (see extract_blocks.py)
            tag = re.sub(r"[^A-Za-z0-9]+", "_", real)[-60:] + f"_{st.st_size}_{int(st.st_mtime)}_x3"
            # Block extraction depends only on the binary, so it can be shared
            # between runs (FFL_BPCOV_CACHE); keyed by path, size and mtime.
            cache = os.environ.get("FFL_BPCOV_CACHE") or self.work_dir
            os.makedirs(cache, exist_ok=True)
            leaders = os.path.join(cache, tag + ".leaders")
            if not os.path.exists(leaders):
                logger.info(f"bpcov: extracting basic blocks of {real} (one-off)...")
                tmp = f"{leaders}.tmp{os.getpid()}"
                subprocess.run([sys.executable, os.path.join(HERE, "extract_blocks.py"), real, tmp], check=True)
                os.replace(tmp, leaders)
            n = sum(1 for _ in open(leaders))
            self.modules.append(dict(id=i, exe=real, tag=tag, leaders=leaders, covered=set(), n_blocks=n))
        self._rebuild()
        threading.Thread(target=self._rebuild_loop, daemon=True).start()
        # absolute: the orchestrator chdirs into rotating workspaces
        self.stats_path = os.path.abspath(stats_path) if stats_path else None
        self.stats_secs = stats_secs
        self._stats_started = False

    def start_stats(self):
        """Start the per-minute CSV; elapsed time counts from this call."""
        if not self.stats_path or self._stats_started:
            return
        self._stats_started = True
        self.t0 = time.time()
        with open(self.stats_path, "w") as f:
            f.write("elapsed_s,covered_blocks,probes_left,table_version," + ",".join(
                self.COUNTER_COLS) + "\n")
        threading.Thread(target=self._stats_loop, daemon=True).start()

    # ---- setup -------------------------------------------------------
    def _compile(self):
        def build(dst, args):
            src = os.path.join(HERE, args[0])
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                return
            subprocess.run(["gcc", "-O2", "-w", "-o", dst + ".tmp"] + [src] + args[1:], check=True)
            os.replace(dst + ".tmp", dst)
        build(self.so_path, ["bpcov.c", "-fPIC", "-shared", "-ldl", "-lpthread"])
        build(self.bptrace_path, ["bptrace.c"])

    def _rebuild(self):
        """(Re)build each module's table/image without the covered blocks,
        then atomically switch current.cfg to them. Processes already running
        keep their (old) mapping; new ones pick the new tables up."""
        v = self.version + 1
        lines = []
        for m in self.modules:
            with self._lock:
                covered = sorted(m["covered"])
            cov_path = os.path.join(self.work_dir, f"{m['tag']}.covered")
            with open(cov_path, "w") as f:
                f.write("".join("%x\n" % a for a in covered))
            tab = os.path.join(self.work_dir, f"{m['tag']}.v{v}.tab")
            img = os.path.join(self.work_dir, f"{m['tag']}.v{v}.img")
            subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"), m["exe"], m["leaders"],
                            cov_path, tab, img], check=True, stderr=subprocess.DEVNULL)
            m["probes"] = struct.unpack("<Q", open(tab, "rb").read(8))[0]
            lines.append(f"{m['exe']} {tab} {img} {m['id']}\n")
        tmp = self.cfg_path + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(lines))
        os.replace(tmp, self.cfg_path)
        old = self.version - 1                     # keep one previous version for in-flight runs
        self.version = v
        for m in self.modules:
            for ext in ("tab", "img"):
                p = os.path.join(self.work_dir, f"{m['tag']}.v{old}.{ext}")
                if old >= 1 and os.path.exists(p):
                    os.remove(p)
        with self._lock:
            self._new_since_rebuild = 0

    def _rebuild_loop(self):
        while True:
            time.sleep(self.rebuild_secs)
            if self._new_since_rebuild:
                try:
                    self._rebuild()
                    logger.info(f"bpcov: probe tables rebuilt (v{self.version}, "
                                f"{self.covered_blocks()} blocks covered)")
                except Exception as e:
                    logger.warning(f"bpcov: rebuild failed: {e}")

    def _stats_loop(self):
        while True:
            time.sleep(self.stats_secs)
            c = self.counters
            row = [int(time.time() - self.t0), self.covered_blocks(), self.probes_left(), self.version] + \
                  [c.get(k, 0) for k in self.COUNTER_COLS]
            with open(self.stats_path, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")

    # ---- per execution (worker side) -----------------------------------
    def wrap(self, cmd, out_path):
        if self.mode == "preload":
            env = (f"export BPCOV_CONFIG={shlex.quote(self.cfg_path)} BPCOV_OUT={shlex.quote(out_path)} "
                   f"LD_PRELOAD={shlex.quote(self.so_path)}; ")
            return env + cmd
        return f"{shlex.quote(self.bptrace_path)} {shlex.quote(self.cfg_path)} {shlex.quote(out_path)} -- sh -c {shlex.quote(cmd)}"

    @staticmethod
    def read(out_path):
        """Records written by one execution (and its children); removes the file."""
        try:
            with open(out_path, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    return []
                n = min(struct.unpack("<Q", head)[0], 1 << 22)
                data = f.read(8 * n)
            return list(struct.unpack("<%dQ" % (len(data) // 8), data))
        except OSError:
            return []
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

    # ---- parent side ---------------------------------------------------
    def update(self, records):
        """Merge one execution's records; returns the number of new blocks."""
        if not records:
            return 0
        mask = (1 << self.MOD_SHIFT) - 1
        new = 0
        with self._lock:
            for r in records:
                mid = r >> self.MOD_SHIFT
                if mid >= len(self.modules):
                    continue
                cov = self.modules[mid]["covered"]
                va = r & mask
                if va not in cov:
                    cov.add(va)
                    new += 1
            self.total_new += new
            self._new_since_rebuild += new
        return new

    def covered_blocks(self):
        return sum(len(m["covered"]) for m in self.modules)

    def probes_left(self):
        return sum(m.get("probes", 0) for m in self.modules)
