#!/usr/bin/env python3
"""subreap.py -- <command...>: run a command as a child of a process that
has declared itself a child subreaper, and reap every orphan that lands
on it.

Why: the fuzzer runs each test as `sh -c <compiler ...>` in a new
session and, on timeout, kills the whole process group. The shell dies,
its children (the compiler, llvm-symbolizer) die a moment later, and
since their parent is gone they are re-parented to PID 1 — which in
these containers is `sleep infinity` and never calls wait(). Every
timed-out test therefore leaves one to three zombies behind for the
life of the container: 10,582 in ffe-ruby on 2026-09-18. With this
wrapper as an ancestor, those orphans re-parent here instead and are
reaped at once. The wrapped command's own exit status is passed through.
"""
import ctypes, os, signal, sys, time

PR_SET_CHILD_SUBREAPER = 36

def main():
    if len(sys.argv) < 2 or sys.argv[1] != "--":
        sys.exit("usage: subreap.py -- <command...>")
    cmd = sys.argv[2:]
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except Exception:
        pass                                   # not Linux: plain pass-through
    child = os.fork()
    if child == 0:
        os.execvp(cmd[0], cmd)
    # Forward termination to the child so `timeout` and the watchdog work.
    def fwd(sig, _frame):
        try:
            os.kill(child, sig)
        except OSError:
            pass
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, fwd)
    status = None
    while True:
        try:
            pid, st = os.waitpid(-1, 0)        # reaps the child and every orphan
        except ChildProcessError:
            break
        if pid == child:
            status = st
            # keep reaping orphans that arrive for a moment after the child
            end = time.time() + 2.0
            while time.time() < end:
                try:
                    os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                time.sleep(0.1)
            break
    if status is None:
        sys.exit(0)
    if os.WIFSIGNALED(status):
        sys.exit(128 + os.WTERMSIG(status))
    sys.exit(os.WEXITSTATUS(status))

if __name__ == "__main__":
    main()
