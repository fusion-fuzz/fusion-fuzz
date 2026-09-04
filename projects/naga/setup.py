"""
projects/naga/setup.py — build naga-cli from source for FusionFuzz.

Why build it here rather than use the Dockerfile's copy
-------------------------------------------------------
The Dockerfile used to run `cargo install --git ... naga-cli`, and `cargo
install` builds in release mode. Cargo's release profile sets
`debug-assertions = false` and `overflow-checks = false`, which switches off
both of naga's internal oracles:

  debug_assert!   naga's own invariant checks — the direct counterpart of
                  LLVM_ENABLE_ASSERTIONS for clang. In a release build a
                  violated invariant produces wrong output instead of a
                  report.

  overflow-checks Naga spends most of its time on index, offset and
                  alignment arithmetic over types and layouts. Without this
                  an overflow wraps silently and the wrong value flows on
                  into a size or an offset; with it the process panics at
                  the point the arithmetic went wrong.

So this builds naga-cli out of the wgpu checkout that is cloned here
anyway, keeping release-level optimisation but turning both back on — the
same Release-plus-assertions trade the clang and mlir adapters make.

Set FFL_NAGA_PREBUILT=1 to skip the build and use whatever `naga` is on
PATH, which is the old behaviour.
"""

import os
import shutil
import subprocess

#: Rust has no equivalent of LLVM_USE_SANITIZER on stable — `-Zsanitizer`
#: needs a nightly toolchain — so these two flags are the whole oracle
#: budget for a stable build.
RUSTFLAGS = "-C debug-assertions=yes -C overflow-checks=yes"


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:200]}")
    subprocess.run(["sh", "-c", cmd], check=True, cwd=cwd, env=env)


def naga_bin(project_root):
    """The binary this adapter builds. driver.py prefers it over PATH."""
    return os.path.join(project_root, "wgpu", "target", "release", "naga")


def _find_cargo():
    """Locate cargo without trusting PATH.

    The rust image puts cargo in /usr/local/cargo/bin via ENV, but a login
    shell (`docker exec -it ... bash`, or anything run with `bash -lc`)
    sources /etc/profile, which resets PATH and drops it. Relying on
    `shutil.which` alone means the build is silently skipped for anyone who
    runs setup from an interactive shell."""
    found = shutil.which("cargo")
    if found:
        return found
    candidates = [
        os.path.join(os.environ.get("CARGO_HOME", ""), "bin", "cargo"),
        "/usr/local/cargo/bin/cargo",
        os.path.expanduser("~/.cargo/bin/cargo"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _build(project_root, src_root):
    if os.environ.get("FFL_NAGA_PREBUILT") == "1":
        print("FFL_NAGA_PREBUILT=1: using the naga on PATH")
        return
    cargo = _find_cargo()
    if not cargo:
        # Not a warning. The Dockerfile no longer installs naga-cli, so
        # without this build there is no naga at all — carrying on would
        # leave every execution failing for a reason that looks nothing
        # like a missing toolchain.
        raise RuntimeError(
            "cargo not found. naga-cli is built from source here (see the "
            "module docstring); install a Rust toolchain, or set "
            "FFL_NAGA_PREBUILT=1 to use a naga already on PATH.")

    env = dict(os.environ)
    # Appended, not assigned: the container may already set RUSTFLAGS, and
    # replacing it would silently drop whatever it had.
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " " + RUSTFLAGS).strip()
    jobs = os.environ.get("FFL_NAGA_JOBS", "")
    j = f" -j{jobs}" if jobs.isdigit() else ""

    print(f"Building naga-cli with RUSTFLAGS={env['RUSTFLAGS']!r}")
    # --locked so the build uses the checked-in Cargo.lock; without it a
    # fresh resolve can pull a dependency version the tree was never tested
    # against, and a build failure there looks like a naga problem.
    _run(f"{cargo} build --release --locked -p naga-cli{j}",
         cwd=src_root, env=env)

    built = naga_bin(project_root)
    if os.path.exists(built):
        print(f"naga built: {built}")
    else:
        raise RuntimeError(
            f"cargo build reported success but {built} is missing")


def setup(project_root):
    """Clone wgpu's trunk branch for Naga WGSL seeds, then build naga-cli
    from it with its assertions and overflow checks enabled."""
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "wgpu")
    seed_root = os.path.join(src_root, "naga", "tests")

    if not os.path.exists(src_root):
        _run(
            "git clone --depth=1 --branch trunk "
            f"https://github.com/gfx-rs/wgpu.git {src_root}"
        )
    elif not os.path.exists(os.path.join(src_root, ".git")):
        raise RuntimeError(f"{src_root} exists but is not a git checkout")
    else:
        _run("git fetch --depth=1 origin trunk && git checkout trunk "
             "&& git pull --ff-only", cwd=src_root)

    _build(project_root, src_root)

    if not os.path.exists(seed_root):
        print(f"Warning: Naga WGSL seed root not found at {seed_root}")
    else:
        print(f"Naga setup complete. Seeds: {seed_root}")
