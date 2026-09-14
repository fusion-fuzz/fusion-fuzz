"""
The C/C++ toolchain every adapter builds its target with.

One rule, applied everywhere: **clang, with LLVM's sanitizers**. Not gcc,
and not gcc's sanitizers.

Why it has to be uniform
------------------------
The sanitizers are the oracle. A finding is only as trustworthy as the
instrumentation that produced it, and mixing runtimes across adapters makes
two reports incomparable — the same memory error reads differently, the
suppression files do not carry over, and a bug triaged against one runtime
cannot be minimised against the other. Measured before this module existed:
tint was built entirely by gcc 13.3, php and triton were built by gcc 11.4
and clang together, and the remaining adapters each resolved a compiler
their own way.

Why gcc cannot simply be swapped in
-----------------------------------
For the LLVM-based targets it is not a preference. `LLVM_USE_SANITIZER`
makes LLVM's cmake emit `-fno-sanitize=function` and
`-fsanitize-blacklist=`, which gcc rejects outright — the build dies a few
dozen objects in, long after configure has reported success. gcc handles
`-fsanitize=address` perfectly well on its own; it is LLVM's own build
system that will not have it.

Why "a clang exists" is not enough
----------------------------------
A clang built from `LLVM_ENABLE_PROJECTS=clang` alone has no compiler-rt,
so it has no `sanitizer/asan_interface.h` and no runtime. It compiles
ordinary code fine and fails only once something includes that header,
thousands of objects into the build. So the compiler is probed with the
flags it will actually be given, rather than being taken on trust.

Usage
-----
    from core.toolchain import cmake_args, autotools_env, require_clang

    cc, cxx = require_clang()                 # raises if unusable
    args = cmake_args(sanitizers="Address;Undefined")   # for LLVM projects
    env  = autotools_env(sanitizers="address,undefined")  # for ./configure

Set FFL_NO_SANITIZERS=1 to build clang-but-uninstrumented, which is the
right thing when measuring throughput rather than hunting memory errors.
FFL_CC / FFL_CXX override the search.
"""

import os
import shutil
import subprocess

#: Names to try, newest first. A versioned clang is common on distributions
#: that do not ship an unversioned symlink.
_CLANG_NAMES = [
    ("clang", "clang++"),
    ("clang-20", "clang++-20"), ("clang-19", "clang++-19"),
    ("clang-18", "clang++-18"), ("clang-17", "clang++-17"),
    ("clang-16", "clang++-16"), ("clang-15", "clang++-15"),
    ("clang-14", "clang++-14"),
]

DEFAULT_SANITIZERS = "Address;Undefined"


def sanitizers_enabled():
    return os.environ.get("FFL_NO_SANITIZERS", "") not in ("1", "true", "yes")


def _probe(cc):
    """Whether this compiler can build sanitized code the way LLVM asks.

    The test compiles against `sanitizer/asan_interface.h` and passes
    `-fno-sanitize=function`, which is one of the flags LLVM's cmake adds.
    gcc rejects that flag and has no such header from clang's runtime, so
    both halves of the check matter.
    """
    if not cc:
        return False
    src = "#include <sanitizer/asan_interface.h>\nint main(void){return 0;}\n"
    try:
        r = subprocess.run(
            [cc, "-fsanitize=address,undefined", "-fno-sanitize=function",
             "-x", "c", "-", "-o", os.devnull],
            input=src, text=True, capture_output=True, timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_clang(require_sanitizers=None):
    """(cc, cxx) as absolute paths, or (None, None).

    With require_sanitizers, only a clang that passes _probe is accepted.
    """
    if require_sanitizers is None:
        require_sanitizers = sanitizers_enabled()

    env_cc, env_cxx = os.environ.get("FFL_CC"), os.environ.get("FFL_CXX")
    candidates = []
    if env_cc and env_cxx:
        candidates.append((env_cc, env_cxx))
    for cc, cxx in _CLANG_NAMES:
        a, b = shutil.which(cc), shutil.which(cxx)
        if a and b:
            candidates.append((a, b))

    for cc, cxx in candidates:
        if not require_sanitizers or _probe(cc):
            return cc, cxx
        print(f"  ({cc} cannot build sanitized code — no compiler-rt?)")
    return None, None


def require_clang(require_sanitizers=None):
    """Same, but refuses to continue without one.

    Failing loudly is deliberate. Falling back to gcc is what produced the
    mixed builds this module exists to prevent, and a silent fallback is
    indistinguishable from success until someone reads `.comment` out of
    the finished binary.
    """
    cc, cxx = find_clang(require_sanitizers)
    if cc and cxx:
        return cc, cxx
    raise RuntimeError(
        "no usable clang found. Every target is built with clang and LLVM's "
        "sanitizers; gcc is not a fallback (LLVM's own build system emits "
        "clang-only sanitizer flags). Install a distro clang that ships "
        "compiler-rt — on Debian/Ubuntu `apt-get install clang` — or set "
        "FFL_CC/FFL_CXX, or FFL_NO_SANITIZERS=1 to build uninstrumented.")


def cmake_args(sanitizers=None, llvm_style=True):
    """cmake -D arguments selecting clang and the sanitizers.

    llvm_style picks how the sanitizers are requested: LLVM-based projects
    take `LLVM_USE_SANITIZER=Address;Undefined`, everything else takes
    ordinary `-fsanitize=` flags in CMAKE_{C,CXX}_FLAGS.
    """
    cc, cxx = require_clang()
    args = [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]
    if not sanitizers_enabled():
        return args
    san = sanitizers or DEFAULT_SANITIZERS
    if llvm_style:
        args.append(f'-DLLVM_USE_SANITIZER={san}')
    else:
        flags = (f"-fsanitize={san.replace(';', ',').lower()} "
                 f"-fno-omit-frame-pointer")
        args += [f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}",
                 f"-DCMAKE_EXE_LINKER_FLAGS=-fsanitize="
                 f"{san.replace(';', ',').lower()}"]
    return args


def autotools_env(sanitizers=None, base=None):
    """Environment for a ./configure build: CC, CXX and sanitizer flags.

    Returned as a dict to merge into os.environ, so a caller that already
    set CFLAGS keeps it — the sanitizer flags are appended, not assigned.
    """
    cc, cxx = require_clang()
    env = dict(base or os.environ)
    env["CC"], env["CXX"] = cc, cxx
    env["LD"] = cxx
    if not sanitizers_enabled():
        return env
    san = (sanitizers or DEFAULT_SANITIZERS).replace(";", ",").lower()
    flags = f"-fsanitize={san} -fno-omit-frame-pointer"
    for key in ("CFLAGS", "CXXFLAGS"):
        env[key] = (env.get(key, "") + " " + flags).strip()
    env["LDFLAGS"] = (env.get("LDFLAGS", "") + f" -fsanitize={san}").strip()
    return env


def describe():
    """One line for a setup log, so the choice is visible in the build output."""
    cc, cxx = find_clang()
    if not cc:
        return "toolchain: NO CLANG FOUND"
    san = DEFAULT_SANITIZERS if sanitizers_enabled() else "off"
    return f"toolchain: {cc} / {cxx}, sanitizers {san}"
