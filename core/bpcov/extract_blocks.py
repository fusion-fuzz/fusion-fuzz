#!/usr/bin/env python3
"""Extract basic-block leader addresses (link-time vaddrs) from an ELF with
objdump's linear disassembly. Works on stripped binaries.

Leaders: targets of direct jumps, the instruction after a conditional jump,
and the first non-padding instruction after an unconditional jmp/ret (start
of the next region, usually a function entry). A leader is kept only if it
starts an instruction in the disassembly and is not padding.
Memory: compact arrays (~9 bytes per instruction), so binaries with tens of
millions of instructions are fine. Output: one hex address per line."""
import bisect, re, subprocess, sys
from array import array

binary, out = sys.argv[1], sys.argv[2]
INS = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$')
FUNC = re.compile(r'^([0-9a-f]+) <(.+)>:$')
# Sanitizer runtimes (ASan/UBSan/...) statically linked into the target run
# some of their code with every signal blocked through raw syscalls (e.g.
# ASan's pthread_create interceptor, before the new thread restores its
# mask); a probe hit there is fatal, and it is not the target's code anyway.
EXCLUDE = re.compile(r'^(_?__(asan|lsan|ubsan|sanitizer|interceptor|interception|msan|tsan|hwasan|dfsan|sancov)'
                     r'|_ZN\d+__(asan|lsan|ubsan|sanitizer|interception|msan|tsan|hwasan|dfsan)'
                     r'|_ZN\d+__sanitizer|___interceptor_|__interceptor_)')
# ... and, anywhere in the (mangled) name: the runtime's static helpers and
# extern "C" glue have no prefix but take or live in sanitizer types
# (`_ZL10read_iovecPvPN11__sanitizer...`, `real_pthread_create`, which ASan
# calls with every signal blocked).
EXCLUDE_ANY = re.compile(r'\d__(sanitizer|asan|lsan|ubsan|interception|msan|tsan|hwasan|dfsan)'
                         r'|Interceptor|_interceptor|^real_pthread_|^_GLOBAL__sub_I_(sanitizer|asan|lsan|ubsan)')
def excluded_name(name):
    return bool(EXCLUDE.match(name) or EXCLUDE_ANY.search(name))
TGT = re.compile(r'^([0-9a-f]+)\b')
PAD = {"nop", "nopw", "nopl", "int3", "xchg", "data16", "cs", "hlt", "ud2"}
UNCOND_END = {"jmp", "jmpq", "ud2", "hlt"}
NORMAL, CJMP, END, PADDING, PAD_END = 0, 1, 2, 3, 4   # PAD_END: ud2/hlt, both
EXCL = 0x80                                           # flag: inside an excluded function

# objdump labels a function with one of its names, and an interceptor is
# usually labelled with the plain libc name it replaces (`memset`, not
# `__interceptor_memset`); exclude every address any sanitizer symbol names.
excl_addrs = set()
info_addrs = set()
for flags in ([], ["-D"]):
    r = subprocess.run(["nm", "--defined-only"] + flags + [binary], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "TtWwiI" and excluded_name(parts[2]):
            excl_addrs.add(int(parts[0], 16))
        if len(parts) == 3 and parts[1] in "Tt" and parts[2].endswith("_info"):
            info_addrs.add(int(parts[0], 16))
# GHC-compiled code (tables next to code) interleaves info tables and other
# data with the code in .text, and probing block leaders found by linear
# disassembly corrupts GHC at -O1 and above (heap corruption, then a jump into
# the heap; root cause not pinned down, the code bytes themselves stay intact).
# Probing only the closures' entry points (`*_info`, data always lies before
# them) is safe: one block per closure entry.
if len(info_addrs) > 1000:
    with open(out, "w") as f:
        for a in sorted(info_addrs):
            f.write("%x\n" % a)
    print(f"{binary}: GHC code, {len(info_addrs)} closure entry points", file=sys.stderr)
    sys.exit(0)

addrs, cls, targets = array("Q"), bytearray(), array("Q")
excluded = False
n_excl_funcs = 0
p = subprocess.Popen(["objdump", "-d", "--no-show-raw-insn", "-w", binary],
                     stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
for line in p.stdout:
    m = INS.match(line)
    if not m:
        f = FUNC.match(line)
        if f:
            excluded = excluded_name(f.group(2)) or int(f.group(1), 16) in excl_addrs
            n_excl_funcs += excluded
        continue
    addr, mn, ops = int(m.group(1), 16), m.group(2), (m.group(3) or "")
    if mn in ("bnd", "notrack") and ops:
        parts = ops.split(None, 1); mn = parts[0]; ops = parts[1] if len(parts) > 1 else ""
    if mn in ("ud2", "hlt"):
        c = PAD_END
    elif mn in PAD:
        c = PADDING
    elif mn in UNCOND_END or mn.startswith("ret"):
        c = END
    elif mn.startswith("j"):
        c = CJMP
    else:
        c = NORMAL
    addrs.append(addr); cls.append(c | (EXCL if excluded else 0))
    if mn.startswith("j") and "*" not in ops:
        t = TGT.match(ops.strip())
        if t:
            targets.append(int(t.group(1), 16))
p.wait()

n = len(addrs)
excl = bytes(1 if c & EXCL else 0 for c in cls)
cls = bytearray(c & 0x7f for c in cls)
assert all(addrs[i] < addrs[i + 1] for i in range(0, n - 1, max(1, n // 1000))), "unsorted disassembly"
leaders = set()
for t in targets:
    i = bisect.bisect_left(addrs, t)
    if i < n and addrs[i] == t and cls[i] not in (PADDING, PAD_END):
        leaders.add(t)
for i in range(n - 1):
    c = cls[i]
    if c == CJMP:
        if cls[i + 1] not in (PADDING, PAD_END):
            leaders.add(addrs[i + 1])
    elif c in (END, PAD_END):
        j = i + 1
        while j < n and cls[j] in (PADDING, PAD_END):
            j += 1
        if j < n:
            leaders.add(addrs[j])
def is_excluded(a):
    i = bisect.bisect_left(addrs, a)
    return i < n and addrs[i] == a and excl[i]
leaders = {a for a in leaders if not is_excluded(a)}
with open(out, "w") as f:
    for a in sorted(leaders):
        f.write("%x\n" % a)
print(f"{binary}: {n} insns, {len(leaders)} block leaders ({n_excl_funcs} sanitizer-runtime functions excluded)", file=sys.stderr)
