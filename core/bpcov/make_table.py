#!/usr/bin/env python3
"""Build the probe table for an ELF: (vaddr, original byte) for every block
leader not yet covered, sorted by vaddr. The binary itself is NOT modified.
usage: make_table.py <elf> <leaders.txt> <covered.txt|-> <out_table> [out_image]
With out_image, also writes a copy of the ELF with int3 at every probe; the
runtime maps its executable segments over the original at startup.
Memory use is independent of the binary's size (it is mmap'ed)."""
import mmap, os, shutil, struct, sys

elf, leaders_f, covered_f, out_tab = sys.argv[1:5]
out_img = sys.argv[5] if len(sys.argv) > 5 else None
fh = open(elf, "rb")
data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
assert data[:4] == b"\x7fELF" and data[4] == 2, "64-bit ELF only"
e_phoff, = struct.unpack_from("<Q", data, 0x20)
e_phentsize, e_phnum = struct.unpack_from("<HH", data, 0x36)
segs = []
for i in range(e_phnum):
    p_type, p_flags, p_off, p_vaddr, _, p_filesz = struct.unpack_from("<IIQQQQ", data, e_phoff + i * e_phentsize)
    if p_type == 1 and p_flags & 1:                  # PT_LOAD, executable
        segs.append((p_vaddr, p_filesz, p_off))
def off_of(va):
    for v, sz, off in segs:
        if v <= va < v + sz:
            return off + va - v
    return None
covered = set()
if covered_f != "-":
    covered = {int(l, 16) for l in open(covered_f) if l.strip()}
tab = []
for l in open(leaders_f):
    if not l.strip():
        continue
    va = int(l, 16)
    if va in covered:
        continue
    o = off_of(va)
    if o is None or data[o] == 0xCC:
        continue
    tab.append((va, data[o], o))
tab.sort()
with open(out_tab + ".tmp", "wb") as f:
    f.write(struct.pack("<Q", len(tab)))
    f.write(b"".join(struct.pack("<Q", va) for va, _, _ in tab))
    f.write(bytes(b for _, b, _ in tab))
os.replace(out_tab + ".tmp", out_tab)
if out_img:
    tmp = out_img + ".tmp"
    shutil.copyfile(elf, tmp)
    with open(tmp, "r+b") as g:
        m = mmap.mmap(g.fileno(), 0)
        for _, _, o in tab:
            m[o] = 0xCC
        m.flush(); m.close()
    os.chmod(tmp, 0o644)
    os.replace(tmp, out_img)        # atomic: running processes keep the old inode
print(f"{out_tab}: {len(tab)} probes ({len(covered)} covered skipped)", file=sys.stderr)
