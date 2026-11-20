// ptrace-based breakpoint coverage, for executables LD_PRELOAD cannot reach
// (static binaries such as Go's toolchain) or that manage signals themselves.
// usage: bptrace <config> <out> -- cmd args...
//   config lines: "<exe> <table> <image|-> <module_id>" (same as BPCOV_CONFIG)
//   out: same format as bpcov.so's BPCOV_OUT (u64 count, u64 records)
// Follows fork/vfork/clone/exec. When a traced process execs a configured
// executable, its executable segments are overwritten (via /proc/pid/mem)
// with the pre-armed image. SIGTRAPs at probe addresses are handled here;
// everything else is delivered to the tracee unchanged. Exit status of the
// command is propagated.
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#define OUT_CAP (1u << 22)
#define MAXMOD 16
#define MAXPROC 4096

typedef struct {
    char exe[PATH_MAX], img[PATH_MAX];
    uint64_t n, *va; const uint8_t *orig; unsigned id;
} module_t;
static module_t mods[MAXMOD]; static int nmods;
static uint64_t *out;

/* per thread-group state: which module (or -1) and load base */
typedef struct { pid_t tgid; int mod; uint64_t base; int resolved; int memfd; } proc_t;
/* tid -> tgid cache (tids are < 2^22 by default) */
#define TIDMAX (1 << 22)
static pid_t *tgid_of;
static proc_t procs[MAXPROC]; static int nprocs;

static proc_t *proc_of(pid_t tid) {
    pid_t tgid = (tid > 0 && tid < TIDMAX) ? tgid_of[tid] : 0;
    if (!tgid) {
        char p[64], buf[256]; snprintf(p, sizeof p, "/proc/%d/status", tid);
        tgid = tid;
        FILE *f = fopen(p, "r");
        if (f) { while (fgets(buf, sizeof buf, f)) if (!strncmp(buf, "Tgid:", 5)) { tgid = atoi(buf + 5); break; } fclose(f); }
        if (tid > 0 && tid < TIDMAX) tgid_of[tid] = tgid;
    }
    for (int i = 0; i < nprocs; i++) if (procs[i].tgid == tgid) return &procs[i];
    if (nprocs == MAXPROC) nprocs = 0;                  /* recycle; good enough here */
    procs[nprocs] = (proc_t){tgid, -1, 0, 0, -1};
    return &procs[nprocs++];
}
static void forget(pid_t tid) {          /* tid exited: it may be reused by another process */
    if (tid > 0 && tid < TIDMAX) tgid_of[tid] = 0;
    for (int i = 0; i < nprocs; i++) if (procs[i].tgid == tid) { if (procs[i].memfd >= 0) close(procs[i].memfd); procs[i] = procs[--nprocs]; break; }
}

static long find(module_t *m, uint64_t va) {
    long lo = 0, hi = (long)m->n - 1;
    while (lo <= hi) { long mid = (lo + hi) >> 1;
        if (m->va[mid] == va) return mid; if (m->va[mid] < va) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

static void load_config(const char *cfg) {
    FILE *f = fopen(cfg, "r"); if (!f) { perror(cfg); exit(2); }
    char line[3 * PATH_MAX + 64], exe[PATH_MAX], tab[PATH_MAX], img[PATH_MAX]; unsigned id;
    while (nmods < MAXMOD && fgets(line, sizeof line, f)) {
        if (sscanf(line, "%4095s %4095s %4095s %u", exe, tab, img, &id) != 4) continue;
        module_t *m = &mods[nmods];
        if (!realpath(exe, m->exe)) continue;
        strcpy(m->img, img); m->id = id;
        int fd = open(tab, O_RDONLY); struct stat st;
        if (fd < 0 || fstat(fd, &st)) continue;
        uint8_t *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        m->n = *(uint64_t *)p; m->va = (uint64_t *)(p + 8); m->orig = p + 8 + 8 * m->n;
        nmods++;
    }
    fclose(f);
}

/* Work out which module (if any) this process runs and where it is loaded;
   with arm=1 (just exec'd) also write the pre-armed image over its text.
   A process forked from an armed one is resolved lazily on its first trap:
   it inherited the armed text, and runs some of it before it execs. */
static void resolve(pid_t pid, proc_t *pr, int arm) {
    pr->mod = -1; pr->resolved = 1;
    if (pr->memfd >= 0) { close(pr->memfd); pr->memfd = -1; }   /* exec: new address space */
    char p[64], self[PATH_MAX]; snprintf(p, sizeof p, "/proc/%d/exe", pid);
    ssize_t n = readlink(p, self, sizeof self - 1); if (n <= 0) return; self[n] = 0;
    int mi = -1;
    for (int i = 0; i < nmods; i++) if (!strcmp(mods[i].exe, self)) { mi = i; break; }
    if (mi < 0 || !strcmp(mods[mi].img, "-")) return;
    /* load base: first mapping of the exe in /proc/pid/maps minus its first PT_LOAD vaddr */
    int ifd = open(mods[mi].img, O_RDONLY); if (ifd < 0) return;
    struct stat st; fstat(ifd, &st);
    uint8_t *img = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, ifd, 0); close(ifd);
    if (img == MAP_FAILED) return;
    Elf64_Ehdr *eh = (Elf64_Ehdr *)img; Elf64_Phdr *ph = (Elf64_Phdr *)(img + eh->e_phoff);
    uint64_t base = 0;
    if (eh->e_type == ET_DYN) {
        snprintf(p, sizeof p, "/proc/%d/maps", pid);
        FILE *mf = fopen(p, "r"); char line[PATH_MAX + 128]; uint64_t lo = UINT64_MAX;
        while (mf && fgets(line, sizeof line, mf)) {
            uint64_t a, b, off; char perms[8]; char path[PATH_MAX] = "";
            if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %4095s", &a, &b, perms, &off, path) >= 4 &&
                !strcmp(path, self) && off == 0 && a < lo) lo = a;
        }
        if (mf) fclose(mf);
        uint64_t first = UINT64_MAX;
        for (int k = 0; k < eh->e_phnum; k++) if (ph[k].p_type == PT_LOAD && ph[k].p_vaddr < first) first = ph[k].p_vaddr & ~0xfffULL;
        if (lo == UINT64_MAX) { munmap(img, st.st_size); return; }
        base = lo - first;
    }
    if (!arm) { pr->mod = mi; pr->base = base; munmap(img, st.st_size); return; }
    snprintf(p, sizeof p, "/proc/%d/mem", pid);
    int memfd = open(p, O_RDWR);
    if (memfd >= 0) {
        for (int k = 0; k < eh->e_phnum; k++) {
            if (ph[k].p_type != PT_LOAD || !(ph[k].p_flags & PF_X)) continue;
            pwrite(memfd, img + ph[k].p_offset, ph[k].p_filesz, base + ph[k].p_vaddr);
        }
        close(memfd);
        pr->mod = mi; pr->base = base;
    }
    munmap(img, st.st_size);
}

/* returns 1 if the trap was one of our probes (handled) */
static int on_trap(pid_t tid) {
    proc_t *pr = proc_of(tid);
    if (!pr->resolved) resolve(tid, pr, 0);
    if (pr->mod < 0) return 0;
    struct user_regs_struct r;
    if (ptrace(PTRACE_GETREGS, tid, 0, &r)) return 0;
    uint64_t pc = r.rip - 1;
    module_t *m = &mods[pr->mod];
    long i = find(m, pc - pr->base);
    if (i < 0) return 0;
    if (pr->memfd < 0) {
        char p[64]; snprintf(p, sizeof p, "/proc/%d/mem", pr->tgid);
        pr->memfd = open(p, O_RDWR);
    }
    uint8_t cur = 0;
    if (pr->memfd >= 0 && pread(pr->memfd, &cur, 1, pc) == 1 && cur == 0xCC) {
        if (pwrite(pr->memfd, &m->orig[i], 1, pc) == 1) {
            uint64_t k = out[0]++;
            if (k < OUT_CAP) out[1 + k] = ((uint64_t)m->id << 56) | (pc - pr->base);
        }
    }
    r.rip = pc;
    ptrace(PTRACE_SETREGS, tid, 0, &r);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[3], "--")) { fprintf(stderr, "usage: bptrace <config> <out> -- cmd...\n"); return 2; }
    load_config(argv[1]);
    tgid_of = calloc(TIDMAX, sizeof(pid_t));
    int ofd = open(argv[2], O_RDWR | O_CREAT, 0644);
    size_t len = 8 * (1 + (size_t)OUT_CAP);
    if (ofd < 0 || ftruncate(ofd, len)) { perror(argv[2]); return 2; }
    out = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, ofd, 0); close(ofd);
    pid_t child = fork();
    if (child == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        raise(SIGSTOP);
        execvp(argv[4], argv + 4);
        _exit(127);
    }
    int st;
    waitpid(child, &st, 0);
    ptrace(PTRACE_SETOPTIONS, child, 0, PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK | PTRACE_O_TRACECLONE |
           PTRACE_O_TRACEEXEC | PTRACE_O_EXITKILL);
    ptrace(PTRACE_CONT, child, 0, 0);
    int exit_code = 0;
    for (;;) {
        pid_t tid = waitpid(-1, &st, __WALL);
        if (tid < 0) { if (errno == EINTR) continue; break; }
        if (WIFEXITED(st) || WIFSIGNALED(st)) {
            if (tid == child) exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
            forget(tid);
            continue;
        }
        if (!WIFSTOPPED(st)) continue;
        int sig = WSTOPSIG(st), ev = st >> 16, deliver = 0;
        if (ev == PTRACE_EVENT_EXEC) { if (tid > 0 && tid < TIDMAX) tgid_of[tid] = 0; resolve(tid, proc_of(tid), 1); }
        else if (ev) { /* fork/vfork/clone: the new task starts stopped and is auto-attached */ }
        else if (sig == SIGTRAP) deliver = on_trap(tid) ? 0 : SIGTRAP;
        else if (sig == SIGSTOP) {
            siginfo_t si;   /* group-stop vs new-task initial stop: swallow the latter */
            deliver = (ptrace(PTRACE_GETSIGINFO, tid, 0, &si) == 0 && si.si_code == SI_USER && si.si_pid != getpid()) ? SIGSTOP : 0;
        }
        else deliver = sig;
        ptrace(PTRACE_CONT, tid, 0, deliver);
    }
    return exit_code;
}
