// LD_PRELOAD runtime for breakpoint ("first-hit") block coverage.
// The binaries on disk are never modified: in this process's private
// copy-on-write text pages, int3 is written over every not-yet-covered
// block leader of each instrumented module (the main executable and/or
// shared libraries). On a hit the block is recorded, the byte restored and
// execution resumed. Modules already loaded are armed at startup (after
// the dynamic linker has finished, so IFUNC resolvers etc. already ran);
// libraries loaded later are armed when dlopen() returns.
//
// env: BPCOV_CONFIG lines "<path> <table> <image|-> <module_id>": every loaded
//                   object (executable or library) whose realpath is <path>
//                   is instrumented. Records are (module_id << 56) | vaddr,
//                   so all processes and modules can share one OUT.
//      BPCOV_OUT    output file, shared mmap: u64 count, u64 rec[]
//      BPCOV_TABLE/BPCOV_EXE/BPCOV_IMAGE: single-module form of the config
//                   (module id 0)
//      table (make_table.py): u64 n, u64 va[n], u8 orig[n]
//      image (make_table.py): pre-armed copy of the module; its executable
//                   segments are mmap'ed over the original, so arming costs
//                   one mmap and untouched pages cost nothing
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <link.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

#define OUT_CAP (1u << 22)
#define MAX_CFG 32
#define MAX_MODS 32

struct cfg { char path[PATH_MAX], tab[PATH_MAX], img[PATH_MAX]; unsigned id; };
static struct cfg cfgs[MAX_CFG];
static int n_cfg;

struct mod {
    uintptr_t base, lo, hi;            /* load bias; [lo,hi) = executable segments */
    uint64_t n_probes, *probe_va, tag;
    const uint8_t *probe_orig;
    uint8_t *armed;                    /* in-memory arming only: 1 if int3 was written */
    int image_mode;
    const ElfW(Phdr) *phdr; int phnum;
    char path[PATH_MAX];
};
static struct mod mods[MAX_MODS];
static char self_exe[PATH_MAX];
static volatile int n_mods;            /* published after a module is fully set up */

static volatile uint64_t *out;
static struct sigaction app_trap;      /* the application's own SIGTRAP disposition */
static int active;
static int debug;
static int arm_lock;

static int raw_mprotect(void *a, size_t l, int p) { return (int)syscall(SYS_mprotect, a, l, p); }

static void dbg_hex(const char *tag, uint64_t v) {
    char b[64]; int n = 0; const char *h = "0123456789abcdef";
    while (*tag) b[n++] = *tag++;
    for (int s = 60; s >= 0; s -= 4) b[n++] = h[(v >> s) & 15];
    b[n++] = '\n'; syscall(SYS_write, 2, b, n);
}
static long find(const struct mod *m, uint64_t va) {
    long lo = 0, hi = (long)m->n_probes - 1;
    while (lo <= hi) {
        long mid = (lo + hi) >> 1;
        if (m->probe_va[mid] == va) return mid;
        if (m->probe_va[mid] < va) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
static struct mod *mod_of(uintptr_t pc) {
    int n = __atomic_load_n(&n_mods, __ATOMIC_ACQUIRE);
    for (int k = 0; k < n; k++)
        if (pc >= mods[k].lo && pc < mods[k].hi) return &mods[k];
    return NULL;
}

static void forward(int sig, siginfo_t *si, void *ctx) {
    if (debug) {
        ucontext_t *uc = ctx; uint64_t pc = uc->uc_mcontext.gregs[REG_RIP] - 1;
        dbg_hex("[bpcov] foreign SIGTRAP pc=", pc);
        dbg_hex("[bpcov]   si_code=", (uint64_t)si->si_code); dbg_hex("[bpcov]   byte=", *(uint8_t *)pc);
    }
    if ((app_trap.sa_flags & SA_SIGINFO) && app_trap.sa_sigaction) { app_trap.sa_sigaction(sig, si, ctx); return; }
    if (app_trap.sa_handler == SIG_IGN) return;
    if (app_trap.sa_handler && app_trap.sa_handler != SIG_DFL) { app_trap.sa_handler(sig); return; }
    struct sigaction d; memset(&d, 0, sizeof d); d.sa_handler = SIG_DFL;
    syscall(SYS_rt_sigaction, SIGTRAP, &d, NULL, 8);
    raise(SIGTRAP);
}

static void on_trap(int sig, siginfo_t *si, void *ctx) {
    ucontext_t *uc = ctx;
    uint64_t pc = uc->uc_mcontext.gregs[REG_RIP] - 1;
    struct mod *m = (si->si_code == SI_KERNEL) ? mod_of(pc) : NULL;
    long i = m ? find(m, pc - m->base) : -1;
    if (i < 0 || (!m->image_mode && (!m->armed || !m->armed[i]))) { forward(sig, si, ctx); return; }
    volatile uint8_t *p = (uint8_t *)pc;
    if (*p == 0xCC) {                  /* not yet restored by another thread */
        uintptr_t page = pc & ~(uintptr_t)4095;
        /* The page is left writable: re-protecting it would race with another
           thread restoring a different probe on the same page (it would
           find the page read-only again by the time it writes). */
        raw_mprotect((void *)page, 4096, PROT_READ | PROT_WRITE | PROT_EXEC);
        *p = m->probe_orig[i];
        if (out) {
            uint64_t k = __atomic_fetch_add(&out[0], 1, __ATOMIC_RELAXED);
            if (k < OUT_CAP) out[1 + k] = m->tag | (pc - m->base);
        }
    }
    uc->uc_mcontext.gregs[REG_RIP] = pc;
}

/* Keep our handler installed; record what the application asks for. */
int sigaction(int sig, const struct sigaction *act, struct sigaction *old) {
    static int (*real)(int, const struct sigaction *, struct sigaction *);
    if (!real) real = dlsym(RTLD_NEXT, "sigaction");
    if (sig == SIGTRAP && active) {
        if (debug && act) dbg_hex("[bpcov] app sigaction(SIGTRAP) handler=", (uint64_t)(uintptr_t)act->sa_handler);
        if (old) *old = app_trap;
        if (act) app_trap = *act;
        return 0;
    }
    if (act && active) {          /* other handlers must not block SIGTRAP while they run */
        struct sigaction a2 = *act;
        sigdelset(&a2.sa_mask, SIGTRAP);
        return real(sig, &a2, old);
    }
    return real(sig, act, old);
}
/* A thread that blocks SIGTRAP would be killed outright by a probe hit
   (the kernel forces the default action for a blocked synchronous trap),
   so never let the application block it. */
static const sigset_t *keep_trap(const sigset_t *set, sigset_t *tmp) {
    if (!set || !active) return set;
    *tmp = *set; sigdelset(tmp, SIGTRAP); return tmp;
}
int pthread_sigmask(int how, const sigset_t *set, sigset_t *old) {
    static int (*real)(int, const sigset_t *, sigset_t *);
    if (!real) real = dlsym(RTLD_NEXT, "pthread_sigmask");
    sigset_t tmp;
    return real(how, how == SIG_UNBLOCK ? set : keep_trap(set, &tmp), old);
}
int sigprocmask(int how, const sigset_t *set, sigset_t *old) {
    static int (*real)(int, const sigset_t *, sigset_t *);
    if (!real) real = dlsym(RTLD_NEXT, "sigprocmask");
    sigset_t tmp;
    return real(how, how == SIG_UNBLOCK ? set : keep_trap(set, &tmp), old);
}
/* A new thread starts with its creator's mask *at clone time*, and some
   runtimes block every signal around thread creation through raw syscalls
   that the wrappers above cannot see (sanitizer_common's ScopedBlockSignals
   in ASan's pthread_create interceptor: Ruby's first thread died on its
   first probe). So every thread unblocks SIGTRAP before its start routine
   runs. Sanitizer-runtime code is never probed, so the ASan thread-start
   wrapper reached through here cannot trap before that. */
/* The caller may itself be running with every signal blocked, so nothing
   here may reach probed code: slots come from a static pool, not malloc
   (which is the sanitizer's, inside the probed executable). */
#define N_SLOTS 4096
struct thread_start { void *(*fn)(void *); void *arg; int busy; };
static struct thread_start slots[N_SLOTS];
static void *thread_trampoline(void *p) {
    struct thread_start *s = p;
    void *(*fn)(void *) = s->fn; void *arg = s->arg;
    __atomic_store_n(&s->busy, 0, __ATOMIC_RELEASE);
    sigset_t u; sigemptyset(&u); sigaddset(&u, SIGTRAP);
    syscall(SYS_rt_sigprocmask, SIG_UNBLOCK, &u, NULL, 8);
    return fn(arg);
}
int pthread_create(pthread_t *th, const pthread_attr_t *attr, void *(*fn)(void *), void *arg) {
    static int (*real)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    if (!real) real = dlsym(RTLD_NEXT, "pthread_create");
    struct thread_start *s = NULL;
    for (int k = 0; active && k < N_SLOTS && !s; k++)
        if (!__atomic_exchange_n(&slots[k].busy, 1, __ATOMIC_ACQUIRE)) s = &slots[k];
    /* The caller itself may have blocked everything (sanitizer_common's
       internal_start_thread does, through a raw syscall) and pthread_create
       reaches probed code: glibc allocates the new thread's TLS with the
       interposed sanitizer calloc. Unblock SIGTRAP for the call. */
    sigset_t cur, u; sigemptyset(&u); sigaddset(&u, SIGTRAP);
    int reblock = active && syscall(SYS_rt_sigprocmask, SIG_UNBLOCK, &u, &cur, 8) == 0 && sigismember(&cur, SIGTRAP);
    int r;
    if (!s) r = real(th, attr, fn, arg);
    else {
        s->fn = fn; s->arg = arg;
        r = real(th, attr, thread_trampoline, s);
        if (r) __atomic_store_n(&s->busy, 0, __ATOMIC_RELEASE);
    }
    if (reblock) syscall(SYS_rt_sigprocmask, SIG_BLOCK, &u, NULL, 8);
    return r;
}
sighandler_t signal(int sig, sighandler_t h) {
    struct sigaction a, o; memset(&a, 0, sizeof a); a.sa_handler = h; sigemptyset(&a.sa_mask);
    a.sa_flags = SA_RESTART;
    if (sigaction(sig, &a, &o) != 0) return SIG_ERR;
    return o.sa_handler;
}

/* Map the executable segments of `file` over module m's (the original to
   disarm, or the pre-armed image to arm). */
static int map_segments(const struct mod *m, const char *file) {
    long fd = syscall(SYS_open, file, O_RDONLY);
    if (fd < 0) return -1;
    int ok = 0;
    for (int k = 0; k < m->phnum; k++) {
        const ElfW(Phdr) *ph = &m->phdr[k];
        if (ph->p_type != PT_LOAD || !(ph->p_flags & PF_X)) continue;
        uintptr_t start = (m->base + ph->p_vaddr) & ~(uintptr_t)4095;
        uintptr_t end = (m->base + ph->p_vaddr + ph->p_filesz + 4095) & ~(uintptr_t)4095;
        off_t off = ph->p_offset & ~(off_t)4095;
        long r = syscall(SYS_mmap, (void *)start, end - start, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_FIXED, fd, off);
        if (r < 0 && r > -4096) { syscall(SYS_close, fd); return -1; }
        ok = 1;
    }
    syscall(SYS_close, fd);
    return ok ? 0 : -1;
}

/* At exit, put the original code back. Exit-time code can run with every
   signal blocked through raw syscalls (e.g. LeakSanitizer's StopTheWorld),
   where a probe hit would be fatal. Registered after the sanitizer runtime
   (which initialises before us), so it runs before LSan's atexit hook. */
static void disarm_at_exit(void) {
    int n = n_mods;
    for (int k = 0; k < n; k++)
        if (mods[k].image_mode) map_segments(&mods[k], mods[k].path);
    active = 0;
}

static void install_handler(void) {
    if (active) return;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_trap;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    int (*real)(int, const struct sigaction *, struct sigaction *) = dlsym(RTLD_NEXT, "sigaction");
    real(SIGTRAP, &sa, &app_trap);
    active = 1;
    { sigset_t u; sigemptyset(&u); sigaddset(&u, SIGTRAP); syscall(SYS_rt_sigprocmask, SIG_UNBLOCK, &u, NULL, 8); }
    atexit(disarm_at_exit);
}

/* Instrument one loaded object (realpath `path`) described by config c. */
static void arm_module(const struct cfg *c, const struct dl_phdr_info *info, const char *path) {
    if (n_mods >= MAX_MODS) return;
    int fd = open(c->tab, O_RDONLY);
    if (fd < 0) return;
    struct stat st; fstat(fd, &st);
    uint8_t *t = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (t == MAP_FAILED) return;
    struct mod *m = &mods[n_mods];
    memset(m, 0, sizeof *m);
    m->n_probes = *(uint64_t *)t;
    m->probe_va = (uint64_t *)(t + 8);
    m->probe_orig = t + 8 + 8 * m->n_probes;
    m->tag = (uint64_t)c->id << 56;
    m->base = info->dlpi_addr; m->phdr = info->dlpi_phdr; m->phnum = info->dlpi_phnum;
    m->lo = UINTPTR_MAX;
    for (int k = 0; k < m->phnum; k++) {
        const ElfW(Phdr) *ph = &m->phdr[k];
        if (ph->p_type != PT_LOAD || !(ph->p_flags & PF_X)) continue;
        uintptr_t a = m->base + ph->p_vaddr, b = a + ph->p_memsz;
        if (a < m->lo) m->lo = a;
        if (b > m->hi) m->hi = b;
    }
    strncpy(m->path, path, PATH_MAX - 1);
    install_handler();
    if (c->img[0] && strcmp(c->img, "-")) {
        m->image_mode = 1;               /* published before mapping: once mapped, any
                                            call into the module (e.g. ASan interceptors
                                            in an executable) may trap */
        __atomic_store_n(&n_mods, n_mods + 1, __ATOMIC_RELEASE);
        if (map_segments(m, c->img) == 0) return;
        m->image_mode = 0;
    } else {
        m->armed = calloc(m->n_probes, 1);
        __atomic_store_n(&n_mods, n_mods + 1, __ATOMIC_RELEASE);
    }
    if (!m->armed) m->armed = calloc(m->n_probes, 1);
    /* Fallback: arm in memory, one mprotect per window, verifying each byte. */
    uintptr_t lo_page = 0, hi_page = 0;
    for (uint64_t i = 0; i < m->n_probes; i++) {
        uintptr_t a = m->base + m->probe_va[i], pg = a & ~(uintptr_t)4095;
        if (pg >= hi_page) {
            if (hi_page) raw_mprotect((void *)lo_page, hi_page - lo_page, PROT_READ | PROT_EXEC);
            lo_page = pg; hi_page = pg + (1u << 20);          /* 1 MB windows */
            raw_mprotect((void *)lo_page, hi_page - lo_page, PROT_READ | PROT_WRITE | PROT_EXEC);
        }
        uint8_t *p = (uint8_t *)a;
        if (*p == m->probe_orig[i]) { *p = 0xCC; m->armed[i] = 1; }
    }
    if (hi_page) raw_mprotect((void *)lo_page, hi_page - lo_page, PROT_READ | PROT_EXEC);
}

static int scan_one(struct dl_phdr_info *info, size_t sz, void *data) {
    (void)sz; (void)data;
    char rp[PATH_MAX];
    const char *path;
    if (info->dlpi_name == NULL || info->dlpi_name[0] == 0) path = self_exe;
    else if (realpath(info->dlpi_name, rp)) path = rp;
    else return 0;
    if (!path[0]) return 0;
    for (int k = 0; k < n_mods; k++)              /* already instrumented */
        if (mods[k].base == info->dlpi_addr && strcmp(mods[k].path, path) == 0) return 0;
    for (int k = 0; k < n_cfg; k++)
        if (strcmp(cfgs[k].path, path) == 0) { arm_module(&cfgs[k], info, path); break; }
    return 0;
}
static void scan(void) {
    while (__atomic_exchange_n(&arm_lock, 1, __ATOMIC_ACQUIRE)) sched_yield();
    dl_iterate_phdr(scan_one, NULL);
    __atomic_store_n(&arm_lock, 0, __ATOMIC_RELEASE);
}

/* A bare library name is searched in the *calling* object's RPATH/RUNPATH,
   and glibc takes the caller from dlopen's return address, which is now in
   this wrapper. If the plain lookup fails, retry in the real caller's
   search path ($ORIGIN expanded) so wrapping does not change the result. */
static void *dlopen_in_caller_path(void *(*real)(const char *, int), const char *file, int mode,
                                   const void *ret) {
    struct link_map *lm = NULL; Dl_info di;
    if (!dladdr1(ret, &di, (void **)&lm, RTLD_DL_LINKMAP) || !lm || !lm->l_ld) return NULL;
    const char *strtab = NULL; const char *paths[2] = {0, 0}; ElfW(Addr) off[2] = {0, 0}; int has[2] = {0, 0};
    for (const ElfW(Dyn) *d = lm->l_ld; d->d_tag != DT_NULL; d++) {
        if (d->d_tag == DT_STRTAB) strtab = (const char *)d->d_un.d_ptr;
        else if (d->d_tag == DT_RUNPATH) { off[0] = d->d_un.d_val; has[0] = 1; }
        else if (d->d_tag == DT_RPATH) { off[1] = d->d_un.d_val; has[1] = 1; }
    }
    if (!strtab) return NULL;
    for (int k = 0; k < 2; k++) if (has[k]) paths[k] = strtab + off[k];
    char origin[PATH_MAX];
    const char *obj = (lm->l_name && lm->l_name[0]) ? lm->l_name : self_exe;
    strncpy(origin, obj, PATH_MAX - 1); origin[PATH_MAX - 1] = 0;
    char *slash = strrchr(origin, '/'); if (slash) *slash = 0; else strcpy(origin, ".");
    for (int k = 0; k < 2; k++) {
        const char *p = paths[k];
        while (p && *p) {
            const char *e = strchr(p, ':'); size_t len = e ? (size_t)(e - p) : strlen(p);
            char dir[PATH_MAX], cand[PATH_MAX]; size_t o = 0;
            for (size_t i = 0; i < len && o < PATH_MAX - 1; ) {
                if (!strncmp(p + i, "$ORIGIN", 7) || !strncmp(p + i, "${ORIGIN}", 9)) {
                    size_t sk = p[i + 1] == '{' ? 9 : 7;
                    for (const char *q = origin; *q && o < PATH_MAX - 1; q++) dir[o++] = *q;
                    i += sk;
                } else dir[o++] = p[i++];
            }
            dir[o] = 0;
            if (snprintf(cand, sizeof cand, "%s/%s", dir, file) < (int)sizeof cand && access(cand, F_OK) == 0) {
                void *h = real(cand, mode);
                if (h) return h;
            }
            p = e ? e + 1 : NULL;
        }
    }
    return NULL;
}

/* Libraries loaded after startup are armed when dlopen returns. */
void *dlopen(const char *file, int mode) {
    static void *(*real)(const char *, int);
    if (!real) real = dlsym(RTLD_NEXT, "dlopen");
    void *h = real(file, mode);
    if (!h && file && !strchr(file, '/')) {
        h = dlopen_in_caller_path(real, file, mode, __builtin_return_address(0));
        if (!h) h = real(file, mode);    /* restore dlerror() of the original attempt */
    }
    if (h && n_cfg) scan();
    return h;
}

static void open_out(const char *outp) {
    int ofd = open(outp, O_RDWR | O_CREAT, 0644);
    if (ofd < 0) return;
    size_t len = 8 * (1 + (size_t)OUT_CAP);
    if (ftruncate(ofd, len) == 0) {
        void *o = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, ofd, 0);
        if (o != MAP_FAILED) out = o;
    }
    close(ofd);
}

__attribute__((constructor(101))) static void bpcov_init(void) {
    const char *tab = getenv("BPCOV_TABLE"), *outp = getenv("BPCOV_OUT"), *exe = getenv("BPCOV_EXE");
    const char *img = getenv("BPCOV_IMAGE"), *cfgp = getenv("BPCOV_CONFIG");
    debug = getenv("BPCOV_DEBUG") != NULL;
    ssize_t n = readlink("/proc/self/exe", self_exe, sizeof self_exe - 1);
    if (n <= 0) return;
    self_exe[n] = 0;
    if (cfgp) {
        FILE *f = fopen(cfgp, "r");
        if (!f) return;
        char line[3 * PATH_MAX + 64], p[PATH_MAX];
        while (n_cfg < MAX_CFG && fgets(line, sizeof line, f)) {
            struct cfg *c = &cfgs[n_cfg];
            if (sscanf(line, "%4095s %4095s %4095s %u", p, c->tab, c->img, &c->id) != 4) continue;
            if (realpath(p, c->path)) n_cfg++;
        }
        fclose(f);
    } else if (tab && exe && realpath(exe, cfgs[0].path)) {
        strncpy(cfgs[0].tab, tab, PATH_MAX - 1);
        strncpy(cfgs[0].img, img ? img : "-", PATH_MAX - 1);
        n_cfg = 1;
    }
    if (!n_cfg) return;
    /* LD_PRELOAD reaches every descendant, including programs a compiler
       under test builds with -fsanitize=address and then runs (mojo run):
       a dynamically linked ASan runtime refuses to start unless it comes
       first in the library list. Children inherit this. */
    const char *ao = getenv("ASAN_OPTIONS");
    if (!ao || !strstr(ao, "verify_asan_link_order")) {
        char buf[4096];
        snprintf(buf, sizeof buf, "%s%sverify_asan_link_order=0", ao ? ao : "", ao && *ao ? ":" : "");
        setenv("ASAN_OPTIONS", buf, 1);
    }
    if (outp) open_out(outp);
    scan();
}
