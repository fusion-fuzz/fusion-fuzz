"""
Shared utilities for fusion-fuzz.
"""

import re

# Patterns that mark the beginning of a crash / sanitiser report.
_CRASH_ANCHORS = [
    "SUMMARY: AddressSanitizer",
    "SUMMARY: UndefinedBehaviorSanitizer",
    "SUMMARY: MemorySanitizer",
    "SUMMARY: ThreadSanitizer",
    "ERROR: AddressSanitizer",
    "ERROR: MemorySanitizer",
    "runtime error:",            # UBSan inline
    "Assertion:",                # PHP / C assert()
    "Fatal error:",              # PHP fatal
    "Segmentation fault",
    "Bus error",
    "(core dumped)",
    "zend_mm_heap corrupted",    # PHP heap
    # Anchors the list was missing, each the first line of a whole
    # project's crash reports. Without them those bundles were saved with
    # only the *tail* of the output — hundreds of stack frames and no line
    # saying what failed. tools/oracle_selftest.py --sweep found 14 such
    # bundles (rust 5, swift 8, gcc 1): replaying their saved output
    # through their own oracle reported nothing, because the line the
    # oracle looks for had been truncated away.
    "internal compiler error",       # gcc, rustc, lfortran
    "error: internal compiler error",
    "thread 'rustc' panicked",
    "the compiler unexpectedly panicked",
    "LLVM ERROR",
    "Assertion failed:",             # swift, LLVM
    "Assertion Failed",              # ruby (RUBY_ASSERT)
    "Please submit a bug report",
    "While evaluating request",      # swift request evaluator
    "While emitting IR",
    "While type-checking",
    "Stack dump:",
    "PLEASE submit a bug report",
    "[BUG]",                         # ruby rb_bug
    "panic:",                        # Go
    "fatal error:",                  # Go runtime
    "caught segfault",               # R
    "UNREACHABLE executed",          # LLVM/flang
    "AssertFailed",                  # lfortran
    "LCompilersException",
    "Debug failure",                 # tsc
]

_LINES_BEFORE_ANCHOR = 25
_LINES_AFTER_ANCHOR  = 300
_MAX_OUTPUT_CHARS    = 48_000


# One alternation over the anchors, longest first so a longer anchor
# that contains a shorter one ("AddressSanitizer: heap-use-after-free"
# vs "AddressSanitizer") still wins the same way the old first-match
# loop did: the loop tested anchors in list order per line, and the
# earliest *line* won; within a line the regex returns the leftmost
# match, which is what the line-then-anchor loop returned too.
_CRASH_ANCHOR_RE = re.compile(
    "|".join(re.escape(a) for a in sorted(_CRASH_ANCHORS, key=len, reverse=True)))


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def _line_offset(text: str, pos: int, n: int) -> int:
    """Offset of the start of the line *n* lines after the one containing
    *pos* (n >= 0); len(text) if there are fewer."""
    for _ in range(n):
        nxt = text.find("\n", pos)
        if nxt == -1:
            return len(text)
        pos = nxt + 1
    return pos


def smart_truncate(text: str,
                   max_chars: int = _MAX_OUTPUT_CHARS,
                   lines_before: int = _LINES_BEFORE_ANCHOR,
                   lines_after: int = _LINES_AFTER_ANCHOR) -> str:
    """
    Truncate *text* to at most *max_chars* characters, keeping content around
    the first crash signature rather than naively cutting from the front or end.

    Strategy
    --------
    1. If the text fits within *max_chars*, return it unchanged.
    2. Find the first line matching a known crash anchor.
    3. Keep *lines_before* lines of context before the anchor and
       *lines_after* lines after it.
    4. Replace skipped sections with a concise marker line.
    5. If no anchor is found, keep the head and the tail.

    Cost is proportional to the part that is *kept*, not to the whole
    output: the previous version split a multi-megabyte sanitizer dump
    into lines and tested 34 anchors against every one of them, on the
    orchestrator's main thread, at 30-140 ms a call — 10 s of a 150 s
    php run, serialised behind every result.
    """
    if len(text) <= max_chars:
        return text

    m = _CRASH_ANCHOR_RE.search(text)
    if m is None:
        # No anchor: keep the head as well as the tail. Whatever the
        # program printed first is the best remaining evidence of what
        # went wrong — a compiler writes its diagnosis before its stack
        # trace — and keeping only the tail threw exactly that away.
        head_lines = max(lines_before, 40)
        body = text[:-1] if text.endswith("\n") else text
        total = body.count("\n") + 1 if body else 0
        skipped = total - head_lines - lines_after
        if skipped <= 0:
            return text
        head_end = _line_offset(body, 0, head_lines)          # after head's newline
        # splitlines() counts a trailing empty line ("...x\n\n" ends in
        # an empty line), so the walk back starts *on* it in that case.
        tail_start = len(body)
        steps = lines_after - 1 if body.endswith("\n") else lines_after
        for _ in range(steps):
            tail_start = _line_start(body, tail_start - 1)
        head, tail = body[:head_end - 1], body[tail_start:]
        skipped_bytes = len(text) - (len(head) + 1) - (len(tail) + 1)
        marker = (f"[... {skipped:,} lines / {skipped_bytes:,} bytes of output "
                  f"truncated — no crash signature found; head and tail kept ...]")
        return head + "\n" + marker + "\n" + tail

    anchor_line_start = _line_start(text, m.start())
    # Walk back lines_before line starts (or to the beginning).
    start = anchor_line_start
    for _ in range(lines_before):
        if start == 0:
            break
        start = _line_start(text, start - 1)
    end = _line_offset(text, anchor_line_start, lines_after)
    kept = text[start:end]
    if kept.endswith("\n"):
        kept = kept[:-1]

    parts = []
    if start > 0:
        parts.append(f"[... {text.count(chr(10), 0, start):,} lines / {start:,} bytes of output "
                     f"truncated before crash signature ...]")
    parts.append(kept)
    if end < len(text):
        remaining = text.count("\n", end) + (0 if text.endswith("\n") else 1)
        parts.append(f"[... {remaining:,} more lines truncated after crash report ...]")
    return "\n".join(parts)
