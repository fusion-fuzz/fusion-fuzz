"""
Shared utilities for fusion-fuzz.
"""

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
]

_LINES_BEFORE_ANCHOR = 25
_LINES_AFTER_ANCHOR  = 300
_MAX_OUTPUT_CHARS    = 48_000


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
    5. If no anchor is found, keep the tail (*lines_after* lines).
    """
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()

    anchor_idx = None
    for i, line in enumerate(lines):
        for pat in _CRASH_ANCHORS:
            if pat in line:
                anchor_idx = i
                break
        if anchor_idx is not None:
            break

    if anchor_idx is None:
        keep    = lines[-lines_after:]
        skipped = len(lines) - len(keep)
        if skipped <= 0:
            return text
        skipped_bytes = len(text) - sum(len(l) + 1 for l in keep)
        marker = (f"[... {skipped:,} lines / {skipped_bytes:,} bytes of output "
                  f"truncated — no crash signature found; showing tail ...]")
        return marker + "\n" + "\n".join(keep)

    start = max(0, anchor_idx - lines_before)
    end   = min(len(lines), anchor_idx + lines_after)
    parts = []

    if start > 0:
        skipped_bytes = sum(len(l) + 1 for l in lines[:start])
        parts.append(f"[... {start:,} lines / {skipped_bytes:,} bytes of output "
                     f"truncated before crash signature ...]")

    parts.append("\n".join(lines[start:end]))

    if end < len(lines):
        remaining = len(lines) - end
        parts.append(f"[... {remaining:,} more lines truncated after crash report ...]")

    return "\n".join(parts)
