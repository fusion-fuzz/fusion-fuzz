import logging
import uuid
import math
from .fusion import Seed

logger = logging.getLogger("FFL.Minimizer")

class DeltaDebugger:
    """
    Implements the DDmin algorithm to minimize crash-inducing inputs.
    Operates on a line-level granularity.
    """
    def __init__(self, driver):
        self.driver = driver

    @staticmethod
    def _sigs_match(expected_sig: str | None, actual_sig: str | None) -> bool:
        """
        Return True when *actual_sig* is "the same kind of crash" as *expected_sig*.

        Exact equality is always accepted.  Beyond that, two signatures are
        considered a match when they share any of the well-known crash-type
        keywords below, e.g. both contain "SUMMARY:" or both contain "Assertion:".
        This prevents false negatives when sanitiser output varies between runs
        (e.g. different leaked-byte counts, randomised addresses).
        """
        if not expected_sig or not actual_sig:
            return expected_sig == actual_sig
        if expected_sig == actual_sig:
            return True
        KEYWORDS = (
            "SUMMARY:",
            "Assertion:",
            "ASAN:",
            "LLVM ERROR:",
            "compiler panic:",
            "compiler fatal:",
            "internal compiler error:",
            "Fatal Python error:",
            "INTERNAL PANIC:",
            "Check failed:",
            "Fatal error:",
            "Segmentation fault",
            "Bus error",
            "SIGSEGV",
            "SIGABRT",
        )
        for kw in KEYWORDS:
            if kw in expected_sig and kw in actual_sig:
                return True
        return False

    def _test(self, lines, expected_sig):
        """
        Executes the partial content to see if it reproduces the crash.

        When *expected_sig* is set it acts as the sole oracle:
          1. Literal substring match anywhere in raw stdout+stderr — handles
             arbitrary user strings like "overflow" or "core dumped".
          2. Keyword-level match on the extracted signature as a fallback
             (handles normalised ASAN/assertion signatures where byte counts vary).
        When *expected_sig* is None any detected crash counts.
        """
        content = "".join(lines)
        test_id = f"min_{uuid.uuid4().hex[:8]}"
        seed = Seed(content=content, id=test_id)

        try:
            result = self.driver.execute(seed)

            if expected_sig:
                combined = result.stdout + result.stderr
                # 1. Direct substring match on raw output (works for any user string)
                if expected_sig in combined:
                    return True
                # 2. Keyword-level match on the extracted signature
                if result.crashed:
                    sig = result.signature or self.driver.extract_crash_signature(
                        result.stdout, result.stderr, result.return_code
                    )
                    return self._sigs_match(expected_sig, sig)
                return False

            return result.crashed

        except Exception as e:
            logger.debug(f"Minimization execution error: {e}")
            return False

    def minimize(self, content, expected_sig=None):
        """
        Minimizes the input content while preserving the crash signature.
        """
        lines = content.splitlines(keepends=True)
        if len(lines) <= 1:
            return content

        logger.info(f"Starting minimization on {len(lines)} lines...")
        
        # DDmin Algorithm
        granularity = 2
        
        while len(lines) >= 2:
            start_lines = lines
            subset_length = len(lines) // granularity
            some_complement_passed = False

            if subset_length == 0:
                subset_length = 1

            # Break into chunks
            chunks = []
            for i in range(0, len(lines), subset_length):
                chunks.append(lines[i:i + subset_length])
            
            # 1. Check Complements (removing one chunk at a time)
            # This logic tries to remove parts of the code.
            for i in range(len(chunks)):
                complement = []
                for j in range(len(chunks)):
                    if i != j:
                        complement.extend(chunks[j])
                
                if self._test(complement, expected_sig):
                    lines = complement
                    granularity = max(granularity - 1, 2)
                    some_complement_passed = True
                    logger.debug(f"Reduced to {len(lines)} lines (complement).")
                    break
            
            if some_complement_passed:
                continue

            if granularity == len(lines):
                break

            # 2. Check Subsets (keeping only one chunk)
            # This logic tries to find a single small chunk that crashes alone.
            some_subset_passed = False
            for chunk in chunks:
                if self._test(chunk, expected_sig):
                    lines = chunk
                    granularity = 2
                    some_subset_passed = True
                    logger.debug(f"Reduced to {len(lines)} lines (subset).")
                    break
            
            if some_subset_passed:
                continue
                
            # 3. Increase Granularity (split into smaller chunks)
            if granularity < len(lines):
                granularity = min(len(lines), 2 * granularity)
                # logger.debug(f"Increased granularity to {granularity}")
            else:
                break
                
        logger.info(f"Minimization complete. Reduced to {len(lines)} lines.")
        return "".join(lines)
