import random
from random import randint, choice
import re

# ==========================================
# Mutators
# ==========================================

class BaseMutator:
    """
    Base Mutator class containing language-agnostic mutation rules.
    Can be extended for specific languages (e.g. PHPMutator, SQLMutator).
    """
    def mutate(self, content: str) -> str:
        content = self._mr_arith_operators(content)
        content = self._mr_assign_operators(content)
        content = self._mr_logical_operators(content)
        content = self._mr_integer(content)
        content = self._mr_string(content)
        return content

    def _mr_arith_operators(self, code):
        """Randomly mutate arithmetic operators (+, -, *, /, %, **)."""
        if random.random() > 0.001:
            return code
        target_regex = r'\+\+|[-*/%]|\*\*'
        replacements = ['+', '-', '*', '/', '%', '**']
        victims = re.findall(target_regex, code)
        if len(victims) == 0:
            return code
        code = code.replace(choice(victims), choice(replacements), 1)
        return code

    def _mr_assign_operators(self, code):
        """Randomly mutate assignment operators (+=, -=, *=, /=, %=)."""
        if random.random() > 0.001:
            return code
        target_regex = r'\+=|-=|\*=|/=|%='
        replacements = ['+=', '-=', '*=', '/=', '%=']
        victims = re.findall(target_regex, code)
        if len(victims) == 0:
            return code
        victim = choice(victims)
        replace = choice([op for op in replacements if op != victim])
        code = re.sub(re.escape(victim), replace, code, 1)
        return code

    def _mr_logical_operators(self, code):
        """Randomly mutate logical operators (and, or, xor, &&, ||)."""
        if random.random() > 0.001:
            return code
        # Includes 'and'/'or' which are common in PHP/Python/SQL
        target_regex = r'\band\b|\bor\b|\bxor\b|&&|\|\|'
        replacements = ['and', 'or', 'xor', '&&', '||']
        victims = re.findall(target_regex, code)
        if len(victims) == 0:
            return code
        victim = choice(victims)
        replace = choice([op for op in replacements if op != victim])
        code = re.sub(re.escape(victim), replace, code, 1)
        return code

    def _mr_integer(self, code):
        """
        Generic integer mutation using standard boundary values.
        Subclasses should override this for language-specific constants.
        """
        if random.random() > 0.001:
            return code
        target_regex = r'(?<![a-zA-Z0-9_])(?:0x[0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*|0)(?![a-zA-Z0-9_])'
        replacements = ['-1', '0', '1', '-2147483648', '2147483647'] # Standard 32-bit limits
        victims = re.findall(target_regex, code)
        if len(victims) == 0:
            return code
        victim = choice(victims)
        replace = choice(replacements)
        code = re.sub(re.escape(victim), replace, code, 1)
        return code

    def _mr_string(self, code):
        """Randomly mutate string literals."""
        if random.random() > 0.01:
            return code
        target_regex = r"'([^'\\]+(\\.[^'\\]*)*)'|\"([^\"\\]+(\\.[^\"\\]*)*)\""
        # Generic string replacements
        replacements = [f"'{chr(randint(0, 255))}'", "''", "'test\\0test'"] 
        victims = re.findall(target_regex, code)
        # Flatten tuple results from findall
        victims = [match[0] if match[0] else match[2] for match in victims]
        if len(victims) == 0:
            return code
        victim = choice(victims)
        replace = choice(replacements)
        code = re.sub(re.escape(victim), replace, code, 1)
        return code

class PHPMutator(BaseMutator):
    """
    PHP-Specific Mutator.
    Inherits generic mutations and adds/overrides PHP-specific rules.
    """

    PHP_SPECIAL_INTS = [
        '-1', '0', '1', '2',
        'PHP_INT_MAX', 'PHP_INT_MIN',
        'PHP_FLOAT_MIN', 'PHP_FLOAT_MAX', 'PHP_FLOAT_EPSILON',
        'NULL', 'NAN', 'INF', '-INF',
        '0x7fffffff', '0x80000000', '0xffffffff',
        '2147483647', '-2147483648',
        '9223372036854775807', '-9223372036854775808',
    ]

    # Strings that trigger PHP type-juggling edge cases
    PHP_SPECIAL_STRINGS = [
        '""', "''",
        '"0"', '"1"', '"-1"',
        '"false"', '"true"', '"null"', '"NULL"',
        '"0.0"', '"1.0"', '"-0"',
        '" "', '"\\0"', '"\\x00"',
        '"0x1"', '"0b1"', '"0777"',
        '"1e100"', '"-1e100"',
        '"2147483648"', '"-2147483649"',
        '"PHP_INT_MAX"', '"Array"',
    ]

    # Loose↔strict comparison swaps to stress type juggling
    _CMP_SWAPS = {'===': '==', '!==': '!=', '==': '===', '!=': '!=='}
    _CMP_PATTERN = re.compile(r'===|!==|==|!=')

    # PHP type cast operators
    _CASTS = ['(int)', '(integer)', '(string)', '(float)', '(double)', '(bool)', '(boolean)', '(array)', '(object)', '(unset)']
    _CAST_PATTERN = re.compile(r'\((int|integer|string|float|double|bool|boolean|array|object|unset)\)')

    def extract_sec(self, test, section):
        if section not in test:
            return ""
        start_idx = test.find(section) + len(section)
        x = re.search("--([_A-Z]+)--", test[start_idx:])
        end_idx = x.start() if x != None else len(test) - 1
        ret = test[start_idx:start_idx + end_idx].strip("\n")
        return ret

    def _mr_integer(self, phpcode):
        """Override with PHP-specific boundary constants."""
        if random.random() > 0.002:
            return phpcode
        target_regex = r'(?<![a-zA-Z0-9_])(?:0x[0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*|0)(?![a-zA-Z0-9_])'
        victims = re.findall(target_regex, phpcode)
        if not victims:
            return phpcode
        victim = choice(victims)
        phpcode = re.sub(re.escape(victim), choice(self.PHP_SPECIAL_INTS), phpcode, 1)
        return phpcode

    def _mr_string(self, phpcode):
        """Override with PHP type-juggling string values."""
        if random.random() > 0.01:
            return phpcode
        target_regex = r"'([^'\\]*(\\.[^'\\]*)*)'|\"([^\"\\]*(\\.[^\"\\]*)*)\""
        victims = re.findall(target_regex, phpcode)
        victims = [m[0] if m[0] else m[2] for m in victims]
        if not victims:
            return phpcode
        victim = choice(victims)
        replace = choice(self.PHP_SPECIAL_STRINGS)
        phpcode = re.sub(re.escape(victim), lambda _: replace, phpcode, count=1)
        return phpcode

    def _mr_variable(self, phpcode):
        """Cross-assign PHP variables to expose type confusion across call sites."""
        if random.random() > 0.005:
            return phpcode
        target_regex = r'\$\w+'
        variables = re.findall(target_regex, phpcode)
        if len(variables) == 0:
            return phpcode
        victim = choice(variables)
        replace = choice(variables)
        occurrences = [m.start() for m in re.finditer(re.escape(victim), phpcode)]
        if not occurrences:
            return phpcode

        num_replacements = choice(range(1, len(occurrences) + 1))
        selected_replacements = set(choice(occurrences) for _ in range(num_replacements))

        result = []
        last_index = 0
        for i, char in enumerate(phpcode):
            if i in selected_replacements:
                result.append(phpcode[last_index:i])
                result.append(replace)
                last_index = i + len(victim)
        result.append(phpcode[last_index:])
        return ''.join(result)

    def _mr_comparison(self, phpcode):
        """Swap loose/strict comparison operators to trigger type-juggling paths."""
        if random.random() > 0.003:
            return phpcode
        matches = list(self._CMP_PATTERN.finditer(phpcode))
        if not matches:
            return phpcode
        m = choice(matches)
        start, end = m.span()
        phpcode = phpcode[:start] + self._CMP_SWAPS[m.group(0)] + phpcode[end:]
        return phpcode

    def _mr_bool_null(self, phpcode):
        """Flip PHP boolean/null literals including uppercase variants."""
        if random.random() > 0.005:
            return phpcode
        pool = ['true', 'false', 'TRUE', 'FALSE', 'null', 'NULL', 'True', 'False']
        matches = list(re.finditer(r'\b(true|false|TRUE|FALSE|True|False|null|NULL)\b', phpcode))
        if not matches:
            return phpcode
        m = choice(matches)
        original = m.group(0)
        replacement = choice([v for v in pool if v != original])
        phpcode = phpcode[:m.start()] + replacement + phpcode[m.end():]
        return phpcode

    def _mr_type_cast(self, phpcode):
        """Swap existing type casts or inject one before a variable."""
        if random.random() > 0.003:
            return phpcode
        matches = list(self._CAST_PATTERN.finditer(phpcode))
        if matches:
            m = choice(matches)
            phpcode = phpcode[:m.start()] + choice(self._CASTS) + phpcode[m.end():]
        else:
            var_matches = list(re.finditer(r'\$\w+', phpcode))
            if var_matches:
                m = choice(var_matches)
                phpcode = phpcode[:m.start()] + choice(self._CASTS) + phpcode[m.start():]
        return phpcode

    def _mr_null_coalesce(self, phpcode):
        """Swap ?? (null coalesce) with ?: (Elvis) or strip the fallback entirely."""
        if random.random() > 0.003:
            return phpcode
        if '??' in phpcode:
            phpcode = phpcode.replace('??', choice(['?:', '||']), 1)
        elif '?:' in phpcode:
            phpcode = phpcode.replace('?:', '??', 1)
        return phpcode

    def _mr_spaceship(self, phpcode):
        """Replace a comparison operator with the spaceship operator <=> or vice versa."""
        if random.random() > 0.002:
            return phpcode
        if '<=>' in phpcode:
            phpcode = phpcode.replace('<=>', choice(['<', '>', '==']), 1)
        else:
            cmp_matches = list(re.finditer(r'[<>]=?|==', phpcode))
            if cmp_matches:
                m = choice(cmp_matches)
                phpcode = phpcode[:m.start()] + '<=>' + phpcode[m.end():]
        return phpcode

    def mutate(self, phpcode: str) -> str:
        phpcode = super().mutate(phpcode)
        phpcode = self._mr_variable(phpcode)
        phpcode = self._mr_comparison(phpcode)
        phpcode = self._mr_bool_null(phpcode)
        phpcode = self._mr_type_cast(phpcode)
        phpcode = self._mr_null_coalesce(phpcode)
        phpcode = self._mr_spaceship(phpcode)
        return phpcode

class CPythonMutator(BaseMutator):
    """
    CPython-Specific Mutator.
    Targets C-level boundary conditions, object model internals, and byte handling.
    """
    
    # ---- Fuzzing-oriented special values -------------------------------------------------
    SPECIAL_INTS = [
        "-0", "0", "1", "-1", "+1",
        "127", "128", "255", "256", "511", "512", "1023", "1024", "4095", "4096",
        "2**15-1", "-(2**15)", "2**31-1", "-(2**31)", "2**63-1", "-(2**63)", "2**64-1", "-(2**64)",
        "10**100", "10**1000",
        "0b0", "0b1", "-0b1", "0o777",
        "0x7fffffff", "0x80000000", "0xffffffff", "0xffffffffffffffff",
        "1_000_000", "-1_000_000",
        "999999999999", "-999999999999",
        "sys.maxsize", "-sys.maxsize - 1"
    ]

    SPECIAL_FLOATS = [
        "float('inf')", "-float('inf')", "float('nan')",
        "0.0", "-0.0",
        "1e308", "1e-308",
        "1.7976931348623157e308", "2.2250738585072014e-308",
        "5e-324", "1e-324",
        "1e309",
        "float.fromhex('0x0.0000000000001p-1022')",
        "float.fromhex('0x1.fffffffffffffp+1023')",
        "3.1415926535897932384626", "2.718281828459045"
    ]

    SPECIAL_STRINGS = [
        "''", '""', "' '",
        "'\\n\\r\\t'", "r'\\n\\r\\t'",
        "'\\x00'", "'\\x1f'",
        "'\\ufeff'", "'\\u200b'", "'\\u200e'",
        "'\\u2603'", "'\\U0001F4A9'",
        "'e\\u0301'",
        "r'\\ud800'", "r'\\udfff'", "r'\\udcff'",
        "'A'*1000", "'{}'*50", "'%s%s%s'",
        "'3.1415926535897932384626'"
    ]

    SPECIAL_BYTES = [
        "b''", "b'\\x00'", "b'\\xff'", "b'\\xff'*64",
        "b'\\x00\\xff\\x80\\x7f'",
        "b'\\xc0\\xaf'", "b'\\xed\\xa0\\x80'", "b'\\xf4\\x90\\x80\\x80'",
        "b'\\xe2\\x28\\xa1'", "b'\\xa0\\xa1'",
        "b'\\xe2\\x98\\x83'",
        "b'\\x00'*1024",
        "bytes(range(256))"
    ]

    SPECIAL_CONSTS = ["None", "True", "False", "Ellipsis", "NotImplemented"]

    ASSIGN_OPS = [
        "=", "+=", "-=", "*=", "/=", "//=", "%=", "&=", "|=", "^=", "<<=", ">>=", "**=", "@="
    ]

    OPS_MUTABLE = ["+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "<<", ">>", "@"]

    def _mr_arith_operators(self, code):
        """Override to support Python specific operators (//, @, **, bitwise)."""
        if random.random() > 0.001:
            return code
        
        # Build regex from the mutable operators list
        # Escape special characters for regex
        ops_escaped = [re.escape(op) for op in self.OPS_MUTABLE]
        # Sort by length descending to match longest operators first (** before *)
        ops_escaped.sort(key=len, reverse=True)
        target_regex = '|'.join(ops_escaped)
        
        victims = re.findall(target_regex, code)
        if not victims:
            return code
            
        victim = choice(victims)
        # Pick replacement distinct from victim
        replacements = [op for op in self.OPS_MUTABLE if op != victim]
        replace = choice(replacements)
        
        # Replace one occurrence
        code = code.replace(victim, replace, 1)
        return code

    def _mr_assign_operators(self, code):
        """Override to support Python specific assignment operators."""
        if random.random() > 0.001:
            return code
            
        ops_escaped = [re.escape(op) for op in self.ASSIGN_OPS]
        ops_escaped.sort(key=len, reverse=True)
        target_regex = '|'.join(ops_escaped)
        
        victims = re.findall(target_regex, code)
        if not victims:
            return code
            
        victim = choice(victims)
        replacements = [op for op in self.ASSIGN_OPS if op != victim]
        replace = choice(replacements)
        
        code = re.sub(re.escape(victim), replace, code, 1)
        return code

    def _mr_integer(self, code):
        """
        Mutate integers with CPython-specific boundary values.
        Also mixes in special floats since they often interact in numeric contexts.
        """
        if random.random() > 0.001:
            return code
            
        target_regex = r'(?<![a-zA-Z0-9_])(?:0x[0-9a-fA-F]+|0b[01]+|0o[0-7]+|[1-9][0-9]*|0)(?![a-zA-Z0-9_])'
        
        pool = self.SPECIAL_INTS + self.SPECIAL_FLOATS
        
        victims = re.findall(target_regex, code)
        if len(victims) == 0:
            return code
            
        victim = choice(victims)
        replace = choice(pool)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_string(self, code):
        """
        Mutate strings to include bytes, unicode edge cases, and massive strings.
        """
        if random.random() > 0.01:
            return code
            
        # Regex for python strings (single/double/triple quoted, raw/bytes/f-strings)
        target_regex = r'(b?r?f?\'\'\'[\s\S]*?\'\'\'|b?r?f?"""[\s\S]*?"""|b?r?f?\'[^\']*\'|b?r?f?"[^"]*")'
        
        pool = self.SPECIAL_STRINGS + self.SPECIAL_BYTES
        
        matches = list(re.finditer(target_regex, code))
        if not matches:
            return code
            
        m = choice(matches)
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_special_constants(self, code):
        """Mutate True, False, None, Ellipsis, NotImplemented."""
        if random.random() > 0.002:
            return code
        
        target_regex = r'\b(' + '|'.join(self.SPECIAL_CONSTS) + r')\b'
        matches = list(re.finditer(target_regex, code))
        if not matches:
            return code
            
        m = choice(matches)
        original = m.group(0)
        # Replace with any other special constant
        replacement = choice([c for c in self.SPECIAL_CONSTS if c != original])
        
        start, end = m.span()
        code = code[:start] + replacement + code[end:]
        return code

    def _mr_attributes(self, code):
        """
        Randomly replaces attribute access with magic attributes.
        """
        if random.random() > 0.005:
            return code
        
        # Match dot access: .attribute
        target_regex = r'\.([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = list(re.finditer(target_regex, code))
        if not matches:
            return code
            
        magic_attrs = [
            '__class__', '__doc__', '__name__', '__dict__', '__code__', 
            '__defaults__', '__globals__', '__bases__', '__mro__', '__subclasses__'
        ]
        
        m = choice(matches)
        # Skip if already a magic attribute
        if m.group(1).startswith('__'):
            return code
            
        start, end = m.span()
        replacement = "." + choice(magic_attrs)
        code = code[:start] + replacement + code[end:]
        return code

    def mutate(self, code: str) -> str:
        # Overriding mutate completely to ensure our specific operator/assign logic is used
        # instead of the BaseMutator's simpler regexes
        
        code = self._mr_arith_operators(code)
        code = self._mr_assign_operators(code)
        code = self._mr_logical_operators(code) # Use base implementation for logical ops
        code = self._mr_integer(code)
        code = self._mr_string(code)
        
        # CPython specific additional mutations
        code = self._mr_special_constants(code)
        code = self._mr_attributes(code)
        
        return code

class SwiftMutator(BaseMutator):
    """
    Swift-Specific Mutator.
    Targets Swift types, overflow operators, and optional handling.
    """
    
    # Swift-specific special values
    SWIFT_INTS = [
        "-1", "0", "1", 
        "Int.max", "Int.min", 
        "Int8.max", "Int8.min",
        "UInt64.max",
        "0xFF", "0xFFFF", "0xFFFFFFFF",
        "1_000_000"
    ]

    SWIFT_FLOATS = [
        "Float.infinity", "-Float.infinity", "Float.nan",
        "Double.infinity", "Double.nan",
        "0.0", "-1.0", "1.0", 
        "1.7976931348623157e+308"
    ]

    SWIFT_STRINGS = [
        '""', '"A" * 1000', 
        '"\\u{0}"', '"\\u{1F4A9}"', # Null char, Emoji
        '#"Raw String"#'
    ]
    
    SWIFT_CONSTS = ["nil", "true", "false"]

    # Swift operators including overflow
    SWIFT_OPS = ["+", "-", "*", "/", "%", "&+", "&-", "&*", "&", "|", "^", "<<", ">>"]
    
    def _mr_integer(self, code):
        if random.random() > 0.001:
            return code
        
        # Match integer literals
        target_regex = r'(?<![a-zA-Z0-9_])(?:0x[0-9a-fA-F]+|[0-9]+)(?![a-zA-Z0-9_])'
        
        victims = re.findall(target_regex, code)
        if not victims:
            return code
            
        victim = choice(victims)
        replace = choice(self.SWIFT_INTS + self.SWIFT_FLOATS)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_operators(self, code):
        if random.random() > 0.001:
            return code
            
        # Escape for regex
        ops_escaped = [re.escape(op) for op in self.SWIFT_OPS]
        ops_escaped.sort(key=len, reverse=True)
        target_regex = '|'.join(ops_escaped)
        
        victims = re.findall(target_regex, code)
        if not victims:
            return code
            
        victim = choice(victims)
        replacements = [op for op in self.SWIFT_OPS if op != victim]
        replace = choice(replacements)
        
        code = code.replace(victim, replace, 1)
        return code

    def _mr_string(self, code):
        if random.random() > 0.01:
            return code
            
        target_regex = r'"([^"\\]*(\\.[^"\\]*)*)"'
        
        matches = list(re.finditer(target_regex, code))
        if not matches:
            return code
            
        m = choice(matches)
        start, end = m.span()
        replace = choice(self.SWIFT_STRINGS + self.SWIFT_CONSTS)
        
        code = code[:start] + replace + code[end:]
        return code

    def _mr_keywords(self, code):
        """Mutate specific Swift keywords."""
        if random.random() > 0.005:
            return code
            
        swaps = {
            "var": "let",
            "let": "var",
            "class": "struct",
            "struct": "class",
            "weak": "unowned",
            "unowned": "weak",
            "as?": "as!",
            "as!": "as?"
        }
        
        # Pick a keyword present in code
        candidates = [k for k in swaps.keys() if k in code]
        if not candidates:
            return code
            
        target = choice(candidates)
        # Simple replacement - could be risky with scope but acceptable for fuzzing
        # Regex to match whole word
        pattern = r'\b' + re.escape(target) + r'\b'
        if target in ["as?", "as!"]: # handle non-word chars
             pattern = re.escape(target)

        # Replace one occurrence
        match = re.search(pattern, code)
        if match:
            start, end = match.span()
            code = code[:start] + swaps[target] + code[end:]
            
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_integer(code)
        code = self._mr_operators(code) # Covers arithmetic
        code = self._mr_string(code)
        code = self._mr_keywords(code)
        return code

class RustMutator(BaseMutator):
    """
    Rust-Specific Mutator.
    Targets integer overflows, unwrap panics, and unsafe blocks.
    """
    
    RUST_INTS = [
        "0", "1", "-1",
        "i32::MAX", "i32::MIN", "u32::MAX",
        "i64::MAX", "i64::MIN", "u64::MAX",
        "usize::MAX",
        "1_000_000"
    ]
    
    RUST_STRINGS = [
        'String::new()', 'String::from("A".repeat(1000))',
        '"\\0"', '"\\u{1F4A9}"'
    ]

    def _mr_integer(self, code):
        if random.random() > 0.001: return code
        # Match literals like 123, 0xABC, 1_000
        target_regex = r'(?<![a-zA-Z0-9_])(?:0x[0-9a-fA-F_]+|[0-9][0-9_]*[iu](?:8|16|32|64|128|size)?)(?![a-zA-Z0-9_])'
        victims = re.findall(target_regex, code)
        if not victims: return code
        victim = choice(victims)
        code = code.replace(victim, choice(self.RUST_INTS), 1)
        return code

    def _mr_unwrap(self, code):
        """Randomly append .unwrap() or .expect() to potential Option/Result calls."""
        if random.random() > 0.005: return code
        # Look for closing parens that might end a function call
        matches = list(re.finditer(r'\)', code))
        if not matches: return code
        
        m = choice(matches)
        pos = m.end()
        
        suffix = choice([".unwrap()", ".expect(\"fuzzed\")", ".unwrap_or_default()"])
        code = code[:pos] + suffix + code[pos:]
        return code

    def _mr_unsafe(self, code):
        """Wrap random blocks in unsafe {} - dangerous but valid for stress testing."""
        if random.random() > 0.002: return code
        # Simple heuristic: wrap a single line assignment or call
        lines = code.splitlines()
        if len(lines) < 3: return code
        
        idx = randint(0, len(lines)-1)
        line = lines[idx].strip()
        if line and not line.startswith("unsafe") and (";" in line or "}" in line):
            lines[idx] = f"unsafe {{ {line} }}"
            return "\n".join(lines)
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_integer(code)
        code = self._mr_unwrap(code)
        code = self._mr_unsafe(code)
        return code


class HaskellMutator(BaseMutator):
    """
    Haskell-Specific Mutator.
    Targets: Int/Integer boundary values, arithmetic/list operators (including
    Haskell-unique ++ and <>), strictness annotation injection (seq/$!/bang
    patterns), and Bool literal flips.
    Each rule fires with a low independent probability so multiple mutations
    can stack in a single call, producing diverse GHC/RTS inputs.
    Overrides the base string/assign-operator rules because Haskell uses
    double-quoted String literals only (single quotes are Char literals,
    which the base regex would corrupt) and has no compound-assignment
    operators (+=, -=, ...).
    """

    HS_INTS = [
        "0", "1", "-1", "2",
        "127", "-128",                            # Int8 boundary
        "32767", "-32768",                        # Int16 boundary
        "2147483647", "-2147483648",              # Int32 boundary
        "9223372036854775807", "-9223372036854775808",  # Int64 (maxBound/minBound :: Int)
        "maxBound", "minBound",
    ]

    # Haskell arithmetic + list/semigroup operators. ++ (list append) and
    # <> (mappend) are Haskell-unique and heavily exercised by fusion RULES
    # pragmas (foldr/build) in the simplifier.
    HS_ARITH_OPS = ["+", "-", "*", "++", "<>"]

    # Matches ++, <>, or a standalone +/-/* — with lookaround guards so a
    # bare '-' is never taken from inside '->' or '<-' (function arrows /
    # do-bind arrows), which would otherwise corrupt syntax on nearly every
    # Haskell file (both tokens are ubiquitous: type signatures, do-blocks).
    _ARITH_TOKEN_RE = re.compile(r'\+\+|<>|(?<!<)-(?!>)|\+(?!\+)|\*')

    def _mr_arith_operators(self, code):
        """Mutate arithmetic/list operators, including ++ and <>."""
        if random.random() > 0.002:
            return code
        matches = list(self._ARITH_TOKEN_RE.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        victim = m.group(0)
        pool = [op for op in self.HS_ARITH_OPS if op != victim]
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_assign_operators(self, code):
        """Haskell has no compound-assignment operators — no-op."""
        return code

    def _mr_integer(self, code):
        """Replace integer literals with Haskell-specific boundary values."""
        if random.random() > 0.002:
            return code
        target_re = r'(?<![a-zA-Z0-9_.\'])(?:0x[0-9a-fA-F]+|[0-9]+)(?![a-zA-Z0-9_.\'])'
        victims = re.findall(target_re, code)
        if not victims:
            return code
        code = code.replace(choice(victims), choice(self.HS_INTS), 1)
        return code

    def _mr_string(self, code):
        """Mutate only double-quoted String literals (single quotes are Char literals)."""
        if random.random() > 0.01:
            return code
        replacements = ['""', '"\\NUL"', '"test\\0test"']
        matches = re.findall(r'"(?:[^"\\]|\\.)*"', code)
        if not matches:
            return code
        victim = choice(matches)
        replace = choice(replacements)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_bool(self, code):
        """Flip Bool literals: True <-> False."""
        if random.random() > 0.005:
            return code
        if re.search(r'\bTrue\b', code):
            code = re.sub(r'\bTrue\b', "False", code, count=1)
        elif re.search(r'\bFalse\b', code):
            code = re.sub(r'\bFalse\b', "True", code, count=1)
        return code

    def _mr_strictness(self, code):
        """Inject a `seq`/`$!` strictness forcing point before a print/return call."""
        if random.random() > 0.002:
            return code
        m = re.search(r'\b(print|return|pure)\s+(\([^()]*\)|[A-Za-z0-9_\']+)', code)
        if not m:
            return code
        whole, fn, arg = m.group(0), m.group(1), m.group(2)
        code = code.replace(whole, f"{fn} $! {arg}", 1)
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_arith_operators(code)
        code = self._mr_assign_operators(code)
        code = self._mr_integer(code)
        code = self._mr_string(code)
        code = self._mr_bool(code)
        code = self._mr_strictness(code)
        return code


