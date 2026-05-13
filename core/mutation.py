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


class GoMutator(BaseMutator):
    """
    Go-Specific Mutator.
    Targets: integer/float boundary values, numeric type swaps, bitwise operators,
    channel direction annotations, nil-check flips, slice/make boundary args,
    comparison operator mutation, goroutine injection, and var/const swaps.
    Each rule fires with a low independent probability so multiple mutations
    can stack in a single call, producing diverse compiler inputs.
    """

    # Boundary integer values — stresses constant folding, overflow detection,
    # and type-dependent wrap-around behavior in the SSA backend
    GO_INTS = [
        "0", "1", "-1", "2",
        # Typed max/min via literals (no import needed)
        "127", "-128",                          # int8
        "32767", "-32768",                       # int16
        "2147483647", "-2147483648",             # int32
        "9223372036854775807",                   # int64 max (1<<63 - 1)
        "-9223372036854775808",                  # int64 min
        "255", "65535", "4294967295",            # uint8/16/32 max
        # Hex / binary / octal forms — stresses literal parsing
        "0xFF", "0xFFFF", "0xFFFFFFFF", "0xFFFFFFFFFFFFFFFF",
        "0b11111111", "0b10000000",
        "0o777",
        # Expression-form boundaries — stresses constant-expression evaluation
        "1<<7 - 1", "-(1<<7)",
        "1<<15 - 1", "-(1<<15)",
        "1<<31 - 1", "-(1<<31)",
        "1<<63 - 1", "-(1<<63)",
        # Large / separator literals — stresses lexer
        "1_000_000", "-1_000_000", "1_000_000_000",
    ]

    # Float boundary values — stresses IEEE 754 edge cases and constant folding
    GO_FLOATS = [
        "0.0", "-0.0", "1.0", "-1.0",
        "1e308", "-1e308", "1e-308",
        "1.7976931348623157e+308",   # math.MaxFloat64 literal
        "5e-324",                    # math.SmallestNonzeroFloat64 literal
        "1.401298464324817e-45",     # math.SmallestNonzeroFloat32 literal
        "3.4028234663852886e+38",    # math.MaxFloat32 literal
        "1.0 / 0.0",                 # +Inf via expression (stresses const eval)
        "-1.0 / 0.0",                # -Inf
        "0.0 / 0.0",                 # NaN
    ]

    # Integer type names — swapping these stresses the type checker and
    # conversion/truncation paths in the compiler backend
    INT_TYPES = [
        "int", "int8", "int16", "int32", "int64",
        "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
        "byte", "rune",
    ]
    FLOAT_TYPES = ["float32", "float64"]

    # Go arithmetic + bitwise operators — &^ (bit-clear) is Go-unique and
    # rarely exercised; << and >> with large shifts hit the SSA lowering
    GO_ARITH_OPS = ["+", "-", "*", "/", "%", "&", "|", "^", "&^", "<<", ">>"]

    def _mr_arith_operators(self, code):
        """Mutate arithmetic/bitwise operators, including Go's unique &^ (bit-clear)."""
        if random.random() > 0.002:
            return code
        ops_escaped = sorted([re.escape(op) for op in self.GO_ARITH_OPS], key=len, reverse=True)
        victims = re.findall('|'.join(ops_escaped), code)
        if not victims:
            return code
        victim = choice(victims)
        pool = [op for op in self.GO_ARITH_OPS if op != victim]
        code = code.replace(victim, choice(pool), 1)
        return code

    def _mr_integer(self, code):
        """Replace integer literals with Go-specific boundary values."""
        if random.random() > 0.002:
            return code
        # Match decimal, hex, binary, octal, and underscore-separated forms.
        # Negative lookahead/lookbehind prevent matching inside identifiers or floats.
        target_re = r'(?<![a-zA-Z0-9_.])(?:0x[0-9a-fA-F][0-9a-fA-F_]*|0b[01][01_]*|0o[0-7][0-7_]*|[0-9][0-9_]*)(?![a-zA-Z0-9_.])'
        victims = re.findall(target_re, code)
        if not victims:
            return code
        victim = choice(victims)
        pool = self.GO_INTS + self.GO_FLOATS
        code = code.replace(victim, choice(pool), 1)
        return code

    def _mr_numeric_type(self, code):
        """
        Swap numeric type names to stress type inference and conversion paths.
        E.g.: int32 → int64, float32 → float64, byte → rune, uint8 → int8.
        Operates within the same family (int-family vs float-family) to keep
        the mutations semantically interesting rather than always invalid.
        """
        if random.random() > 0.005:
            return code
        all_types = self.INT_TYPES + self.FLOAT_TYPES
        type_re = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in sorted(all_types, key=len, reverse=True)) + r')\b'
        )
        matches = list(type_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        original = m.group(1)
        pool = (
            [t for t in self.FLOAT_TYPES if t != original]
            if original in self.FLOAT_TYPES
            else [t for t in self.INT_TYPES if t != original]
        )
        if not pool:
            return code
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_string(self, code):
        """
        Mutate string/rune literals with Go-specific edge cases.
        Targets: null bytes, surrogates, overlong UTF-8, raw strings,
        and very long strings that stress string interning.
        """
        if random.random() > 0.01:
            return code
        # Match interpreted strings and raw (backtick) strings; skip rune literals
        target_re = r'`[^`]*`|"(?:[^"\\]|\\.)*"'
        matches = list(re.finditer(target_re, code))
        if not matches:
            return code
        m = choice(matches)
        replacements = [
            '""',
            '"\\x00"',                   # null byte — stresses string handling
            '"\\xff\\xfe"',              # invalid UTF-8 sequence
            '"\\u0000"',                 # null code point
            '"\\ufffd"',                 # replacement character
            '"\\U0001F4A9"',             # 4-byte Unicode (above BMP)
            '`' + 'A' * 128 + '`',      # long raw string — stresses interning
            '`\\n`',                     # raw string with literal backslash-n
            '"\\n\\r\\t\\a\\b\\f\\v"',   # all whitespace escapes
        ]
        start, end = m.span()
        code = code[:start] + choice(replacements) + code[end:]
        return code

    def _mr_channel_dir(self, code):
        """
        Mutate channel direction annotations to stress the type system.
        chan T → <-chan T (receive-only) → chan<- T (send-only).
        Bidirectional channels are assignable to directional ones but
        not vice versa — mutation crosses this boundary intentionally.
        """
        if random.random() > 0.005:
            return code
        chan_re = re.compile(r'(?<![=<>!])(<-chan|chan<-|\bchan\b)(?!<-)')
        matches = list(chan_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        original = m.group(0)
        pool = [d for d in ['chan', '<-chan', 'chan<-'] if d != original]
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_nil_comparison(self, code):
        """
        Flip == nil ↔ != nil to stress nil-check elimination and
        branch inversion in the SSA optimizer.
        """
        if random.random() > 0.005:
            return code
        if '== nil' in code:
            code = code.replace('== nil', '!= nil', 1)
        elif '!= nil' in code:
            code = code.replace('!= nil', '== nil', 1)
        return code

    def _mr_slice_op(self, code):
        """
        Mutate index expressions s[i] into boundary slice expressions
        to stress bounds check elimination (BCE) and slice bounds checking.
        E.g.: s[i] → s[i:i], s[0:i], s[i:], s[:i]
        """
        if random.random() > 0.003:
            return code
        # Match s[expr] but not s[a:b] (already a slice) or s[a][b] chains
        index_re = re.compile(r'(\b\w+)\[([^\[\]:\n]+)\](?!\s*[\[:])')
        matches = list(index_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        name, idx = m.group(1), m.group(2).strip()
        pool = [
            f'{name}[{idx}:{idx}]',      # zero-length slice at index
            f'{name}[0:{idx}]',          # from start to index
            f'{name}[{idx}:]',           # from index to end
            f'{name}[:{idx}]',           # from start (cap form)
        ]
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_make_args(self, code):
        """
        Mutate make() length/capacity arguments to boundary values to stress
        the slice/map/channel allocation paths and capacity validation.
        make([]T, n) → make([]T, 0, 0) / make([]T, 1<<16)
        """
        if random.random() > 0.005:
            return code
        make_re = re.compile(r'\bmake\s*\((\[\][^,)]+|map\[[^\]]+\][^,)]+|chan\s+[^,)]+),([^)]+)\)')
        matches = list(make_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        typ = m.group(1)
        boundary_sizes = ['0', '1', '-1', '1<<10', '1<<16', '1<<24']
        new_len = choice(boundary_sizes)
        # Occasionally add an explicit cap arg too
        if random.random() < 0.4:
            new_cap = choice(boundary_sizes)
            replacement = f'make({typ}, {new_len}, {new_cap})'
        else:
            replacement = f'make({typ}, {new_len})'
        start, end = m.span()
        code = code[:start] + replacement + code[end:]
        return code

    def _mr_goroutine(self, code):
        """
        Prepend 'go ' to a standalone function call statement to stress the
        goroutine scheduler, stack growth, and the race detector.
        Only targets indented call statements (inside function bodies).
        """
        if random.random() > 0.002:
            return code
        # Match indented calls: leading whitespace + identifier( ... )
        call_re = re.compile(r'^([ \t]+)(?!go\s|defer\s|return\s|//|if\s|for\s)([a-zA-Z_]\w*\s*\([^)\n]*\))(\s*)$', re.MULTILINE)
        matches = list(call_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        indent, call, trail = m.group(1), m.group(2), m.group(3)
        start, end = m.span()
        code = code[:start] + indent + 'go ' + call + trail + code[end:]
        return code

    def _mr_comparison_op(self, code):
        """
        Mutate comparison operators to stress constant folding and
        dead code elimination — e.g., x < y → x >= y flips branch direction.
        """
        if random.random() > 0.003:
            return code
        cmp_ops = ['==', '!=', '<=', '>=', '<', '>']
        cmp_re = re.compile('|'.join(re.escape(op) for op in sorted(cmp_ops, key=len, reverse=True)))
        matches = list(cmp_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        original = m.group(0)
        pool = [op for op in cmp_ops if op != original]
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def _mr_var_const(self, code):
        """
        Swap 'var' and 'const' at declaration sites to stress the compiler's
        constant evaluation path — const requires a compile-time-evaluable
        expression, so swapping can expose constant-folding bugs.
        """
        if random.random() > 0.003:
            return code
        vc_re = re.compile(r'^(\s*)(var|const)(\s+\w)', re.MULTILINE)
        matches = list(vc_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        original = m.group(2)
        replacement = 'const' if original == 'var' else 'var'
        start, end = m.start(2), m.end(2)
        code = code[:start] + replacement + code[end:]
        return code

    def _mr_assign_op(self, code):
        """
        Mutate compound assignment operators (+=, -=, &=, |=, ^=, <<=, >>=)
        to stress the SSA lowering for in-place operations.
        """
        if random.random() > 0.002:
            return code
        assign_ops = ['+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '&^=']
        assign_re = re.compile('|'.join(re.escape(op) for op in sorted(assign_ops, key=len, reverse=True)))
        matches = list(assign_re.finditer(code))
        if not matches:
            return code
        m = choice(matches)
        original = m.group(0)
        pool = [op for op in assign_ops if op != original]
        start, end = m.span()
        code = code[:start] + choice(pool) + code[end:]
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_arith_operators(code)
        code = self._mr_assign_op(code)
        code = self._mr_integer(code)
        code = self._mr_numeric_type(code)
        code = self._mr_string(code)
        code = self._mr_channel_dir(code)
        code = self._mr_nil_comparison(code)
        code = self._mr_slice_op(code)
        code = self._mr_make_args(code)
        code = self._mr_goroutine(code)
        code = self._mr_comparison_op(code)
        code = self._mr_var_const(code)
        return code


class WGSLMutator(BaseMutator):
    """
    WGSL-Specific Mutator for naga fuzzing.
    Targets numeric literals with type suffixes, vector/scalar types, and address spaces.
    """

    WGSL_INTS = ['0i', '1i', '-1i', '2147483647i', '-2147483648i']
    WGSL_UINTS = ['0u', '1u', '4294967295u', '2u', '256u']
    WGSL_FLOATS = ['0f', '1f', '-1f', '0.0', '3.40282347e+38f', '-3.40282347e+38f', '1.0e-45f']
    WGSL_TYPES = ['f32', 'i32', 'u32', 'f16', 'bool']
    WGSL_VEC_SIZES = ['vec2', 'vec3', 'vec4']
    WGSL_ADDR_SPACES = ['private', 'workgroup', 'uniform', 'storage']

    def _mr_wgsl_float(self, code):
        """Mutate WGSL float literals like 3f, 2.5f, 0.0."""
        if random.random() > 0.02:
            return code
        victims = re.findall(r'(?<![a-zA-Z0-9_])(?:\d+\.\d*|\d*\.\d+|\d+)[fe][\d+-]*f?(?![a-zA-Z0-9_])|(?<![a-zA-Z0-9_])\d+f(?![a-zA-Z0-9_])', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_FLOATS)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_wgsl_int(self, code):
        """Mutate WGSL integer literals like 3i, 0i."""
        if random.random() > 0.02:
            return code
        victims = re.findall(r'(?<![a-zA-Z0-9_])\d+i(?![a-zA-Z0-9_])', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_INTS)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_wgsl_uint(self, code):
        """Mutate WGSL unsigned integer literals like 3u, 0u."""
        if random.random() > 0.02:
            return code
        victims = re.findall(r'(?<![a-zA-Z0-9_])\d+u(?![a-zA-Z0-9_])', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_UINTS)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_wgsl_vec_type(self, code):
        """Mutate vector dimensionality: vec2 <-> vec3 <-> vec4."""
        if random.random() > 0.01:
            return code
        victims = re.findall(r'\bvec[234]\b', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_VEC_SIZES)
        code = code.replace(victim, replace, 1)
        return code

    def _mr_wgsl_scalar_type(self, code):
        """Mutate scalar types: f32 <-> i32 <-> u32."""
        if random.random() > 0.005:
            return code
        victims = re.findall(r'\b(?:f32|i32|u32|f16|bool)\b', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_TYPES)
        if replace == victim:
            return code
        code = code.replace(victim, replace, 1)
        return code

    def _mr_wgsl_address_space(self, code):
        """Mutate address spaces in var declarations."""
        if random.random() > 0.005:
            return code
        victims = re.findall(r'var<(private|workgroup|uniform|storage)', code)
        if not victims:
            return code
        victim = choice(victims)
        replace = choice(self.WGSL_ADDR_SPACES)
        code = code.replace('var<' + victim, 'var<' + replace, 1)
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_wgsl_float(code)
        code = self._mr_wgsl_int(code)
        code = self._mr_wgsl_uint(code)
        code = self._mr_wgsl_vec_type(code)
        code = self._mr_wgsl_scalar_type(code)
        code = self._mr_wgsl_address_space(code)
        code = self._mr_arith_operators(code)
        return code


class LeanMutator(BaseMutator):
    """
    Lean 4 Specific Mutator.
    Targets: numeric boundary values, Bool literal flips, bitwise/arithmetic
    operator swaps (including Lean's &&& / ||| / ^^^), universe level bumps,
    and noncomputable annotation injection.
    Each rule fires with a low independent probability so multiple mutations
    can stack in a single call, producing diverse elaborator inputs.
    """

    LEAN_NATS = [
        "0", "1", "2",
        "255", "256",                        # UInt8 boundary
        "65535", "65536",                    # UInt16 boundary
        "4294967295", "4294967296",          # UInt32 boundary
        "9223372036854775807",               # Int64 max
        "18446744073709551615",              # UInt64 max
    ]

    LEAN_INTS = [
        "0", "1", "-1",
        "Int.minValue", "Int.maxValue",
        "(2147483647 : Int)", "(-2147483648 : Int)",
    ]

    # Lean 4 arithmetic / bitwise operators (including Lean-unique &&& ||| ^^^)
    LEAN_ARITH_OPS = ["+", "-", "*", "/", "%", "&&&", "|||", "^^^", "<<<", ">>>"]

    # Universe levels — swapping stresses universe unification
    LEAN_UNIVERSE_LEVELS = ["0", "1", "2", "u", "u+1", "max u v"]

    def _mr_lean_nat(self, code):
        """Replace Nat literals with Lean-specific boundary values."""
        if random.random() > 0.002:
            return code
        target_re = r'(?<![a-zA-Z0-9_\.])(?:0x[0-9a-fA-F][0-9a-fA-F_]*|[0-9][0-9_]*)(?![a-zA-Z0-9_\.])'
        victims = re.findall(target_re, code)
        if not victims:
            return code
        code = code.replace(choice(victims), choice(self.LEAN_NATS), 1)
        return code

    def _mr_lean_bool(self, code):
        """Flip Bool literals: true ↔ false."""
        if random.random() > 0.005:
            return code
        if "true" in code:
            code = re.sub(r'\btrue\b', "false", code, count=1)
        elif "false" in code:
            code = re.sub(r'\bfalse\b', "true", code, count=1)
        return code

    def _mr_lean_arith(self, code):
        """Mutate arithmetic/bitwise operators, including Lean's &&& ||| ^^^."""
        if random.random() > 0.002:
            return code
        ops_pat = sorted([re.escape(op) for op in self.LEAN_ARITH_OPS], key=len, reverse=True)
        victims = re.findall("|".join(ops_pat), code)
        if not victims:
            return code
        victim = choice(victims)
        pool = [op for op in self.LEAN_ARITH_OPS if op != victim]
        code = code.replace(victim, choice(pool), 1)
        return code

    def _mr_lean_universe(self, code):
        """Bump universe level annotations: Type N → Type (N±1)."""
        if random.random() > 0.003:
            return code
        # Match 'Type N' or 'Sort N'
        victims = re.findall(r'(?:Type|Sort)\s+(\d+)', code)
        if not victims:
            return code
        victim_n = choice(victims)
        new_n = str(max(0, int(victim_n) + random.choice([-1, 1])))
        code = re.sub(
            r'((?:Type|Sort)\s+)' + re.escape(victim_n),
            r'\g<1>' + new_n,
            code,
            count=1,
        )
        return code

    def _mr_lean_noncomputable(self, code):
        """Randomly inject or strip 'noncomputable' before a def."""
        if random.random() > 0.003:
            return code
        if "noncomputable def" in code:
            code = code.replace("noncomputable def", "def", 1)
        elif re.search(r'(?<!noncomputable )def ', code):
            code = re.sub(r'\bdef ', "noncomputable def ", code, count=1)
        return code

    def _mr_lean_option(self, code):
        """Wrap a 'none' with 'some none' or unwrap 'some x' to 'x'."""
        if random.random() > 0.003:
            return code
        if "none" in code:
            code = re.sub(r'\bnone\b', "some none", code, count=1)
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_lean_nat(code)
        code = self._mr_lean_bool(code)
        code = self._mr_lean_arith(code)
        code = self._mr_lean_universe(code)
        code = self._mr_lean_noncomputable(code)
        code = self._mr_lean_option(code)
        return code


class JSMutator(BaseMutator):
    """
    JavaScript-Specific Mutator.
    Targets: JS number boundary values, boolean literal flips,
    loose/strict equality swaps, and typeof/instanceof probes.
    """

    JS_NUMBERS = [
        "0", "1", "-1",
        "0.5", "-0",
        "Infinity", "-Infinity", "NaN",
        "Number.MAX_SAFE_INTEGER",   # 2**53 - 1
        "Number.MIN_SAFE_INTEGER",   # -(2**53 - 1)
        "Number.MAX_VALUE",
        "Number.EPSILON",
        "2147483647", "-2147483648",  # 32-bit int boundaries (typed arrays, bitwise)
        "4294967295",                 # Uint32 max
        "2**31", "2**32", "2**53",
    ]

    def _mr_integer(self, code):
        if random.random() > 0.002:
            return code
        target_re = r'(?<![a-zA-Z0-9_$])(?:0x[0-9a-fA-F]+|-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)(?![a-zA-Z0-9_$])'
        victims = re.findall(target_re, code)
        if not victims:
            return code
        code = code.replace(choice(victims), choice(self.JS_NUMBERS), 1)
        return code

    def _mr_js_bool(self, code):
        """Flip boolean literals: true ↔ false."""
        if random.random() > 0.005:
            return code
        if re.search(r'\btrue\b', code):
            code = re.sub(r'\btrue\b', 'false', code, count=1)
        elif re.search(r'\bfalse\b', code):
            code = re.sub(r'\bfalse\b', 'true', code, count=1)
        return code

    def _mr_js_equality(self, code):
        """Swap == ↔ === and != ↔ !==."""
        if random.random() > 0.002:
            return code
        if '===' in code:
            code = code.replace('===', '==', 1)
        elif '==' in code:
            code = code.replace('==', '===', 1)
        return code

    def _mr_js_typeof(self, code):
        """Inject a typeof/instanceof expression around a random identifier."""
        if random.random() > 0.003:
            return code
        names = re.findall(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b', code)
        if not names:
            return code
        name = choice(names)
        probe = choice([
            f'typeof {name}',
            f'{name} instanceof Object',
            f'Array.isArray({name})',
        ])
        code += f'\n// ffl probe: {probe}\n'
        return code

    def mutate(self, code: str) -> str:
        code = self._mr_arith_operators(code)
        code = self._mr_assign_operators(code)
        code = self._mr_logical_operators(code)
        code = self._mr_integer(code)
        code = self._mr_string(code)
        code = self._mr_js_bool(code)
        code = self._mr_js_equality(code)
        code = self._mr_js_typeof(code)
        return code


class CangjeMutator(BaseMutator):
    """
    Cangjie-specific mutator. Every mutation preserves syntactic validity and
    aims to preserve type-correctness so that the compiler front-end (rather
    than a trivial parse error) is exercised.

    Safe mutations applied:
        - integer / float literal replacement with boundary values
        - true  ↔ false flip
        - comparison operator swap  (<  >  <=  >=  ==  !=)
        - logical operator swap     (&& ↔ ||)
        - arithmetic sign flip      (+  ↔  -)     inside expressions
        - loop-range bound mutation (X..Y  →  X..Z)
        - string content replacement
        - function parameter value shuffling
        - return value substitution (integers / booleans)
    """

    # ── integer boundary values that fit in every Cangjie integer type ──
    _INT_VALS = ['0', '1', '-1', '2', '-2', '127', '-128', '255',
                 '32767', '-32768', '2147483647', '-2147483648',
                 '9223372036854775807', '-9223372036854775808']

    # ── float boundary values ─────────────────────────────────────────────
    _FLOAT_VALS = ['0.0', '1.0', '-1.0', '0.5', '-0.5', '3.14',
                   '1.0e10', '-1.0e10', '1.0e-10', 'Float64.infinity',
                   'Float64.nan']

    # ── comparison operators ───────────────────────────────────────────────
    _CMP_OPS = ['<', '>', '<=', '>=', '==', '!=']

    # ── string replacement pool ────────────────────────────────────────────
    _STR_VALS = ['', 'a', 'AB', '0', ' ', '\x00', 'hello', 'test\nvalue',
                 'a' * 256]

    # ─────────────────────────────────────────────────────────────────────

    def mutate(self, content: str) -> str:
        """Apply 1–3 randomly chosen mutations to *content*."""
        mutations = [
            self._cj_integer,
            self._cj_float,
            self._cj_bool_flip,
            self._cj_compare_op,
            self._cj_logical_op,
            self._cj_arith_sign,
            self._cj_range_bound,
            self._cj_string_content,
            self._cj_return_value,
        ]
        n = random.randint(1, 3)
        chosen = random.sample(mutations, min(n, len(mutations)))
        for fn in chosen:
            content = fn(content)
        return content

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _find_outside_strings(code: str, pattern: str, flags: int = 0):
        """Return re.Match objects for *pattern* that are not inside string literals."""
        # Build a simple mask: True where character is inside a string
        in_str = [False] * len(code)
        i = 0
        while i < len(code):
            if code[i] == '"':
                j = i + 1
                while j < len(code):
                    if code[j] == '\\':
                        j += 2
                        continue
                    if code[j] == '"':
                        break
                    j += 1
                for k in range(i, min(j + 1, len(code))):
                    in_str[k] = True
                i = j + 1
            else:
                i += 1

        results = []
        for m in re.finditer(pattern, code, flags):
            if not any(in_str[m.start():m.end()]):
                results.append(m)
        return results

    # ── individual mutation rules ──────────────────────────────────────────

    def _cj_integer(self, code: str) -> str:
        """Replace a bare integer literal with a boundary value."""
        if random.random() > 0.4:
            return code
        # Match standalone integers: not preceded/followed by digit, dot, letter
        matches = self._find_outside_strings(
            code, r'(?<![.\w])-?\b\d+\b(?![\d.\w])')
        if not matches:
            return code
        m = random.choice(matches)
        val = random.choice(self._INT_VALS)
        return code[:m.start()] + val + code[m.end():]

    def _cj_float(self, code: str) -> str:
        """Replace a float literal with a boundary value."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(
            code, r'(?<![\w])-?\b\d+\.\d+(?:[eE][+-]?\d+)?\b(?![\w])')
        if not matches:
            return code
        m = random.choice(matches)
        val = random.choice(self._FLOAT_VALS)
        return code[:m.start()] + val + code[m.end():]

    def _cj_bool_flip(self, code: str) -> str:
        """Flip a boolean literal: true ↔ false."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(code, r'\b(true|false)\b')
        if not matches:
            return code
        m = random.choice(matches)
        replacement = 'false' if m.group(1) == 'true' else 'true'
        return code[:m.start()] + replacement + code[m.end():]

    def _cj_compare_op(self, code: str) -> str:
        """Swap a comparison operator to another comparison operator.
        Only replaces inside condition contexts ( ... ) to stay type-safe."""
        if random.random() > 0.4:
            return code
        # Greedy match of multi-char ops first so `<=` beats `<`
        matches = self._find_outside_strings(
            code, r'(?<=\s)(<=|>=|==|!=|<(?!=|>)|>(?!=|<))(?=\s)')
        if not matches:
            return code
        m = random.choice(matches)
        current = m.group(1)
        pool = [op for op in self._CMP_OPS if op != current]
        return code[:m.start(1)] + random.choice(pool) + code[m.end(1):]

    def _cj_logical_op(self, code: str) -> str:
        """Swap && ↔ ||."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(code, r'(&&|\|\|)')
        if not matches:
            return code
        m = random.choice(matches)
        replacement = '||' if m.group(1) == '&&' else '&&'
        return code[:m.start()] + replacement + code[m.end():]

    def _cj_arith_sign(self, code: str) -> str:
        """Flip + ↔ - in arithmetic expressions (spaces required on both sides
        to avoid matching unary minus or prefix operators)."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(code, r'(?<=\s)([+\-])(?=\s\d|\s[a-zA-Z_(])')
        if not matches:
            return code
        m = random.choice(matches)
        replacement = '-' if m.group(1) == '+' else '+'
        return code[:m.start()] + replacement + code[m.end():]

    def _cj_range_bound(self, code: str) -> str:
        """Mutate the upper bound of a for-loop range  X..Y → X..Z."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(code, r'(\d+)\.\.(\d+)')
        if not matches:
            return code
        m = random.choice(matches)
        lo = int(m.group(1))
        hi = int(m.group(2))
        choice_hi = [0, 1, lo, hi - 1, hi + 1, hi * 2, 2147483647]
        new_hi = random.choice([v for v in choice_hi if v >= lo])
        if not [v for v in choice_hi if v >= lo]:
            new_hi = lo + 1
        return code[:m.start()] + f'{lo}..{new_hi}' + code[m.end():]

    def _cj_string_content(self, code: str) -> str:
        """Replace the content of a double-quoted string literal."""
        if random.random() > 0.4:
            return code
        matches = self._find_outside_strings(code, r'"([^"\\]*(?:\\.[^"\\]*)*)"')
        # _find_outside_strings returns matches that start at the opening quote —
        # since the quote itself is masked, filter to matches that start BEFORE masking.
        # Simpler: redo with raw finditer (strings only, which is what we want here)
        matches = list(re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', code))
        if not matches:
            return code
        m = random.choice(matches)
        new_content = random.choice(self._STR_VALS)
        return code[:m.start()] + '"' + new_content + '"' + code[m.end():]

    def _cj_return_value(self, code: str) -> str:
        """Replace a literal in a return statement with a boundary value."""
        if random.random() > 0.4:
            return code
        # Match `return <integer>` or `return true/false`
        m_int = re.search(r'(\breturn\s+)(-?\d+)(\s*\n)', code)
        m_bool = re.search(r'(\breturn\s+)(true|false)(\s*\n)', code)
        candidates = [c for c in [m_int, m_bool] if c is not None]
        if not candidates:
            return code
        m = random.choice(candidates)
        if m.group(2) in ('true', 'false'):
            new_val = 'false' if m.group(2) == 'true' else 'true'
        else:
            new_val = random.choice(self._INT_VALS)
        return code[:m.start(2)] + new_val + code[m.end(2):]

