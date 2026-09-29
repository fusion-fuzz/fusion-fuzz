# [clang] `static_assert` on a typo-corrected template-id asserts: "Expression evaluator can't be called on a dependent expression"

**Repo:** llvm/llvm-project · **Commit:** `b9b5fb9fec7a` · **Build:** assertions enabled · plain C++, every `-std=` from c++11 to c++20

## What happens

```
clang-24: clang/lib/AST/ExprConstant.cpp:21962:
  Assertion `!isValueDependent() && "Expression evaluator can't be called on a dependent expression"' failed.
```

`digits<>` names no template. Clang typo-corrects it to the static data member
`L::digits` ("use of undeclared identifier 'digits'; did you mean 'L::digits'?"), keeps the
template-argument list on the recovered expression, which makes it value-dependent, and
`static_assert` then hands that expression to the constant evaluator, which asserts.

## Reproducer

Two lines, no headers:

```cpp
struct L { static constexpr int digits = 0; };
static_assert(digits<>);
```

```bash
clang++ -fsyntax-only -std=c++17 static_assert_digits.cpp
```

| Variant | Result |
|---|---|
| `static_assert(digits<>);` | assertion |
| `static_assert(digits<int>);` | assertion |
| `bool b = digits<>;` | clean diagnostic |
| `static_assert(digits);` (no template arguments) | clean diagnostic |

So the trigger is exactly: a typo-corrected name that is *not* a template, used with a
template-argument list, inside a context that immediately constant-evaluates it.

## Expected

Only the diagnostics already issued. Either the recovery expression should not be
value-dependent once the correction is a non-template, or `static_assert` should skip
evaluation of a recovery expression that is.

## Notes

Found compiling CUDA sources, where `<limits>` (with `std::numeric_limits<T>::digits`) is
always in scope through clang's CUDA wrapper headers, so any stray `digits<...>` in a
`static_assert` reaches this; the plain-C++ form above is the same defect.
