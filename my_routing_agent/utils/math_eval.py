"""Sandboxed arithmetic evaluator for MATH_PYTHON route."""

from __future__ import annotations

import math
import re

MATH_EXPRESSION = re.compile(
    r"^\s*(?:what\s+is\s+)?[\d\s+\-*/().,%^]+(?:=|\?)?\s*$",
    re.IGNORECASE,
)
MATH_EXTRACT_PATTERN = re.compile(r"([\d\s\+\-\*\/\(\)\.]{3,})")
MATH_OPERATOR_PATTERN = re.compile(r"[\+\-\*\/]")

_SYMBOLIC_KEYWORDS = (
    "solve for",
    "derivative",
    "integrate",
    "differentiate",
    "symbolically",
    "equation",
    "polynomial",
    "calculus",
    "algebra",
    "factorize",
    "factorise",
    "quadratic",
    "integral",
    "d/dx",
    "lim ",
    "limit ",
)
# Algebra/calculus notation — not plain English words.
_VARIABLE_PATTERN = re.compile(
    r"(?:\d[a-zA-Z]|[a-zA-Z]\d|[a-zA-Z]\s*\^|\^[a-zA-Z]|\b[a-z]\s*[=+*/])",
    re.IGNORECASE,
)
_PRIME_CHECK_PATTERN = re.compile(
    r"(?:is\s+(-?\d+)\s+prime|"
    r"(?:list|find|give)\s+(?:all\s+)?primes?\s+(?:under|below|less than)\s+(\d+)|"
    r"primes?\s+(?:under|below)\s+(\d+))",
    re.IGNORECASE,
)


def is_symbolic_math(text: str) -> bool:
    """True when the prompt needs algebra/calculus, not raw arithmetic eval."""
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(kw in lowered for kw in _SYMBOLIC_KEYWORDS):
        return True
    if _VARIABLE_PATTERN.search(text):
        return True
    # equation with = where sides are not pure numeric
    if "=" in text:
        lhs, _, rhs = text.partition("=")
        combined = lhs + rhs
        if _VARIABLE_PATTERN.search(combined):
            return True
        if re.search(r"\d[a-zA-Z]|[a-zA-Z]\d", combined):
            return True
    return False


def is_simple_math(text: str) -> bool:
    if is_symbolic_math(text):
        return False
    if not MATH_EXPRESSION.match(text):
        return False
    expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
    allowed = set("0123456789+-*/().,%^ ")
    return bool(expr) and all(ch in allowed for ch in expr)


def _eval_arithmetic(expr: str) -> str | None:
    allowed = set("0123456789+-*/(). ")
    cleaned = expr.strip()
    if not cleaned or not all(ch in allowed for ch in cleaned):
        return None
    try:
        value = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307
        if isinstance(value, float) and math.isfinite(value):
            if value.is_integer():
                return str(int(value))
            return str(round(value, 10))
        if isinstance(value, int):
            return str(value)
    except Exception:
        return None
    return None


def try_evaluate_math(text: str) -> str | None:
    if is_symbolic_math(text):
        return None
    if not is_simple_math(text):
        return None
    expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
    return _eval_arithmetic(expr)


def extract_arithmetic_expression(text: str) -> str | None:
    """Return the first embedded pure-arithmetic expression, if any."""
    if is_symbolic_math(text):
        return None
    stripped = text.strip()
    if try_evaluate_math(stripped) is not None:
        expr = re.sub(r"(?i)^what\s+is\s+", "", stripped).strip().rstrip("=?")
        return expr
    math_prefix = re.match(r"^math:\s*(.+)$", stripped, re.IGNORECASE)
    if math_prefix:
        candidate = math_prefix.group(1).strip()
        if try_evaluate_math(candidate) is not None:
            return candidate
    for match in MATH_EXTRACT_PATTERN.finditer(stripped):
        candidate = match.group(1).strip()
        if len(candidate) < 3:
            continue
        if not MATH_OPERATOR_PATTERN.search(candidate):
            continue
        if not any(ch.isdigit() for ch in candidate):
            continue
        if _eval_arithmetic(candidate) is not None:
            return candidate
    return None


def is_local_arithmetic(text: str) -> bool:
    """True when python-eval can answer without symbolic reasoning."""
    return extract_arithmetic_expression(text) is not None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def is_prime_check_prompt(text: str) -> bool:
    lowered = text.lower()
    if "prime" not in lowered:
        return False
    return _PRIME_CHECK_PATTERN.search(text) is not None


def try_prime_check(text: str) -> str | None:
    """Deterministic primality answers for simple prime queries."""
    if not is_prime_check_prompt(text):
        return None
    match = _PRIME_CHECK_PATTERN.search(text)
    if not match:
        return None
    if match.group(1) is not None:
        n = int(match.group(1))
        return "yes" if _is_prime(n) else "no"
    limit_str = match.group(2) or match.group(3)
    if limit_str is None:
        return None
    limit = int(limit_str)
    primes = [str(p) for p in range(2, max(2, limit)) if _is_prime(p)]
    return ", ".join(primes) if primes else "none"
