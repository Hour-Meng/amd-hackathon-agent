"""Sandboxed arithmetic evaluator for MATH_PYTHON route."""

from __future__ import annotations

import math
import re

MATH_EXPRESSION = re.compile(
    r"^\s*(?:what\s+is\s+)?[\d\s+\-*/().,%^]+(?:=|\?)?\s*$",
    re.IGNORECASE,
)


def is_simple_math(text: str) -> bool:
    if not MATH_EXPRESSION.match(text):
        return False
    expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
    allowed = set("0123456789+-*/().,%^ ")
    return bool(expr) and all(ch in allowed for ch in expr)


def try_evaluate_math(text: str) -> str | None:
    if not is_simple_math(text):
        return None
    expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
    allowed = set("0123456789+-*/(). ")
    if not expr or not all(ch in allowed for ch in expr):
        return None
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        if isinstance(value, float) and math.isfinite(value):
            if value.is_integer():
                return str(int(value))
            return str(round(value, 10))
        if isinstance(value, int):
            return str(value)
    except Exception:
        return None
    return None
