"""Small, deterministic verification layer for claims with unambiguous answers."""

from __future__ import annotations

import ast
import operator
import re


TRUE_PATTERNS = (
    (r"\bwater freezes at 0(?:\.0)?°?\s*c(?:elsius)? at sea level\b",
     "scientific", "Water freezes at approximately 0°C under standard atmospheric pressure."),
    (r"\bthe earth revolves around the sun\b",
     "scientific", "Earth orbits the Sun."),
    (r"\bhumans need oxygen to survive\b",
     "biological", "Human cellular respiration requires oxygen under ordinary conditions."),
    (r"\bthe speed of light in (?:a )?vacuum is approximately 299[,.]?792 km/?s\b",
     "scientific", "The speed of light in vacuum is approximately 299,792 km/s."),
    (r"\ba triangle has three sides\b",
     "mathematical", "A triangle is a polygon with three sides."),
    (r"\bworld war ii ended in 1945\b",
     "historical", "World War II ended in 1945."),
    (r"\bthe moon landing occurred in 1969\b",
     "historical", "Apollo 11 landed on the Moon in 1969."),
    (r"\bmount everest is the highest mountain above sea level\b",
     "geographical", "Mount Everest has the highest elevation above mean sea level."),
    (r"\bdna contains genetic information\b",
     "biological", "DNA stores hereditary genetic information."),
    (r"\bhttp is an application[- ]layer protocol\b",
     "technical", "HTTP is an application-layer protocol."),
    (r"\bbinary search requires sorted data\b",
     "technical", "Binary search relies on ordered data."),
)

FALSE_PATTERNS = (
    (r"\ba triangle has four sides\b", "mathematical", "A triangle has three sides, not four."),
    (r"\bthe sun revolves around the earth\b", "scientific", "Earth revolves around the Sun."),
    (r"\bthe square root of 16 is 7\b", "mathematical", "The principal square root of 16 is 4."),
    (r"\bhumans can breathe underwater without equipment\b",
     "biological", "Humans cannot extract dissolved oxygen like aquatic animals."),
    (r"\bthe great wall of china is visible from the moon with the naked eye\b",
     "historical", "The Great Wall is not visibly distinguishable from the Moon with the naked eye."),
)

SUBJECTIVE_PATTERN = re.compile(
    r"\b(?:is )?(?:the best|amazing|beautiful|delicious|worst)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?")


def _safe_arithmetic(expression: str) -> float | None:
    """Evaluate only numeric arithmetic syntax; never execute arbitrary code."""
    allowed = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        value = visit(tree.body)
        return value if abs(value) < 1e12 else None
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def assess_claim(statement: str) -> dict | None:
    """Return a high-confidence assessment only when a safe rule applies."""
    normalized = _normalise(statement)

    for pattern, claim_type, reasoning in TRUE_PATTERNS:
        if re.search(pattern, normalized):
            return _result("true", claim_type, reasoning)
    for pattern, claim_type, reasoning in FALSE_PATTERNS:
        if re.search(pattern, normalized):
            return _result("false", claim_type, reasoning)

    arithmetic = re.fullmatch(r"(?:what is )?(-?[\d\s()+*/.%.-]+)\s*=\s*(-?[\d.]+)", normalized)
    if arithmetic:
        actual = _safe_arithmetic(arithmetic.group(1).replace("%", "/100"))
        expected = float(arithmetic.group(2))
        if actual is not None:
            verdict = "true" if abs(actual - expected) < 1e-9 else "false"
            return _result(
                verdict, "mathematical",
                f"The expression evaluates to {actual:g}, not {expected:g}." if verdict == "false"
                else f"The expression evaluates to {actual:g}.",
            )

    if SUBJECTIVE_PATTERN.search(normalized):
        return {
            "verdict": "not objectively verifiable",
            "status": "not_objectively_verifiable",
            "claim_type": "subjective",
            "confidence": "high",
            "confidence_score": 0.9,
            "reasoning": "This is a value judgment rather than an objectively testable fact.",
        }
    return None


def _result(verdict: str, claim_type: str, reasoning: str) -> dict:
    status = "supported" if verdict == "true" else "contradicted"
    return {
        "verdict": verdict,
        "status": status,
        "claim_type": claim_type,
        "confidence": "very high",
        "confidence_score": 0.98,
        "reasoning": reasoning,
    }
