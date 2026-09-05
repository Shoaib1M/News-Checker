"""Small, deterministic verification layer for claims with unambiguous answers.

This layer returns "very high" confidence and skips evidence retrieval
entirely, so a false positive here is the most confidently wrong output the
system can produce. It therefore fires only on plain, unqualified statements:
anything that negates, quotes, denies or comments on the proposition it
contains is handed to the evidence pipeline instead.
"""

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

# Superlatives only mark an opinion in predicative position. Matching the bare
# phrase classified "the best-selling car in 2024 was the Model Y" — a fact
# about sales figures — as a value judgment. This mirrors the same rule in
# claim_triage._OPINION_MARKERS; both run, so both had to be fixed.
# Frames that change what a sentence asserts relative to the proposition
# inside it. The pattern tables below match substrings, so without this check
# they fired on the embedded proposition and ignored the sentence around it:
#
#   "It is false that a triangle has four sides"  ->  verdict "false"
#       (the statement is TRUE — it correctly denies the four-sided triangle)
#   "Nobody claims world war ii ended in 1945"    ->  verdict "true"
#       (the statement is FALSE — people do claim that)
#
# Detecting the frame is enough; resolving what it means is the evidence
# pipeline's job, so a match here simply declines to answer deterministically.
_QUALIFYING_FRAME = re.compile(
    r"\b(?:not|never|no one|nobody|none|isn't|is not|aren't|are not|"
    r"wasn't|was not|doesn't|does not|don't|do not|didn't|did not|"
    r"false that|untrue that|myth|mythical|hoax|debunk\w*|disproven|"
    r"disproved|refut\w*|contrary to|contrary|allegedly|supposedly|"
    r"claims? that|claimed that|believe[sd]? that|some say|it is said|"
    r"used to|no longer|incorrect\w*|wrongly)\b",
    re.IGNORECASE,
)

SUBJECTIVE_PATTERN = re.compile(
    r"\b(?:is|are|was|were)\s+(?:the\s+(?:best|worst|greatest)\b(?!-|\s*(?:selling|"
    r"known|paid|performing|rated|documented|recorded|attended))"
    r"|(?:so\s+|very\s+|really\s+)?(?:amazing|beautiful|delicious))\b",
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

    # A qualified statement is not the proposition it contains. Decline and
    # let the evidence pipeline handle it rather than answering the wrong
    # question with very high confidence.
    if _QUALIFYING_FRAME.search(normalized):
        return None

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
