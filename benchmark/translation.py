from __future__ import annotations

import re

TECHNICAL_PATTERNS = (
    re.compile(r"`([^`]+)`"),
    re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+"),
    re.compile(r"\b[A-Za-z0-9_.-]+\.(?:py|js|ts|json|yaml|yml|md|csv|sql)\b"),
)


def protected_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for pattern in TECHNICAL_PATTERNS:
        entities.update(match.group(1) if match.lastindex else match.group(0) for match in pattern.finditer(text))
    return {entity for entity in entities if entity.strip()}


def missing_entities(source: str, variant: str) -> set[str]:
    return protected_entities(source) - protected_entities(variant)
