from __future__ import annotations

import re


IGNORED_WORDLIKE_ENTITIES = {"e.g", "i.e", "etc", "/"}

TECHNICAL_PATTERNS = (
    re.compile(r"`([^`]+)`"),
    re.compile(r"\$[A-Z][A-Z0-9_]*(?:/[A-Za-z0-9_./-]+)?"),
    re.compile(r"https?://[^\s)\]}>]+"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\s)\]}>]+"),
    re.compile(r"(?<![A-Za-z0-9])/[^\s)\]}>]+"),
    re.compile(r"(?<![A-Za-z0-9])--[A-Za-z0-9][A-Za-z0-9_-]*"),
    re.compile(r"(?:[A-Za-z]:\\|/)[^\s]+"),
    re.compile(r"(?<![A-Za-z0-9])(?:in|out|docs|tests|runbooks|samples)(?:/[A-Za-z0-9_.-]+)+/?"),
    re.compile(r"\b[A-Za-z0-9_.-]+\.(?:py|js|ts|json|yaml|yml|md|csv|sql)\b"),
    re.compile(r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b"),
    re.compile(r"\b[A-Z]{2,}[0-9]+\b"),
    re.compile(r"\b\d+(?:[./:-]\d+)+\b"),
    re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"),
)


def protected_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for pattern in TECHNICAL_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(1) if match.lastindex and match.group(1) else match.group(0)
            # Inline-code captures are already delimited by backticks. Keep their
            # internal punctuation intact; stripping it would turn valid Python
            # expressions such as ``hasattr(value, 'items')`` into a different
            # protected entity. Regexes that match prose-adjacent tokens still
            # discard sentence punctuation.
            if pattern.pattern.startswith("`"):
                entity = entity.strip()
            else:
                entity = entity.rstrip("`'\".,;:!?)]}")
            # A slash fragment such as the ``/A`` in ``N/A`` or the
            # ``/normalized`` suffix inside a workspace path is not a useful
            # standalone invariant. The complete workspace path is protected
            # by the longer match.
            if entity.startswith(("/in/", "/out/", "/docs/", "/tests/", "/runbooks/", "/samples/")):
                continue
            if entity.startswith("/") and entity.count("/") == 1 and "." not in entity:
                continue
            if entity.strip() and entity not in IGNORED_WORDLIKE_ENTITIES:
                entities.add(entity)
    # A workspace path also contains shorter slash-prefixed fragments. Keep the
    # complete token and discard those redundant suffixes so the validator does
    # not ask translators to repeat the same path several times.
    return {
        entity
        for entity in entities
        if not entity.startswith("/")
        or not any(other != entity and other.endswith(entity) for other in entities)
    }


def missing_entities(source: str, variant: str) -> set[str]:
    return protected_entities(source) - protected_entities(variant)


def validate_translation(source: str, variant: str) -> list[str]:
    errors: list[str] = []
    if not variant.strip():
        errors.append("translation is empty")
    missing = sorted(missing_entities(source, variant))
    if missing:
        errors.append("missing protected entities: " + ", ".join(missing))
    return errors
