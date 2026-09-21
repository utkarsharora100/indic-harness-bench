from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def redact_text(value: str, secrets: Sequence[str] = ()) -> str:
    result = value
    # Longest first prevents a short token from partially masking a longer one
    # in a confusing way. Empty values are never treated as secrets.
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        result = result.replace(secret, "[REDACTED]")
    return result


def redact(value: Any, secrets: Sequence[str] = ()) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, Mapping):
        return {str(key): redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact(item, secrets) for item in value]
    return value
