from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InferenceConfigurationError(RuntimeError):
    """The configured inference service cannot be used for this experiment."""


class InferenceTransientError(RuntimeError):
    """A request failed in a way that is safe to retry."""


@dataclass(frozen=True, slots=True)
class InferenceEndpoint:
    base_url: str
    api_key: str
    requested_model: str | None
    resolved_model: str
    served_models: tuple[str, ...]

    def model_config(self) -> dict[str, Any]:
        # The key stays in process memory and is never included in traces,
        # manifests, or run metadata.
        return {
            "model": self.resolved_model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "provider": "university_gpu",
        }


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise InferenceConfigurationError(f"Private inference file not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise InferenceConfigurationError(f"Invalid environment line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _json_request(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    body = None
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        if exc.code in {408, 425, 429} or exc.code >= 500:
            raise InferenceTransientError(f"HTTP {exc.code}: {detail}") from exc
        raise InferenceConfigurationError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InferenceTransientError(str(exc)) from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceConfigurationError(f"Inference endpoint returned invalid JSON: {raw[:500]!r}") from exc
    if not isinstance(result, dict):
        raise InferenceConfigurationError("Inference endpoint returned a non-object JSON value")
    return result


def list_models(base_url: str, api_key: str, timeout_seconds: int = 30) -> list[dict[str, Any]]:
    response = _json_request(
        f"{base_url.rstrip('/')}/models",
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    raw_models = response.get("data") or response.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise InferenceConfigurationError("Inference endpoint returned no models")
    models = [item for item in raw_models if isinstance(item, dict)]
    if not models:
        raise InferenceConfigurationError("Inference endpoint returned no usable model objects")
    return models


def model_id(model: dict[str, Any]) -> str:
    for key in ("id", "model", "name"):
        value = model.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise InferenceConfigurationError(f"Model object has no usable identifier: {model!r}")


def resolve_university_gpu(
    root: Path,
    inference_config: dict[str, Any],
    *,
    verify_tool: bool = False,
) -> tuple[InferenceEndpoint, dict[str, Any]]:
    env_path = root / inference_config.get("env_file", ".env.uni-gpu.local")
    values = load_env_file(env_path)
    base_url = values.get("INDIC_UNI_GPU_BASE_URL", "").strip()
    api_key = values.get("INDIC_UNI_GPU_API_KEY", "").strip()
    requested_name = inference_config.get("requested_model_env", "INDIC_UNI_GPU_MODEL")
    requested_model = values.get(requested_name, "").strip() or None
    if not base_url or not api_key:
        raise InferenceConfigurationError(
            f"{env_path} must define INDIC_UNI_GPU_BASE_URL and INDIC_UNI_GPU_API_KEY"
        )

    models = list_models(base_url, api_key, int(inference_config.get("timeout_seconds", 90)))
    served_ids = tuple(model_id(item) for item in models)
    resolved = None
    if requested_model in served_ids:
        resolved = requested_model
    elif len(served_ids) == 1:
        # The supplied private file may contain a friendly label while the
        # server exposes a filesystem-backed model ID. A single served model
        # is unambiguous and is pinned in the local manifest below.
        resolved = served_ids[0]
    else:
        raise InferenceConfigurationError(
            f"Requested model {requested_model!r} is not served; available models: {list(served_ids)!r}"
        )

    endpoint = InferenceEndpoint(
        base_url=base_url,
        api_key=api_key,
        requested_model=requested_model,
        resolved_model=resolved,
        served_models=served_ids,
    )
    tool_check: dict[str, Any] = {"verified": False}
    if verify_tool:
        tool_check = verify_tool_call(endpoint, int(inference_config.get("timeout_seconds", 90)))
    manifest = {
        "schema_version": 1,
        "provider": "university_gpu",
        "requested_model": requested_model,
        "resolved_model": resolved,
        "served_models": list(served_ids),
        "tool_call_check": tool_check,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    return endpoint, manifest


def verify_tool_call(endpoint: InferenceEndpoint, timeout_seconds: int = 90) -> dict[str, Any]:
    payload = {
        "model": endpoint.resolved_model,
        "messages": [{"role": "user", "content": "Use the add tool to add 2 and 3."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 100,
        "stream": False,
    }
    response = _json_request(
        f"{endpoint.base_url.rstrip('/')}/chat/completions",
        api_key=endpoint.api_key,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise InferenceConfigurationError("Tool-call check returned no completion choice")
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        raise InferenceConfigurationError("University model did not return a tool call")
    first = calls[0] if isinstance(calls[0], dict) else {}
    function = first.get("function") if isinstance(first, dict) else {}
    if not isinstance(function, dict) or function.get("name") != "add":
        raise InferenceConfigurationError("University model returned the wrong tool call")
    return {
        "verified": True,
        "tool": function.get("name"),
        "finish_reason": choices[0].get("finish_reason"),
    }


def write_model_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_model_manifest(
    root: Path,
    inference_config: dict[str, Any],
    *,
    verify_tool: bool,
) -> tuple[InferenceEndpoint, dict[str, Any]]:
    endpoint, current = resolve_university_gpu(root, inference_config, verify_tool=verify_tool)
    manifest_path = root / inference_config.get("manifest", "data/phase1/model-manifest.json")
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceConfigurationError(f"Cannot read model manifest: {manifest_path}") from exc
        if (
            previous.get("resolved_model") != current["resolved_model"]
            or previous.get("served_models") != current.get("served_models")
        ):
            raise InferenceConfigurationError(
                "Served model inventory changed since the existing experiment manifest was created"
            )
        if verify_tool and not previous.get("tool_call_check", {}).get("verified"):
            raise InferenceConfigurationError("Existing model manifest has no verified tool-call check")
        return endpoint, previous
    if not current["tool_call_check"].get("verified"):
        raise InferenceConfigurationError("A new model manifest requires a verified tool-call check")
    write_model_manifest(manifest_path, current)
    return endpoint, current


def public_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return model metadata safe to put in a run record or report."""
    return {
        "provider": config.get("provider"),
        "model": config.get("model"),
    }
