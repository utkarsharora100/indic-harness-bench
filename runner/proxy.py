from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


class ModelProxyError(RuntimeError):
    """The local model proxy could not be started or reached its upstream."""


class InferenceProxy:
    """A small allow-listed OpenAI-compatible model proxy.

    Agents receive only a loopback URL and a short-lived client token.  The
    university URL and bearer key stay in this process and are never included
    in request logs, traces, or model metadata.  The proxy also normalizes the
    public model alias to the exact model pinned during preflight.
    """

    def __init__(self, upstream_base_url: str, upstream_key: str, resolved_model: str) -> None:
        parsed = urlsplit(upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelProxyError("Inference proxy requires an HTTP(S) upstream")
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_key = upstream_key
        self.resolved_model = resolved_model
        self.client_key = "phase1-" + secrets.token_urlsafe(18)
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.requests = 0
        self._trace_lock = threading.Lock()
        self._active_trace: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        if self.server is None:
            raise ModelProxyError("Inference proxy is not running")
        _host, port = self.server.server_address[:2]
        return f"http://127.0.0.1:{port}/v1"

    @property
    def container_base_url(self) -> str:
        if self.server is None:
            raise ModelProxyError("Inference proxy is not running")
        _host, port = self.server.server_address[:2]
        return f"http://host.docker.internal:{port}/v1"

    def start(self) -> "InferenceProxy":
        if self.server is not None:
            return self
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "IndicHarnessProxy/1"

            def log_message(self, *_args: Any) -> None:
                # Never write request URLs, headers, prompts, or credentials.
                return

            def _authorized(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {proxy.client_key}"

            def _target(self) -> str:
                path = self.path.split("?", 1)[0]
                if path.startswith("/v1/"):
                    path = path[3:]
                elif path == "/v1":
                    path = "/"
                if not path.startswith("/"):
                    path = "/" + path
                return proxy.upstream_base_url + path

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self._write_json(200, {"ok": True})
                    return
                if not self._authorized():
                    self._write_json(401, {"error": "unauthorized"})
                    return
                if self.path.rstrip("/") not in {"/v1/models", "/models"}:
                    self._write_json(404, {"error": "route_not_allowed"})
                    return
                self._forward(None)

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._write_json(401, {"error": "unauthorized"})
                    return
                if self.path.split("?", 1)[0] not in {"/v1/chat/completions", "/chat/completions"}:
                    self._write_json(404, {"error": "route_not_allowed"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16 * 1024 * 1024:
                    self._write_json(413, {"error": "invalid_body_size"})
                    return
                try:
                    payload = json.loads(self.rfile.read(size))
                except json.JSONDecodeError:
                    self._write_json(400, {"error": "invalid_json"})
                    return
                if not isinstance(payload, dict):
                    self._write_json(400, {"error": "body_must_be_object"})
                    return
                payload["model"] = proxy.resolved_model
                proxy._forward_count()
                self._forward(payload)

            def _forward(self, payload: dict[str, Any] | None) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
                if payload is not None:
                    proxy._record_proxy_event("request", payload)
                request = urllib.request.Request(
                    proxy._target_for(self.path),
                    data=body,
                    headers={
                        "Authorization": f"Bearer {proxy.upstream_key}",
                        "Accept": "application/json",
                        **({"Content-Type": "application/json"} if body is not None else {}),
                    },
                    method="POST" if body is not None else "GET",
                )
                try:
                    with urllib.request.urlopen(request, timeout=proxy.timeout_seconds) as response:
                        result = response.read()
                        self.send_response(response.status)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Content-Length", str(len(result)))
                        self.end_headers()
                        self.wfile.write(result)
                        if payload is not None:
                            try:
                                proxy._record_proxy_event("response", json.loads(result))
                            except json.JSONDecodeError:
                                proxy._record_proxy_event("response", {"raw": "invalid_json"})
                except urllib.error.HTTPError as exc:
                    detail = exc.read(4096).decode("utf-8", errors="replace")
                    proxy._record_proxy_event(
                        "error", {"status": exc.code, "detail": detail}
                    )
                    self._write_json(exc.code, {"error": "upstream_http_error", "detail": detail})
                except (urllib.error.URLError, TimeoutError, OSError):
                    proxy._record_proxy_event("error", {"error": "upstream_unreachable"})
                    self._write_json(502, {"error": "upstream_unreachable"})

        # The native harness container reaches this port through Docker's host
        # gateway.  The client token and route allow-list still prevent it from
        # becoming a general-purpose forwarding proxy.
        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.timeout_seconds = 120
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def _target_for(self, path: str) -> str:
        clean = path.split("?", 1)[0]
        if clean.startswith("/v1/"):
            clean = clean[3:]
        elif clean == "/v1":
            clean = "/"
        if not clean.startswith("/"):
            clean = "/" + clean
        return self.upstream_base_url + clean

    def _forward_count(self) -> None:
        self.requests += 1

    def begin_cell(self, cell_id: str) -> None:
        with self._trace_lock:
            self._active_trace = {"cell_id": cell_id, "events": []}

    def end_cell(self) -> list[dict[str, Any]]:
        with self._trace_lock:
            active = self._active_trace
            self._active_trace = None
        return list(active.get("events", [])) if active is not None else []

    @staticmethod
    def _observable_request(payload: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for message in payload.get("messages", []):
            if not isinstance(message, dict):
                continue
            item: dict[str, Any] = {"role": message.get("role")}
            if message.get("role") in {"system", "user", "tool"}:
                item["content"] = message.get("content")
            if isinstance(message.get("tool_calls"), list):
                item["tool_calls"] = message["tool_calls"]
            if message.get("tool_call_id") is not None:
                item["tool_call_id"] = message["tool_call_id"]
            messages.append(item)
        return {
            "model": payload.get("model"),
            "messages": messages,
            "tools": payload.get("tools"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "max_tokens": payload.get("max_tokens"),
        }

    @staticmethod
    def _observable_response(payload: dict[str, Any]) -> dict[str, Any]:
        choices = []
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            visible: dict[str, Any] = {
                "finish_reason": choice.get("finish_reason"),
                "message": {"role": message.get("role")},
            }
            if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                visible["message"]["tool_calls"] = message["tool_calls"]
            choices.append(visible)
        return {"choices": choices, "usage": payload.get("usage")}

    def _record_proxy_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._trace_lock:
            if self._active_trace is None:
                return
            if event_type == "request":
                observable = self._observable_request(payload)
            elif event_type == "response":
                observable = self._observable_response(payload)
            else:
                observable = payload
            self._active_trace["events"].append(
                {
                    "event_type": f"proxy_{event_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": observable,
                }
            )

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.server = None
        self.thread = None
