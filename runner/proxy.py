from __future__ import annotations

import json
import re
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

    def __init__(
        self,
        upstream_base_url: str,
        upstream_key: str,
        resolved_model: str,
        *,
        public_model: str = "phase1-university-model",
        max_calls_per_cell: int = 40,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        bind_host: str = "0.0.0.0",
        listen_port: int = 0,
        advertised_host: str | None = None,
        client_key: str | None = None,
    ) -> None:
        parsed = urlsplit(upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelProxyError("Inference proxy requires an HTTP(S) upstream")
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_key = upstream_key
        self.resolved_model = resolved_model
        self.public_model = public_model
        self.max_calls_per_cell = max_calls_per_cell
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client_key = client_key or "phase1-" + secrets.token_urlsafe(18)
        self.bind_host = bind_host
        self.listen_port = listen_port
        self.advertised_host = advertised_host
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.requests = 0
        self._trace_lock = threading.RLock()
        self._active_trace: dict[str, Any] | None = None
        self._cell_calls = 0
        self._cell_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @property
    def base_url(self) -> str:
        if self.server is None:
            raise ModelProxyError("Inference proxy is not running")
        _host, port = self.server.server_address[:2]
        host = self.advertised_host or "127.0.0.1"
        return f"http://{host}:{port}/v1"

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
                # Do not forward /models: the upstream may reveal a private
                # filesystem-backed model identifier. Agents only need the
                # stable public alias.
                self._write_json(
                    200,
                    {"object": "list", "data": [{"id": proxy.public_model, "object": "model"}]},
                )

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
                validation_error = proxy._validate_payload(payload)
                if validation_error is not None:
                    self._write_json(400, {"error": validation_error})
                    return
                proxy._normalize_payload(payload)
                with proxy._trace_lock:
                    if proxy._active_trace is None:
                        self._write_json(409, {"error": "cell_not_started"})
                        return
                    if proxy._cell_calls >= proxy.max_calls_per_cell:
                        # This is a deterministic per-cell budget violation,
                        # not a transient upstream rate limit. A 4xx response
                        # prevents native clients from wasting their own retry
                        # budget after the experiment cap has fired.
                        self._write_json(400, {"error": "model_call_limit_exceeded"})
                        return
                    proxy._cell_calls += 1
                requested_model = payload.get("model")
                payload["model"] = proxy.resolved_model
                proxy._forward_count()
                self._forward(payload, observable_model=requested_model)

            def _forward(
                self,
                payload: dict[str, Any] | None,
                *,
                observable_model: Any = None,
            ) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
                if payload is not None:
                    proxy._record_proxy_event(
                        "request",
                        {**payload, "model": observable_model or proxy.public_model},
                    )
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
                        public_result = proxy._public_response(result, response.headers.get("Content-Type", "application/json"))
                        if payload is not None:
                            if "text/event-stream" in response.headers.get("Content-Type", "application/json"):
                                proxy._record_proxy_event("response", proxy._stream_observable(result))
                            else:
                                try:
                                    proxy._record_proxy_event("response", json.loads(result))
                                except json.JSONDecodeError:
                                    proxy._record_proxy_event("response", {"raw": "invalid_json"})
                        self.send_response(response.status)
                        content_type = response.headers.get("Content-Type", "application/json")
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(public_result)))
                        self.end_headers()
                        # Never expose an upstream filesystem-backed model ID
                        # to an agent. The proxy's public alias is the only
                        # model identity visible outside the sidecar.
                        self.wfile.write(public_result)
                except urllib.error.HTTPError as exc:
                    # Upstream error bodies can contain private deployment
                    # paths or echoed credentials. Keep them out of traces.
                    exc.read(4096)
                    proxy._record_proxy_event("error", {"status": exc.code})
                    self._write_json(exc.code, {"error": "upstream_http_error"})
                except (urllib.error.URLError, TimeoutError, OSError):
                    proxy._record_proxy_event("error", {"error": "upstream_unreachable"})
                    self._write_json(502, {"error": "upstream_unreachable"})

        # The native harness container reaches this port through Docker's host
        # gateway.  The client token and route allow-list still prevent it from
        # becoming a general-purpose forwarding proxy.
        self.server = ThreadingHTTPServer((self.bind_host, self.listen_port), Handler)
        self.timeout_seconds = 120
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def _public_response(self, raw: bytes, content_type: str) -> bytes:
        if "text/event-stream" in content_type:
            # Blank lines delimit SSE events. Preserve them: collapsing event
            # boundaries makes native clients buffer all tool-call deltas until
            # the end and can prevent execution of an otherwise valid call.
            events = []
            text = raw.decode("utf-8", errors="replace")
            for event in re.split(r"\r?\n\r?\n", text):
                lines: list[str] = []
                for line in event.splitlines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data and data != "[DONE]":
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                payload = None
                            if isinstance(payload, dict) and "model" in payload:
                                payload["model"] = self.public_model
                                line = "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    lines.append(line)
                events.append("\n".join(lines))
            suffix = "\n\n" if text.endswith(("\n\n", "\r\n\r\n")) else ""
            return ("\n\n".join(events) + suffix).encode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict) and "model" in payload:
            payload["model"] = self.public_model
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return raw

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
            self._cell_calls = 0
            self._cell_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def end_cell(self) -> list[dict[str, Any]]:
        with self._trace_lock:
            active = self._active_trace
            self._active_trace = None
        return list(active.get("events", [])) if active is not None else []

    def cell_usage(self) -> dict[str, int | None]:
        with self._trace_lock:
            if self._active_trace is not None:
                return dict(self._cell_usage)
            return dict(self._cell_usage)

    def snapshot_cell(self) -> dict[str, Any]:
        with self._trace_lock:
            events = list((self._active_trace or {}).get("events", []))
            return {
                "events": events,
                "usage": dict(self._cell_usage),
                "calls": self._cell_calls,
            }

    def _validate_payload(self, payload: dict[str, Any]) -> str | None:
        if payload.get("temperature", self.temperature) not in {None, self.temperature}:
            return "temperature_must_match_experiment"
        if payload.get("top_p", self.top_p) not in {None, self.top_p}:
            return "top_p_must_match_experiment"
        requested_tokens = payload.get("max_tokens", self.max_tokens)
        if requested_tokens not in {None, self.max_tokens}:
            return "max_tokens_must_match_experiment"
        return None

    def _normalize_payload(self, payload: dict[str, Any]) -> None:
        """Fill provider-omitted generation fields with the frozen settings."""
        if payload.get("temperature") is None:
            payload["temperature"] = self.temperature
        if payload.get("top_p") is None:
            payload["top_p"] = self.top_p
        if payload.get("max_tokens") is None:
            payload["max_tokens"] = self.max_tokens

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
            delta = choice.get("delta") or {}
            visible: dict[str, Any] = {
                "finish_reason": choice.get("finish_reason"),
                "message": {"role": message.get("role")},
            }
            if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                visible["message"]["tool_calls"] = message["tool_calls"]
            if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
                visible["message"]["tool_calls"] = delta["tool_calls"]
            choices.append(visible)
        return {"choices": choices, "usage": payload.get("usage")}

    @staticmethod
    def _stream_observable(raw: bytes) -> dict[str, Any]:
        choices: list[dict[str, Any]] = []
        usage: dict[str, Any] | None = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                raw_choices = payload.get("choices")
                if isinstance(raw_choices, list):
                    choices.extend(raw_choices)
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
        return {"choices": choices, "usage": usage, "streamed": True}

    def _record_proxy_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._trace_lock:
            if self._active_trace is None:
                return
            if event_type == "request":
                observable = self._observable_request(payload)
            elif event_type == "response":
                observable = self._observable_response(payload)
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    with self._trace_lock:
                        for source, target in (
                            ("prompt_tokens", "input_tokens"),
                            ("completion_tokens", "output_tokens"),
                            ("total_tokens", "total_tokens"),
                        ):
                            value = usage.get(source)
                            if isinstance(value, int):
                                self._cell_usage[target] += value
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
