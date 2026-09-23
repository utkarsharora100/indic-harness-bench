from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

from runner.inference import InferenceTransientError


class ProxySidecar:
    """One-cell model proxy reachable only from the internal agent network."""

    def __init__(
        self,
        *,
        image: str,
        internal_network: str,
        egress_network: str,
        upstream_base_url: str,
        upstream_key: str,
        resolved_model: str,
        public_model: str,
        trace_root: Path,
        cell_id: str,
        max_calls: int = 40,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> None:
        self.image = image
        self.internal_network = internal_network
        self.egress_network = egress_network
        self.upstream_base_url = upstream_base_url
        self.upstream_key = upstream_key
        self.resolved_model = resolved_model
        self.public_model = public_model
        self.trace_root = trace_root
        self.cell_id = cell_id
        self.max_calls = max_calls
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client_key = "phase1-" + secrets.token_urlsafe(18)
        self.container: Any | None = None
        self.client: Any | None = None
        self.host_trace = trace_root / f"{cell_id}-proxy"
        self.host_trace.mkdir(parents=True, exist_ok=True)

    def start(self) -> "ProxySidecar":
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError("Docker Python package is required for the model proxy sidecar") from exc
        self.client = docker.from_env()
        self.client.images.get(self.image)
        try:
            stale = self.client.containers.get("phase1-model-proxy")
            stale.remove(force=True)
        except docker.errors.NotFound:
            pass
        networking_config = self.client.api.create_networking_config(
            {
                self.internal_network: self.client.api.create_endpoint_config(
                    aliases=["phase1-model-proxy"]
                )
            }
        )
        self.container = self.client.containers.run(
            self.image,
            detach=True,
            network=self.internal_network,
            networking_config=networking_config,
            network_disabled=False,
            hostname="phase1-model-proxy",
            # Runs are sequential, so a stable DNS name gives the native
            # harnesses a deterministic endpoint on the internal network.
            name="phase1-model-proxy",
            environment={
                "PHASE1_UPSTREAM_BASE_URL": self.upstream_base_url,
                "PHASE1_UPSTREAM_API_KEY": self.upstream_key,
                "PHASE1_RESOLVED_MODEL": self.resolved_model,
                "PHASE1_PUBLIC_MODEL": self.public_model,
                "PHASE1_CLIENT_KEY": self.client_key,
                "PHASE1_CELL_ID": self.cell_id,
                "PHASE1_MAX_CALLS": str(self.max_calls),
                "PHASE1_TEMPERATURE": str(self.temperature),
                "PHASE1_TOP_P": str(self.top_p),
                "PHASE1_MAX_TOKENS": str(self.max_tokens),
            },
            volumes={str(self.host_trace): {"bind": "/trace", "mode": "rw"}},
        )
        egress = self.client.networks.get(self.egress_network)
        try:
            # The default Docker bridge rejects network-scoped aliases; the
            # stable container name is already resolvable on the agent net.
            egress.connect(self.container)
        except Exception:
            self.container.remove(force=True)
            self.container = None
            self.client.close()
            self.client = None
            raise
        time.sleep(0.5)
        return self

    @property
    def base_url(self) -> str:
        return "http://phase1-model-proxy:8080/v1"

    def snapshot(self) -> dict[str, Any]:
        path = self.host_trace / "proxy.json"
        for _ in range(10):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
        raise RuntimeError("Model proxy did not write a valid trace snapshot")

    def seal(self) -> dict[str, Any]:
        """Flush the sidecar after agent exit, then read its final trace."""
        if self.container is None:
            raise RuntimeError("Model proxy sidecar was not started")
        self.container.kill(signal="SIGTERM")
        self.container.wait(timeout=10)
        snapshot = self.snapshot()
        events = snapshot.get("events") or []
        calls = int(snapshot.get("calls", 0))
        requests = sum(event.get("event_type") == "proxy_request" for event in events)
        responses = sum(event.get("event_type") == "proxy_response" for event in events)
        errors = [event.get("data") or {} for event in events
                  if event.get("event_type") == "proxy_error"]
        if calls >= 1 and requests == calls and responses + len(errors) == calls and errors:
            if all(data.get("error") == "upstream_unreachable"
                   or data.get("status") in {408, 425, 429}
                   or isinstance(data.get("status"), int) and data["status"] >= 500
                   for data in errors):
                raise InferenceTransientError("Transient upstream failure recorded by the model sidecar")
        if calls < 1 or requests != calls or responses != calls:
            raise RuntimeError(
                f"Incomplete model proxy trace: calls={calls}, requests={requests}, responses={responses}"
            )
        return snapshot

    def model_config(self, original: dict[str, Any]) -> dict[str, Any]:
        config = dict(original)
        # The native harness must see only the stable public alias.  The
        # private served model identifier is held by the sidecar and used
        # only when it forwards a request upstream.
        config.update(
            {
                "model": self.public_model,
                "base_url": self.base_url,
                "container_base_url": self.base_url,
                "api_key": self.client_key,
            }
        )
        return config

    def close(self) -> None:
        try:
            if self.container is not None:
                try:
                    self.container.reload()
                    if self.container.attrs.get("State", {}).get("Running"):
                        self.container.kill()
                except Exception:
                    pass
                try:
                    self.container.remove(force=True)
                except Exception:
                    pass
        finally:
            if self.client is not None:
                self.client.close()
            try:
                for path in self.host_trace.iterdir():
                    path.unlink()
                self.host_trace.rmdir()
            except OSError:
                pass
