import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from runner.proxy import InferenceProxy


class _FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        payload = {"data": [{"id": "served-model"}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        response = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }]},
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "model": payload["model"],
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_proxy_allowlist_and_observable_trace():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = InferenceProxy(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        "private-upstream-key",
        "served-model",
    ).start()
    try:
        unauthorized = Request(f"{proxy.base_url}/models")
        try:
            urlopen(unauthorized)
        except Exception as exc:
            assert getattr(exc, "code", None) == 401

        proxy.begin_cell("cell-1")
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({"model": "public-alias", "messages": []}).encode(),
            headers={
                "Authorization": f"Bearer {proxy.client_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 200
        events = proxy.end_cell()
        assert [event["event_type"] for event in events] == ["proxy_request", "proxy_response"]
        assert events[0]["data"]["model"] == "public-alias"
        assert events[0]["data"]["temperature"] == 0.0
        assert events[0]["data"]["top_p"] == 1.0
        assert events[0]["data"]["max_tokens"] == 2048
        assert urlopen(
            Request(
                f"{proxy.base_url}/models",
                headers={"Authorization": f"Bearer {proxy.client_key}"},
            )
        ).read() == b'{"object": "list", "data": [{"id": "phase1-university-model", "object": "model"}]}'
        assert events[1]["data"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "add"
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_proxy_enforces_per_cell_call_limit():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = InferenceProxy(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        "private-upstream-key",
        "served-model",
        max_calls_per_cell=1,
    ).start()
    try:
        proxy.begin_cell("cell-limit")
        body = json.dumps({"model": "phase1-university-model", "messages": []}).encode()
        headers = {
            "Authorization": f"Bearer {proxy.client_key}",
            "Content-Type": "application/json",
        }
        with urlopen(Request(f"{proxy.base_url}/chat/completions", data=body, headers=headers, method="POST")):
            pass
        try:
            urlopen(Request(f"{proxy.base_url}/chat/completions", data=body, headers=headers, method="POST"))
        except Exception as exc:
            assert getattr(exc, "code", None) == 400
        else:
            raise AssertionError("proxy accepted a second model call")
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_proxy_requests_stream_usage_and_redacts_streamed_text():
    class StreamingUpstream(BaseHTTPRequestHandler):
        stream_options = None

        def log_message(self, *_args):
            return

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            type(self).stream_options = payload.get("stream_options")
            events = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "private reasoning text"},
                        "finish_reason": None,
                    }]
                },
                {"choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "id": "call-1", "type": "function",
                    "function": {"name": "add", "arguments": "{\\\"a\\\":2,\\\"b\\\":3}"},
                }]}, "finish_reason": "tool_calls"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                },
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            body += "data: [DONE]\n\n"
            encoded = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = InferenceProxy(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        "private-upstream-key",
        "served-model",
    ).start()
    try:
        proxy.begin_cell("stream-cell")
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({"model": "public-alias", "messages": [], "stream": True}).encode(),
            headers={
                "Authorization": f"Bearer {proxy.client_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request) as response:
            streamed_body = response.read().decode()

        events = proxy.end_cell()
        assert StreamingUpstream.stream_options == {"include_usage": True}
        assert events[0]["data"]["stream"] is True
        assert events[0]["data"]["stream_options"] == {"include_usage": True}
        assert events[1]["data"]["usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
        assert proxy.cell_usage() == {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
        trace_text = json.dumps(events)
        assert "private reasoning text" not in trace_text
        assert "content" not in trace_text
        assert "private reasoning text" in streamed_body
        tool_call = events[1]["data"]["choices"][2]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "add"
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
