"""A local HTTP stub speaking the OpenAI wire format.

The adapter is exercised over a real socket rather than a mocked transport. That
keeps the test honest — request bodies, batching and error handling all travel
the same path as in production — and it needs neither a network nor an API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    """Answers embeddings and chat completions, recording every request."""

    requests: list[dict[str, Any]] = []
    embedding_dimensions: int = 3
    chat_replies: list[str] = []
    drop_last_embedding: bool = False

    def do_POST(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "payload": payload})

        if self.path.endswith("/embeddings"):
            body = self._embeddings(payload)
        elif self.path.endswith("/chat/completions"):
            body = self._chat(payload)
        else:
            self.send_response(404)
            self.end_headers()
            return

        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        sent = payload.get("input", "")
        texts = sent if isinstance(sent, list) else [sent]
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": self._vector(str(text), index),
            }
            for index, text in enumerate(texts)
        ]
        if type(self).drop_last_embedding:
            data = data[:-1]
        # Deliberately reversed: the API is not required to preserve the order of
        # its inputs, and the index restarts with every batch.
        data.reverse()
        return {"object": "list", "model": "stub-embeddings", "data": data}

    def _vector(self, text: str, index: int) -> list[float]:
        """Encode the first character of the text so order can be verified."""
        marker = float(ord(text[0])) if text else 0.0
        vector = [marker, float(index), 0.0]
        width = type(self).embedding_dimensions
        return vector[:width] + [0.0] * max(0, width - len(vector))

    def _chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        replies = type(self).chat_replies
        content = replies.pop(0) if replies else "{}"
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr logging."""


class ProviderServer:
    """Context manager running the stub on a free localhost port."""

    def __init__(
        self,
        *,
        embedding_dimensions: int = 3,
        chat_replies: list[str] | None = None,
        drop_last_embedding: bool = False,
    ) -> None:
        _Handler.requests = []
        _Handler.embedding_dimensions = embedding_dimensions
        _Handler.chat_replies = list(chat_replies or [])
        _Handler.drop_last_embedding = drop_last_embedding

        self._host = "127.0.0.1"
        self._server = ThreadingHTTPServer((self._host, 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._server.server_address[1]}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return _Handler.requests

    def embedding_requests(self) -> list[dict[str, Any]]:
        return [request for request in _Handler.requests if request["path"].endswith("/embeddings")]

    def chat_requests(self) -> list[dict[str, Any]]:
        return [
            request
            for request in _Handler.requests
            if request["path"].endswith("/chat/completions")
        ]

    def __enter__(self) -> ProviderServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
