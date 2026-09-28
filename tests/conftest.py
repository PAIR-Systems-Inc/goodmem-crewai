"""Test support: drive the real SDK over a mock HTTP transport.

Tests exercise the SDK's own request building, serialization and NDJSON
parsing. Nothing inside the integration or the SDK is monkeypatched, so a
passing test means the wire behaviour is right rather than that a stub was
called.

Event shapes are taken from ``fixtures/retrieve_real.ndjson``, captured from a
live GoodMem server (v1.0.320), not hand-written from a guess at the schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import urlsplit

from goodmem import Goodmem
import httpx
import pytest


FIXTURES = Path(__file__).parent / "fixtures"

# A real vector relevance score, straight from the capture. It is negative:
# GoodMem vector scores are opaque similarities, not 0-1 relevance.
REAL_VECTOR_SCORE = -0.5345187187194824


def real_events() -> list[dict[str, Any]]:
    text = (FIXTURES / "retrieve_real.ndjson").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _template(kind: str) -> dict[str, Any]:
    for event in real_events():
        if kind in event:
            return event
    raise AssertionError(f"fixture has no {kind} event")


def chunk_event(
    chunk_id: str,
    text: str,
    memory_id: str,
    score: float = REAL_VECTOR_SCORE,
) -> dict[str, Any]:
    """A retrievedItem with the real field set, overriding only what matters."""
    event = json.loads(json.dumps(_template("retrievedItem")))
    reference = event["retrievedItem"]["chunk"]
    reference["chunk"]["chunkId"] = chunk_id
    reference["chunk"]["chunkText"] = text
    reference["chunk"]["memoryId"] = memory_id
    reference["relevanceScore"] = score
    return event


def memory_event(
    memory_id: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    event = json.loads(json.dumps(_template("memoryDefinition")))
    event["memoryDefinition"]["memoryId"] = memory_id
    event["memoryDefinition"]["metadata"] = metadata or {}
    return event


def status_event(code: str | None, message: str, **details: str) -> dict[str, Any]:
    """A status event. ``code=None`` omits the field entirely.

    A code this SDK does not know decodes to ``None``; pass a made-up string
    to simulate a newer server.
    """
    status: dict[str, Any] = {"message": message}
    if code is not None:
        status["code"] = code
    if details:
        status["details"] = details
    return {"status": status}


def ndjson(*objects: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n".join(json.dumps(o) for o in objects),
        headers={"content-type": "application/x-ndjson"},
    )


class Recorder:
    """Records outgoing requests and serves queued responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, Any]] = []
        self.default: Any = None

    def route(self, method: str, path_suffix: str, response: Any) -> None:
        self.routes.append((method.upper(), path_suffix, response))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, suffix, response in self.routes:
            if request.method == method and request.url.path.endswith(suffix):
                return response(request) if callable(response) else response
        if self.default is not None:
            return self.default(request) if callable(self.default) else self.default
        return httpx.Response(
            404, json={"error": f"unrouted {request.method} {request.url.path}"}
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        content = self.requests[index].content
        return json.loads(content) if content else {}

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def space_json(space_id: str, name: str) -> dict[str, Any]:
    """A Space with the fields the server really returns (captured live)."""
    return {
        "spaceId": space_id,
        "name": name,
        "labels": {},
        "spaceEmbedders": [
            {
                "spaceId": space_id,
                "embedderId": "019cfd1c-c033-7517-b7de-f73941a0464b",
                "defaultRetrievalWeight": 1.0,
                "createdAt": 1789607045012,
                "updatedAt": 1789607045012,
                "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
                "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
            }
        ],
        "createdAt": 1789607045012,
        "updatedAt": 1789607045012,
        "ownerId": "019cfcff-37c5-76d0-bd46-8525e29a9c82",
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 256,
                "chunkOverlap": 25,
                "separators": [],
                "keepStrategy": "KEEP_END",
                "separatorIsRegex": False,
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    }


def memory_json(
    memory_id: str, *, content_b64: str | None = None, content_type: str = "text/plain"
) -> dict[str, Any]:
    """A Memory as the server returns it. ``originalContent`` is base64."""
    payload: dict[str, Any] = {
        "memoryId": memory_id,
        "spaceId": "01a0ace4-678d-7459-aa91-b6ccd46d97d8",
        "originalContentLength": 11,
        "originalContentSha256": "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        "contentType": content_type,
        "processingStatus": "COMPLETED",
        "pageImageStatus": "PENDING",
        "pageImageCount": 0,
        "metadata": {"title": "t"},
        "createdAt": 1789607045060,
        "updatedAt": 1789607047968,
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
    }
    if content_b64 is not None:
        payload["originalContent"] = content_b64
    return payload


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(recorder: Recorder) -> Goodmem:
    """An SDK client whose transport is the recorder.

    base_url and the API key live on the httpx client: the SDK refuses to take
    them alongside an injected http_client.
    """
    http_client = httpx.Client(
        transport=httpx.MockTransport(recorder.handler),
        base_url="https://goodmem.test",
        headers={"x-api-key": "test-key"},
    )
    return Goodmem(http_client=http_client)


# ------------------------------------------------ a real, recording HTTP server
class RecordingServer(ThreadingHTTPServer):
    """A local HTTP server that records every request exactly as it arrived.

    Unlike ``Recorder``, requests cross a real socket: the integration builds
    its own SDK client from ``base_url``/``api_key``, and ``requests`` holds
    each request line's method and raw path (dot segments and percent escapes
    as the client sent them). Responses are just plausible enough for the SDK
    to parse.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.requests: list[tuple[str, str, bytes]] = []
        self.overrides: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.overrides.clear()

    def record(self, method: str, path: str, body: bytes) -> None:
        with self._lock:
            self.requests.append((method, path, body))

    def seen(self) -> list[tuple[str, str]]:
        """``(method, raw path)`` of every request received, in order."""
        with self._lock:
            return [(method, path) for method, path, _ in self.requests]

    def body(self, index: int = -1) -> dict[str, Any]:
        with self._lock:
            content = self.requests[index][2]
        return json.loads(content) if content else {}

    def respond(self, method: str, raw_path: str) -> tuple[int, str, bytes]:
        path = urlsplit(raw_path).path
        override = self.overrides.get((method, path))
        if override is not None:
            return 200, "application/json", json.dumps(override).encode()
        segment = path.rsplit("/", 1)[-1]
        if method == "DELETE":
            return 204, "application/json", b""
        if method == "POST" and path.endswith(":retrieve"):
            return 200, "application/x-ndjson", b""
        if method == "POST" and path == "/v1/memories:batchCreate":
            payload: Any = {
                "results": [{"success": True, "memory": memory_json(CREATED_MEMORY)}]
            }
            return 200, "application/json", json.dumps(payload).encode()
        if method == "POST" and path == "/v1/memories":
            return (
                200,
                "application/json",
                json.dumps(memory_json(CREATED_MEMORY)).encode(),
            )
        if method == "POST" and path == "/v1/spaces":
            payload = space_json(CREATED_SPACE, "created")
            return 200, "application/json", json.dumps(payload).encode()
        if method == "GET" and re.fullmatch(r"/v1/spaces/[^/]*/memories", path):
            return 200, "application/json", b'{"memories": []}'
        if method in ("GET", "PUT") and re.fullmatch(r"/v1/spaces/[^/]*", path):
            payload = space_json(segment, "space")
            return 200, "application/json", json.dumps(payload).encode()
        if method == "GET" and re.fullmatch(r"/v1/memories/[^/]*", path):
            return 200, "application/json", json.dumps(memory_json(segment)).encode()
        error = {"error": f"unrouted {method} {raw_path}"}
        return 404, "application/json", json.dumps(error).encode()


class _RecordingHandler(BaseHTTPRequestHandler):
    server: RecordingServer

    def _handle(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.record(self.command, self.path, body)
        status, content_type, payload = self.server.respond(self.command, self.path)
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    # http.server dispatches on these exact names.
    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: Any) -> None:
        pass


CREATED_MEMORY = "7c1d4e2a-5b6f-4a3c-9d8e-0f1a2b3c4d5e"
CREATED_SPACE = "0b9a8c7d-6e5f-4a1b-8c2d-3e4f5a6b7c8d"


@pytest.fixture(scope="session")
def _shared_http_server() -> Iterator[RecordingServer]:
    # One server per session: shutdown() waits out serve_forever's poll
    # interval, which per test would dominate the suite's run time.
    server = RecordingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def http_server(_shared_http_server: RecordingServer) -> RecordingServer:
    """The recording server, with no requests recorded and no overrides."""
    _shared_http_server.reset()
    return _shared_http_server
