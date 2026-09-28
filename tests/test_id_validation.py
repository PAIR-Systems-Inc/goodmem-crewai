"""An id is refused before it can reach a URL path.

The ``goodmem`` SDK builds paths as ``f"/v1/memories/{id}"`` with the id
unescaped, and httpx resolves dot segments before sending. Against 0.2.0,
``GoodMemDeleteMemoryTool(memory_id="../spaces/<id>")`` sent
``DELETE /v1/spaces/<id>`` and reported ``{"deleted": true}``; the GoodMem
server also decodes ``%2e%2e`` into a traversal, so neither client-side
encoding nor the server can be relied on. Every GoodMem id is a UUID, so
anything else is refused and no request is made.

These tests talk to a real local HTTP server through the SDK client the
integration builds for itself, so what the server records is exactly what
would have gone over the wire. Nothing in the integration or the SDK is
mocked. The UUID pattern is spelled out here rather than imported, so the
file also runs, and fails, against code that predates the check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
from crewai.utilities.agent_utils import convert_tools_to_openai_schema
import pytest

from crewai_goodmem import (
    GoodMemConnection,
    GoodMemCreateMemoryTool,
    GoodMemCreateSpaceTool,
    GoodMemDeleteMemoryTool,
    GoodMemDeleteSpaceTool,
    GoodMemGetMemoryTool,
    GoodMemGetSpaceTool,
    GoodMemKnowledgeStorage,
    GoodMemListMemoriesTool,
    GoodMemSearchTool,
    GoodMemUpdateSpaceTool,
    GoodMemUploadFileTool,
    wait_for_memories,
)

from .conftest import RecordingServer, memory_json


TARGET = "3f2b6c1e-9a4d-4e8b-b1c2-7d5e6f8a9b0c"
# A second valid id, for entry points where the payload goes in one of two slots.
OTHER = "5d0c9b8a-7f6e-4d3c-a2b1-c0d9e8f7a6b5"

UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

PAYLOADS = [
    f"../spaces/{TARGET}",
    f"a/../../spaces/{TARGET}",
    f"%2e%2e/spaces/{TARGET}",
    f"..%2Fspaces%2F{TARGET}",
    f"{TARGET}/../../spaces/{TARGET}",
    "",
    f" {TARGET}",
    f"{TARGET}?x=1",
    f"{TARGET}#frag",
    # Python's `$` also matches just before a trailing newline.
    f"{TARGET}\n",
]


@dataclass(frozen=True)
class Ctx:
    conn: dict[str, Any]
    upload_dir: str


@dataclass(frozen=True)
class EntryPoint:
    """One place an id enters the package.

    ``tool`` entry points refuse with a ``ToolFailure``; developer-facing ones
    raise ``ValueError``, as they already do for other configuration errors.
    """

    name: str
    field: str
    call: Callable[[Ctx, str], Any]
    method: str
    path: str
    tool: bool


def _storage(ctx: Ctx, **kwargs: Any) -> GoodMemKnowledgeStorage:
    return GoodMemKnowledgeStorage(**ctx.conn, **kwargs)


ENTRY_POINTS = [
    # ---- ids the model chooses: each one is a path segment
    EntryPoint(
        "GoodMemGetSpaceTool",
        "space_id",
        lambda ctx, v: GoodMemGetSpaceTool(**ctx.conn).run(space_id=v),
        "GET",
        "/v1/spaces/{id}",
        tool=True,
    ),
    EntryPoint(
        "GoodMemUpdateSpaceTool",
        "space_id",
        lambda ctx, v: GoodMemUpdateSpaceTool(**ctx.conn).run(space_id=v, name="n"),
        "PUT",
        "/v1/spaces/{id}",
        tool=True,
    ),
    EntryPoint(
        "GoodMemDeleteSpaceTool",
        "space_id",
        lambda ctx, v: GoodMemDeleteSpaceTool(**ctx.conn).run(space_id=v),
        "DELETE",
        "/v1/spaces/{id}",
        tool=True,
    ),
    EntryPoint(
        "GoodMemListMemoriesTool",
        "space_id",
        lambda ctx, v: GoodMemListMemoriesTool(**ctx.conn).run(space_id=v),
        "GET",
        "/v1/spaces/{id}/memories",
        tool=True,
    ),
    EntryPoint(
        "GoodMemGetMemoryTool",
        "memory_id",
        lambda ctx, v: GoodMemGetMemoryTool(**ctx.conn).run(memory_id=v),
        "GET",
        "/v1/memories/{id}",
        tool=True,
    ),
    EntryPoint(
        "GoodMemDeleteMemoryTool",
        "memory_id",
        lambda ctx, v: GoodMemDeleteMemoryTool(**ctx.conn).run(memory_id=v),
        "DELETE",
        "/v1/memories/{id}",
        tool=True,
    ),
    # ---- ids a developer passes or configures that become a path segment
    EntryPoint(
        "wait_for_memories",
        "memory_ids",
        lambda ctx, v: wait_for_memories(
            [v], connection=GoodMemConnection(**ctx.conn), timeout=0.0
        ),
        "GET",
        "/v1/memories/{id}",
        tool=False,
    ),
    EntryPoint(
        "GoodMemKnowledgeStorage.reset",
        "space_id",
        lambda ctx, v: _storage(ctx, space_id=v, allow_reset=True).reset(),
        "GET",
        "/v1/spaces/{id}/memories",
        tool=False,
    ),
    # ---- configured ids sent in a request body: no traversal, checked alike
    EntryPoint(
        "GoodMemSearchTool.space_ids",
        "space_ids",
        lambda ctx, v: GoodMemSearchTool(space_ids=[v], **ctx.conn).run(query="q"),
        "POST",
        "/v1/memories:retrieve",
        tool=True,
    ),
    EntryPoint(
        "GoodMemSearchTool.reranker_id",
        "reranker_id",
        lambda ctx, v: GoodMemSearchTool(
            space_ids=[OTHER], reranker_id=v, **ctx.conn
        ).run(query="q"),
        "POST",
        "/v1/memories:retrieve",
        tool=True,
    ),
    EntryPoint(
        "GoodMemCreateSpaceTool.embedder_id",
        "embedder_id",
        lambda ctx, v: GoodMemCreateSpaceTool(embedder_id=v, **ctx.conn).run(name="n"),
        "POST",
        "/v1/spaces",
        tool=True,
    ),
    EntryPoint(
        "GoodMemCreateMemoryTool.space_id",
        "space_id",
        lambda ctx, v: GoodMemCreateMemoryTool(space_id=v, wait=False, **ctx.conn).run(
            text_content="t"
        ),
        "POST",
        "/v1/memories",
        tool=True,
    ),
    EntryPoint(
        "GoodMemUploadFileTool.space_id",
        "space_id",
        lambda ctx, v: GoodMemUploadFileTool(
            space_id=v, upload_dir=ctx.upload_dir, wait=False, **ctx.conn
        ).run(file_name="note.txt"),
        "POST",
        "/v1/memories",
        tool=True,
    ),
    EntryPoint(
        "GoodMemKnowledgeStorage.search.space_id",
        "space_id",
        lambda ctx, v: _storage(ctx, space_id=v).search(["q"], score_threshold=0.0),
        "POST",
        "/v1/memories:retrieve",
        tool=False,
    ),
    EntryPoint(
        "GoodMemKnowledgeStorage.search.space_ids",
        "space_ids",
        lambda ctx, v: _storage(ctx, space_id=OTHER, space_ids=[v]).search(
            ["q"], score_threshold=0.0
        ),
        "POST",
        "/v1/memories:retrieve",
        tool=False,
    ),
    EntryPoint(
        "GoodMemKnowledgeStorage.search.reranker_id",
        "reranker_id",
        lambda ctx, v: _storage(ctx, space_id=OTHER, reranker_id=v).search(
            ["q"], score_threshold=0.0
        ),
        "POST",
        "/v1/memories:retrieve",
        tool=False,
    ),
    EntryPoint(
        "GoodMemKnowledgeStorage.save",
        "space_id",
        lambda ctx, v: _storage(ctx, space_id=v, wait_for_indexing=False).save(["d"]),
        "POST",
        "/v1/memories:batchCreate",
        tool=False,
    ),
]

MODEL_FACING = [
    (GoodMemGetSpaceTool, "space_id"),
    (GoodMemUpdateSpaceTool, "space_id"),
    (GoodMemDeleteSpaceTool, "space_id"),
    (GoodMemListMemoriesTool, "space_id"),
    (GoodMemGetMemoryTool, "memory_id"),
    (GoodMemDeleteMemoryTool, "memory_id"),
]


@pytest.fixture
def ctx(http_server: RecordingServer, tmp_path: Path) -> Ctx:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "note.txt").write_text("a note")
    return Ctx(
        conn={"base_url": http_server.base_url, "api_key": "test-key"},
        upload_dir=str(upload_dir),
    )


def _outcome(entry: EntryPoint, ctx: Ctx, value: str) -> Any:
    """What the entry point returned, or the exception it raised."""
    try:
        return entry.call(ctx, value)
    except Exception as exc:
        return exc


def _assert_refused(entry: EntryPoint, outcome: Any) -> None:
    if entry.tool:
        assert isinstance(outcome, ToolFailure), f"not refused: {outcome!r}"
        assert outcome.reason is ToolFailureReason.INVALID_INPUT
        assert outcome.code == "invalid_input"
        assert outcome.retryable is False
        message = outcome.message
    else:
        assert isinstance(outcome, ValueError), f"not refused: {outcome!r}"
        message = str(outcome)
    assert entry.field in message, message
    assert "must be a UUID" in message, message


def _assert_succeeded(outcome: Any) -> None:
    assert not isinstance(outcome, (ToolFailure, Exception)), outcome


@pytest.mark.parametrize("payload", PAYLOADS, ids=repr)
@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda e: e.name)
def test_a_non_uuid_id_is_refused_before_any_request(
    entry: EntryPoint, payload: str, ctx: Ctx, http_server: RecordingServer
) -> None:
    outcome = _outcome(entry, ctx, payload)

    seen = http_server.seen()
    assert seen == [], f"{entry.name}({payload!r}) reached the server as {seen}"
    _assert_refused(entry, outcome)


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda e: e.name)
def test_a_valid_uuid_reaches_exactly_the_intended_path(
    entry: EntryPoint, ctx: Ctx, http_server: RecordingServer
) -> None:
    outcome = _outcome(entry, ctx, TARGET)

    _assert_succeeded(outcome)
    seen = [(method, urlsplit(path).path) for method, path in http_server.seen()]
    assert seen == [(entry.method, entry.path.format(id=TARGET))]
    raw_path = http_server.seen()[0][1]
    assert TARGET in raw_path or TARGET in http_server.requests[0][2].decode()


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda e: e.name)
def test_an_uppercase_uuid_is_sent_in_canonical_lowercase(
    entry: EntryPoint, ctx: Ctx, http_server: RecordingServer
) -> None:
    outcome = _outcome(entry, ctx, TARGET.upper())

    _assert_succeeded(outcome)
    method, raw_path = http_server.seen()[0]
    body = http_server.requests[0][2].decode()
    assert (method, urlsplit(raw_path).path) == (
        entry.method,
        entry.path.format(id=TARGET),
    )
    assert TARGET in raw_path or TARGET in body
    assert TARGET.upper() not in raw_path + body


def test_the_reported_space_deletion_through_delete_memory_is_refused(
    ctx: Ctx, http_server: RecordingServer
) -> None:
    """The external report: on 0.2.0 this deleted the whole space and the tool
    answered {"deleted": true, "memory_id": "../spaces/<id>"}."""
    result = GoodMemDeleteMemoryTool(**ctx.conn).run(memory_id=f"../spaces/{TARGET}")

    assert http_server.seen() == []
    assert isinstance(result, ToolFailure)
    assert result.reason is ToolFailureReason.INVALID_INPUT
    assert "memory_id must be a UUID" in result.message
    assert '"deleted": true' not in result.as_agent_message()


@pytest.mark.parametrize(
    ("tool_cls", "field"), MODEL_FACING, ids=lambda x: getattr(x, "__name__", x)
)
def test_the_model_is_told_ids_are_uuids(tool_cls: type, field: str) -> None:
    """Checked against the function schema CrewAI actually sends the LLM.

    The pattern is declared for the model but not enforced by pydantic: CrewAI
    validates arguments before ``_run`` and records a schema failure as a bare
    exception, while the tool's own check reports INVALID_INPUT. The refusal
    tests above call ``run()``, so they would fail if the schema pre-empted it.
    """
    tool = tool_cls(base_url="http://unused.invalid", api_key="k")
    schemas, _, _ = convert_tools_to_openai_schema([tool])
    prop = schemas[0]["function"]["parameters"]["properties"][field]
    assert prop["type"] == "string"
    assert prop["pattern"] == UUID_PATTERN
    assert "UUID" in prop["description"]


def test_reset_checks_every_listed_id_before_deleting_any(
    ctx: Ctx, http_server: RecordingServer
) -> None:
    """Ids the server lists are path segments too. One that is not a UUID
    stops the reset before anything is deleted, rather than halfway."""
    http_server.overrides[("GET", f"/v1/spaces/{TARGET}/memories")] = {
        "memories": [memory_json(OTHER), memory_json(f"../spaces/{TARGET}")]
    }
    storage = GoodMemKnowledgeStorage(space_id=TARGET, allow_reset=True, **ctx.conn)

    try:
        outcome: Any = storage.reset()
    except Exception as exc:
        outcome = exc

    seen = http_server.seen()
    assert [method for method, _ in seen] == ["GET"], f"reset sent {seen}"
    assert isinstance(outcome, ValueError)
    assert "memory_id must be a UUID" in str(outcome)
