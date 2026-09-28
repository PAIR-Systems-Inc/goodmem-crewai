"""Live tests against a running GoodMem server.

Opt in with GOODMEM_BASE_URL. Every space created here is registered for
deletion the moment it is created, deletion is verified, and teardown failures
are reported rather than swallowed.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from crewai.tools.tool_failure import ToolFailure
import pytest

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemGetMemoryTool,
    GoodMemKnowledgeStorage,
    GoodMemListEmbeddersTool,
    GoodMemListRerankersTool,
    GoodMemListSpacesTool,
    GoodMemSearchTool,
    GoodMemUpdateSpaceTool,
    GoodMemUploadFileTool,
)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("GOODMEM_BASE_URL"),
        reason="GOODMEM_BASE_URL not set",
    ),
]

EMBEDDER = os.environ.get("GOODMEM_EMBEDDER_ID", "")


def _verify() -> bool:
    return os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"


@pytest.fixture
def conn() -> dict:
    return {"verify_ssl": _verify()}


@pytest.fixture
def space(conn):
    """A real space, deleted and verified gone afterwards."""
    from goodmem import Goodmem

    client = Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        verify=_verify(),
    )
    created = client.spaces.create(
        name=f"crewai-e2e-{uuid.uuid4().hex[:10]}",
        space_embedders=[{"embedderId": EMBEDDER}],
    )
    space_id = created.space_id
    try:
        yield space_id
    finally:
        client.spaces.delete(id=space_id)
        remaining = [s.space_id for s in client.spaces.list(max_items=1000)]
        assert space_id not in remaining, f"cleanup failed: {space_id} still present"
        client.close()


@pytest.fixture
def indexed_space(space, conn):
    tool = GoodMemCreateMemoryTool(space_id=space, **conn)
    result = tool._run(
        text_content="The internal audit codeword is ZEPHYR-7. Revenue rose in the north.",
        metadata={"title": "audit note", "category": "audit"},
    )
    assert not isinstance(result, ToolFailure), result
    assert json.loads(result)["status"] == "COMPLETED", "create waits for indexing"
    return space


def test_update_space_succeeds_without_public_read(space, conn):
    result = GoodMemUpdateSpaceTool(**conn)._run(space_id=space, name="renamed-by-e2e")
    assert not isinstance(result, ToolFailure), result
    assert json.loads(result)["name"] == "renamed-by-e2e"


def test_create_waits_so_search_finds_it_immediately(indexed_space, conn):
    """0.1.1 polled the search for up to 10s hoping indexing would finish."""
    started = time.monotonic()
    result = GoodMemSearchTool(space_ids=[indexed_space], **conn)._run(
        query="audit codeword"
    )
    elapsed = time.monotonic() - started
    payload = json.loads(result)

    assert payload["total_results"] >= 1
    assert "ZEPHYR-7" in payload["results"][0]["chunk_text"]
    assert elapsed < 5, f"search should be one request, took {elapsed:.1f}s"


def test_empty_space_answers_immediately(space, conn):
    """0.1.1 spent 12.2s polling a space that was simply empty."""
    started = time.monotonic()
    payload = json.loads(
        GoodMemSearchTool(space_ids=[space], **conn)._run(query="anything")
    )
    elapsed = time.monotonic() - started

    assert payload["total_results"] == 0
    assert payload["partial"] is False
    assert elapsed < 5, f"empty is an answer; took {elapsed:.1f}s"


def test_invalid_reranker_is_reported_not_hidden(indexed_space, conn):
    """0.1.1 returned success with unreranked chunks and no mention of failure."""
    result = GoodMemSearchTool(
        space_ids=[indexed_space],
        reranker_id="00000000-0000-0000-0000-000000000000",
        **conn,
    )._run(query="audit codeword")

    if isinstance(result, ToolFailure):
        assert "RERANKING_FAILED" in result.message or "NOT_FOUND" in result.message
    else:
        payload = json.loads(result)
        assert payload["partial"] is True, "reranking failed; results must be flagged"
        codes = {s.get("code") for s in payload["statuses"]}
        assert codes & {"RERANKING_FAILED", "NOT_FOUND"}, codes
        # The server falls back to vector hits; they are labelled by what
        # happened, not by the reranker that was configured.
        assert {h["score_kind"] for h in payload["results"]} <= {"vector"}


def test_knowledge_storage_keeps_fallback_hits_when_the_reranker_fails(
    indexed_space, conn
):
    """0.2.1 thresholded the fallback's vector scores at CrewAI's default 0.6
    as if a reranker had produced them, and returned []."""
    storage = GoodMemKnowledgeStorage(
        space_id=indexed_space,
        reranker_id="00000000-0000-0000-0000-000000000000",
        **conn,
    )
    results = storage.search(["audit codeword"])  # default score_threshold
    assert results, "the server's fallback hits must not be discarded"
    for result in results:
        assert result["metadata"]["score_kind"] == "vector"
        assert result["metadata"]["goodmem_partial"] is True
        assert result["score"] == -result["metadata"]["raw_score"]


def test_boolean_metadata_filter_matches_a_stored_boolean(space, conn):
    """0.2.1 compared True as the text 'True', which matched nothing."""
    created = GoodMemCreateMemoryTool(space_id=space, **conn)._run(
        text_content="Flagged note about the audit codeword.",
        metadata={"flag": True},
    )
    assert not isinstance(created, ToolFailure), created

    def total(value: bool) -> int:
        result = GoodMemSearchTool(
            space_ids=[space], metadata_filter={"flag": value}, **conn
        )._run(query="audit codeword")
        assert not isinstance(result, ToolFailure), result
        return int(json.loads(result)["total_results"])

    assert total(True) >= 1
    assert total(False) == 0


def test_list_embedders_and_rerankers(conn):
    """0.2.1 failed both on every call: the SDK's list() takes no max_items."""
    embedders = GoodMemListEmbeddersTool(**conn)._run()
    assert not isinstance(embedders, ToolFailure), embedders
    ids = {e["embedder_id"] for e in json.loads(embedders)["embedders"]}
    if EMBEDDER:
        assert EMBEDDER in ids
    rerankers = GoodMemListRerankersTool(**conn)._run()
    assert not isinstance(rerankers, ToolFailure), rerankers


def test_metadata_filter_is_applied_server_side(indexed_space, conn):
    hit = json.loads(
        GoodMemSearchTool(
            space_ids=[indexed_space], metadata_filter={"category": "audit"}, **conn
        )._run(query="codeword")
    )
    miss = json.loads(
        GoodMemSearchTool(
            space_ids=[indexed_space],
            metadata_filter={"category": "nonexistent"},
            **conn,
        )._run(query="codeword")
    )
    assert hit["total_results"] >= 1
    assert miss["total_results"] == 0


def test_filter_value_with_an_apostrophe_is_escaped(indexed_space, conn):
    """Live proof that the escaping the grammar accepts is the one we emit."""
    result = GoodMemSearchTool(
        space_ids=[indexed_space], metadata_filter={"category": "o'brien"}, **conn
    )._run(query="codeword")
    assert not isinstance(result, ToolFailure), result
    assert json.loads(result)["total_results"] == 0


def test_get_memory_returns_readable_text(indexed_space, conn):
    listed = json.loads(
        GoodMemSearchTool(space_ids=[indexed_space], **conn)._run(query="codeword")
    )
    memory_id = listed["results"][0]["memory_id"]
    payload = json.loads(GoodMemGetMemoryTool(**conn)._run(memory_id=memory_id))
    assert "ZEPHYR-7" in payload["content"]


def test_upload_refuses_a_path_outside_the_upload_dir(space, conn, tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    (allowed / "note.txt").write_text("Uploaded through the approved directory.")

    tool = GoodMemUploadFileTool(space_id=space, upload_dir=str(allowed), **conn)
    refused = tool._run(file_name="/etc/passwd")
    assert isinstance(refused, ToolFailure)

    accepted = tool._run(file_name="note.txt")
    assert not isinstance(accepted, ToolFailure), accepted
    assert json.loads(accepted)["status"] == "COMPLETED"


def test_list_spaces_reports_what_it_returned(conn):
    payload = json.loads(GoodMemListSpacesTool(**conn)._run())
    assert "returned" in payload and "truncated" in payload


def test_knowledge_storage_round_trip(space, conn):
    """The native CrewAI surface: save through it, search through it."""
    storage = GoodMemKnowledgeStorage(space_id=space, **conn)
    storage.save(["Ada Lovelace wrote the first published algorithm for a machine."])

    results = storage.search(["who wrote the first algorithm"], limit=3)
    assert results, "knowledge storage found nothing it had just saved"
    assert "Ada Lovelace" in results[0]["content"]
    assert results[0]["metadata"]["score_kind"] == "vector"
    # The live server's vector score is a negative inner product; CrewAI's
    # SearchResult.score is higher-is-better, so the sign is flipped and the
    # server value kept alongside.
    assert results[0]["metadata"]["raw_score"] < 0
    assert results[0]["score"] == -results[0]["metadata"]["raw_score"]
    assert results[0]["id"], "a stable chunk id is required"


def test_knowledge_storage_reset_requires_opt_in(space, conn):
    storage = GoodMemKnowledgeStorage(space_id=space, **conn)
    with pytest.raises(PermissionError, match="allow_reset"):
        storage.reset()
