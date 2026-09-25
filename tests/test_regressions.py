"""Regression tests. Each one pins a defect reproduced against 0.1.1.

The reproduction class is named in each docstring: "live" means it was
reproduced against a running GoodMem server, "mock" against a local HTTP
server, "local" with no server at all.
"""

from __future__ import annotations

import json

from crewai.tools.tool_failure import ToolFailure
import httpx
import pytest

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemGetMemoryTool,
    GoodMemKnowledgeStorage,
    GoodMemListSpacesTool,
    GoodMemSearchTool,
    GoodMemUpdateSpaceTool,
    GoodMemUploadFileTool,
)
from crewai_goodmem.filters import from_mapping, text_equals

from .conftest import (
    REAL_VECTOR_SCORE,
    chunk_event,
    memory_event,
    memory_json,
    ndjson,
    space_json,
    status_event,
)


# GoodMem ids are UUIDs, and the integration refuses anything else before a
# request is made (see test_id_validation.py), so configured ids are real ones.
SPACE = "01a0ace4-678d-7459-aa91-b6ccd46d97d8"
RERANKER = "019cfd1c-c033-7517-b7de-f73941a0464c"
MEMORY = "019d2a7e-44c1-7b3a-9e0f-5a1b2c3d4e5f"
PDF_MEMORY = "019d2a7e-44c1-7b3a-9e0f-5a1b2c3d4e60"


# ---------------------------------------------------------------- P2
def test_update_space_cannot_send_public_read():
    """live: PUT with publicRead returned HTTP 400 'Unrecognized field'."""
    schema = GoodMemUpdateSpaceTool.model_fields["args_schema"].default
    assert "public_read" not in schema.model_fields
    assert set(schema.model_fields) == {
        "space_id",
        "name",
        "merge_labels",
        "replace_labels",
    }


def test_update_space_sends_only_known_fields(client, recorder):
    recorder.route(
        "PUT", f"/spaces/{SPACE}", httpx.Response(200, json=space_json(SPACE, "n"))
    )
    result = GoodMemUpdateSpaceTool(client=client)._run(space_id=SPACE, name="n")
    assert not isinstance(result, ToolFailure), result
    body = recorder.body()
    assert "publicRead" not in body
    assert set(body) <= {
        "name",
        "mergeLabels",
        "replaceLabels",
        "defaultChunkingConfig",
    }


# ---------------------------------------------------------------- P4
def test_failed_reranking_is_not_reported_as_success(client, recorder):
    """live: server sent NOT_FOUND + RERANKING_FAILED; 0.1.1 returned success
    with unreranked fallback chunks and no mention of the failure."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1", {"title": "doc"}),
            chunk_event("c1", "fallback text", "m1"),
            status_event("RERANKING_FAILED", "Failed to create reranker client"),
        ),
    )
    tool = GoodMemSearchTool(client=client, space_ids=[SPACE], reranker_id=RERANKER)
    payload = json.loads(tool._run(query="anything"))

    assert payload["partial"] is True, "degraded retrieval must be flagged"
    assert any(s["code"] == "RERANKING_FAILED" for s in payload["statuses"])
    assert payload["total_results"] == 1, "usable chunks are still returned"


def test_total_failure_is_empty_and_flagged_not_a_tool_failure(client, recorder):
    """mock: VECTOR_SEARCH_FAILED with no usable chunk.

    Contract Q4b: empty results with partial=True and the statuses, so the
    model can tell a failed search from a miss -- and not a ToolFailure,
    which CrewAI would turn into a retry or an abort.
    """
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(status_event("VECTOR_SEARCH_FAILED", "search backend unavailable")),
    )
    tool = GoodMemSearchTool(client=client, space_ids=[SPACE])
    result = tool._run(query="anything")

    assert not isinstance(result, ToolFailure)
    payload = json.loads(result)
    assert payload["results"] == []
    assert payload["partial"] is True
    assert payload["statuses"][0]["code"] == "VECTOR_SEARCH_FAILED"


def test_knowledge_storage_total_failure_warns_and_returns_empty(client, recorder):
    """Contract Q4b for a bare-list return: nothing to hang a flag on, so the
    failure surfaces as a warning and a log line instead of an exception."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(status_event("SPACE_NOT_FOUND", "space gone")),
    )
    storage = GoodMemKnowledgeStorage(client=client, space_id=SPACE)
    with pytest.warns(UserWarning, match="SPACE_NOT_FOUND"):
        results = storage.search(["q"], limit=5, score_threshold=0.0)
    assert results == []


def test_feature_disabled_is_informational_whatever_its_details(client, recorder):
    """Contract Q1: the code alone decides. The server defines FEATURE_DISABLED
    as 'disabled due to missing configuration', so no details check."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "text", "m1"),
            status_event(
                "FEATURE_DISABLED",
                "Reranking disabled: no reranker configured.",
                feature="reranking",
                required_param="reranker_id",
            ),
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    )
    assert payload["partial"] is False
    assert "statuses" not in payload
    assert payload["total_results"] == 1


def test_informational_notice_is_not_a_failure(client, recorder):
    """The 'no LLM configured' notice is normal and must not mark results partial."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "text", "m1"),
            status_event(
                "FEATURE_DISABLED",
                "Abstract reply generation disabled",
                feature="summarization",
                required_param="llm_id",
            ),
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    )
    assert payload["partial"] is False
    assert "statuses" not in payload


# ---------------------------------------------------------------- P3
def test_unknown_future_status_code_does_not_discard_results(client, recorder):
    """A code from a newer server decodes to None. It must be surfaced, must
    not raise, and must not throw away the chunks that did come back."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "still useful", "m1"),
            status_event("SOME_CODE_FROM_A_NEWER_SERVER", "hello from the future"),
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    )

    assert payload["total_results"] == 1, "unknown code must not discard chunks"
    assert payload["partial"] is True, "but it is not silently ignored either"
    assert payload["statuses"][0]["code"] == "UNKNOWN"
    assert payload["statuses"][0]["unrecognized"] is True


# ---------------------------------------------------------------- P5
def test_search_makes_exactly_one_request(client, recorder):
    """live: an empty space cost 12.2s of polling in 0.1.1 vs 0.3s without."""
    recorder.route("POST", ":retrieve", ndjson())
    tool = GoodMemSearchTool(client=client, space_ids=[SPACE])
    payload = json.loads(tool._run(query="nothing here"))

    assert payload["total_results"] == 0
    assert payload["partial"] is False
    retrieves = [p for p in recorder.paths() if p.endswith(":retrieve")]
    assert len(retrieves) == 1, "an empty result is an answer, not a reason to poll"


def test_search_schema_has_no_polling_knobs():
    schema = GoodMemSearchTool.model_fields["args_schema"].default
    for gone in ("wait_for_indexing", "max_wait_seconds", "poll_interval"):
        assert gone not in schema.model_fields


# ---------------------------------------------------------------- P6
def test_list_spaces_follows_pagination(client, recorder):
    """mock: 0.1.1 read page 1 and reported 50 as the total."""
    page1 = [space_json(f"s{i}", f"space-{i}") for i in range(50)]
    page2 = [space_json("s50", "space-50")]

    def respond(request: httpx.Request) -> httpx.Response:
        # The SDK requests the next page as next_token=<token>.
        if "next_token=" in str(request.url):
            return httpx.Response(200, json={"spaces": page2})
        return httpx.Response(200, json={"spaces": page1, "nextToken": "PAGE2"})

    recorder.route("GET", "/v1/spaces", respond)
    payload = json.loads(GoodMemListSpacesTool(client=client)._run())

    assert payload["returned"] == 51, "the space on page 2 must not be invisible"
    assert len([p for p in recorder.paths() if p.endswith("/v1/spaces")]) == 2


# ---------------------------------------------------------------- P10
def test_upload_refuses_paths_outside_the_configured_directory(tmp_path):
    """local: 0.1.1 read /etc/passwd and ~/.goodmem/config.toml into an upload."""
    allowed = tmp_path / "uploads"
    allowed.mkdir()
    (allowed / "ok.txt").write_text("fine")
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")

    tool = GoodMemUploadFileTool(
        space_id=SPACE, upload_dir=str(allowed), base_url="https://x", api_key="k"
    )

    for escape in ("../secret.txt", str(secret), "/etc/passwd"):
        result = tool._run(file_name=escape)
        assert isinstance(result, ToolFailure), f"{escape} should be refused"
        assert result.code == "invalid_input"


def test_upload_refuses_symlink_escape(tmp_path):
    allowed = tmp_path / "uploads"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    (allowed / "link.txt").symlink_to(secret)

    tool = GoodMemUploadFileTool(
        space_id=SPACE, upload_dir=str(allowed), base_url="https://x", api_key="k"
    )
    result = tool._run(file_name="link.txt")
    assert isinstance(result, ToolFailure)


def test_upload_is_disabled_until_a_directory_is_configured(tmp_path):
    tool = GoodMemUploadFileTool(space_id=SPACE, base_url="https://x", api_key="k")
    result = tool._run(file_name="anything.txt")
    assert isinstance(result, ToolFailure)
    assert "upload_dir" in result.message


def test_create_memory_takes_no_file_path():
    schema = GoodMemCreateMemoryTool.model_fields["args_schema"].default
    assert "file_path" not in schema.model_fields


# ---------------------------------------------------------------- P21
def test_api_key_is_not_serialized():
    """local: 0.1.1 exposed the key in model_dump() and repr()."""
    tool = GoodMemSearchTool(
        space_ids=[SPACE], base_url="https://x", api_key="sk-SECRET"
    )
    assert "sk-SECRET" not in json.dumps(tool.model_dump(), default=str)
    assert "sk-SECRET" not in repr(tool)
    assert tool.api_key.get_secret_value() == "sk-SECRET"


# ---------------------------------------------------------------- P28
def test_only_the_query_is_model_visible():
    """0.1.1 exposed 12 arguments, including poll_interval and llm_temperature."""
    schema = GoodMemSearchTool.model_fields["args_schema"].default
    assert list(schema.model_fields) == ["query"]


def test_developer_configuration_is_not_model_reachable(client, recorder):
    recorder.route("POST", ":retrieve", ndjson())
    tool = GoodMemSearchTool(
        client=client,
        space_ids=[SPACE],
        filter="CAST(val('$.a') AS TEXT) = 'b'",
    )
    tool._run(query="q")
    body = recorder.body()
    assert body["spaceKeys"][0]["spaceId"] == SPACE
    assert body["spaceKeys"][0]["filter"] == "CAST(val('$.a') AS TEXT) = 'b'"


# ---------------------------------------------------------------- P29
def test_real_vector_scores_are_presented_higher_is_better(client, recorder):
    """live capture: a real vector relevanceScore is -0.5345.

    CrewAI documents score as 'higher is better, typically 0-1'. The server's
    vector score is a negative inner product (best match = lowest number), so
    it is negated to honour the host convention, and the raw value is kept in
    metadata. It is still not 0-1: CrewAI's default score_threshold of 0.6
    applied to it would be arbitrary, so the threshold is only applied when a
    reranker produced the scores.
    """
    recorder.route(
        "POST", ":retrieve", ndjson(memory_event("m1"), chunk_event("c1", "text", "m1"))
    )
    storage = GoodMemKnowledgeStorage(client=client, space_id=SPACE)
    with pytest.warns(UserWarning, match="score_threshold is ignored"):
        results = storage.search(["q"], limit=5, score_threshold=0.6)

    assert len(results) == 1, "a negative vector score is not a reason to drop a hit"
    assert results[0]["score"] == -REAL_VECTOR_SCORE
    assert results[0]["score"] > 0
    assert results[0]["metadata"]["raw_score"] == REAL_VECTOR_SCORE
    assert results[0]["metadata"]["score_kind"] == "vector"


def test_vector_negation_keeps_server_order_sortable(client, recorder):
    """Two real-shaped hits in server order: the better one has the lower raw
    score. After negation, sorting by score descending agrees with the
    server, which is what a caller reading SearchResult.score expects."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "best", "m1", score=-0.6154),
            chunk_event("c2", "worse", "m1", score=-0.3873),
        ),
    )
    storage = GoodMemKnowledgeStorage(client=client, space_id=SPACE)
    with pytest.warns(UserWarning):
        results = storage.search(["q"], limit=5, score_threshold=0.6)
    assert [r["content"] for r in results] == ["best", "worse"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r["metadata"]["raw_score"] for r in results] == [-0.6154, -0.3873]


def test_reranker_scores_are_not_negated(client, recorder):
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(memory_event("m1"), chunk_event("c1", "strong", "m1", score=0.91)),
    )
    storage = GoodMemKnowledgeStorage(
        client=client, space_id=SPACE, reranker_id=RERANKER
    )
    results = storage.search(["q"], limit=5, score_threshold=0.0)
    assert results[0]["score"] == 0.91
    assert results[0]["metadata"]["raw_score"] == 0.91


def test_threshold_applies_when_a_reranker_produced_the_scores(client, recorder):
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "strong", "m1", score=0.91),
            chunk_event("c2", "weak", "m1", score=0.10),
        ),
    )
    storage = GoodMemKnowledgeStorage(
        client=client, space_id=SPACE, reranker_id=RERANKER
    )
    results = storage.search(["q"], limit=5, score_threshold=0.6)

    assert [r["id"] for r in results] == ["c1"]
    assert results[0]["metadata"]["score_kind"] == "reranker"


# ---------------------------------------------------------------- P24 / P30
def test_two_chunks_of_one_memory_are_two_results(client, recorder):
    """Deduplicating by memoryId would drop the second fact."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1", {"title": "doc"}),
            chunk_event("c1", "first fact", "m1"),
            chunk_event("c2", "second fact", "m1"),
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    )
    assert payload["total_results"] == 2
    assert {r["chunk_text"] for r in payload["results"]} == {
        "first fact",
        "second fact",
    }


def test_metadata_is_joined_even_when_the_definition_arrives_last(client, recorder):
    """0.1.1 returned chunks and memories as two unjoined arrays."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            chunk_event("c1", "text", "m1"),
            memory_event("m1", {"title": "Quarterly report", "category": "finance"}),
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    )
    assert payload["results"][0]["metadata"]["title"] == "Quarterly report"


def test_get_memory_returns_readable_text_in_one_request(client, recorder):
    """0.1.1 made a second call for content and returned success:true with a
    contentError when it failed. The SDK returns bytes, so the text also has
    to be decoded rather than handed to the agent as base64."""
    recorder.route(
        "GET",
        f"/v1/memories/{MEMORY}",
        # "aGVsbG8gd29ybGQ=" is how the server really sends "hello world".
        httpx.Response(200, json=memory_json(MEMORY, content_b64="aGVsbG8gd29ybGQ=")),
    )
    payload = json.loads(GoodMemGetMemoryTool(client=client)._run(memory_id=MEMORY))

    assert payload["content"] == "hello world", "readable text, not base64"
    assert "original_content" not in payload
    assert len(recorder.requests) == 1, "content must not cost a second request"


def test_get_memory_describes_binary_content_instead_of_dumping_it(client, recorder):
    recorder.route(
        "GET",
        f"/v1/memories/{PDF_MEMORY}",
        httpx.Response(
            200,
            json=memory_json(
                PDF_MEMORY, content_b64="JVBERi0xLjQK", content_type="application/pdf"
            ),
        ),
    )
    payload = json.loads(GoodMemGetMemoryTool(client=client)._run(memory_id=PDF_MEMORY))
    assert "content" not in payload
    assert "application/pdf" in payload["content_omitted"]


# ---------------------------------------------------------------- filters
def test_filter_values_are_escaped_the_way_the_server_accepts():
    """live: SQL '' doubling is rejected with HTTP 400; backslash works."""
    assert (
        text_equals("owner", "o'brien") == r"CAST(val('$.owner') AS TEXT) = 'o\'brien'"
    )
    assert text_equals("p", "a\\b") == r"CAST(val('$.p') AS TEXT) = 'a\\b'"


def test_filter_injection_is_quoted_not_executed():
    """live: this payload matched only its own row (1 of 5), never the others."""
    built = text_equals("tag", "x' OR '1'='1")
    assert built == r"CAST(val('$.tag') AS TEXT) = 'x\' OR \'1\'=\'1'"
    # Every quote the payload contributed is backslash-escaped, so none of them
    # can close the literal and start a new clause.
    literal = built.split(" = ", 1)[1]
    assert literal.startswith("'") and literal.endswith("'")
    assert "\\'" in literal
    assert literal.count("'") - literal.count("\\'") == 2  # only the delimiters


def test_filter_rejects_unsafe_field_names_and_control_characters():
    with pytest.raises(ValueError, match="Unsupported metadata field"):
        text_equals("bad name", "v")
    with pytest.raises(ValueError, match="control characters"):
        text_equals("ok", "line1\nline2")


def test_mapping_filter_ands_clauses():
    assert from_mapping({}) is None
    assert from_mapping({"a": "1"}) == "CAST(val('$.a') AS TEXT) = '1'"
    combined = from_mapping({"a": "1", "b": "2"})
    assert (
        combined
        == "(CAST(val('$.a') AS TEXT) = '1') AND (CAST(val('$.b') AS TEXT) = '2')"
    )


# ---------------------------------------------------------------- P22 / P33
def test_injected_client_is_used_and_not_closed(client, recorder):
    recorder.route("POST", ":retrieve", ndjson())
    tool = GoodMemSearchTool(client=client, space_ids=[SPACE])
    tool._run(query="q")
    tool._run(query="q again")
    assert len(recorder.requests) == 2, "the injected client stays usable"


def test_environment_cannot_redirect_an_injected_client(client, recorder, monkeypatch):
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://attacker.example")
    monkeypatch.setenv("GOODMEM_API_KEY", "other-key")
    recorder.route("POST", ":retrieve", ndjson())
    GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    assert recorder.requests[0].url.host == "goodmem.test"


def test_a_threshold_that_removes_every_reranked_result_says_so(client, recorder):
    """Measured live 2026-09-24: Voyage rerank-2.5 scores 0.27..0.93, Jina
    jina-reranker-v3 scores -0.14..0.43 on the same documents. CrewAI's
    default score_threshold of 0.6 keeps the top two on one and nothing on
    the other; the empty case must not look like a miss."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            memory_event("m1"),
            chunk_event("c1", "best", "m1", score=0.43),
            chunk_event("c2", "next", "m1", score=-0.14),
        ),
    )
    storage = GoodMemKnowledgeStorage(
        client=client, space_id=SPACE, reranker_id=RERANKER
    )
    with pytest.warns(UserWarning, match="removed all 2 reranked result"):
        results = storage.search(["q"], limit=5, score_threshold=0.6)
    assert results == []
