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
    recorder.route("PUT", "/spaces/s1", httpx.Response(200, json=space_json("s1", "n")))
    result = GoodMemUpdateSpaceTool(client=client)._run(space_id="s1", name="n")
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
    tool = GoodMemSearchTool(client=client, space_ids=["s1"], reranker_id="missing")
    payload = json.loads(tool._run(query="anything"))

    assert payload["partial"] is True, "degraded retrieval must be flagged"
    assert any(s["code"] == "RERANKING_FAILED" for s in payload["statuses"])
    assert payload["total_results"] == 1, "usable chunks are still returned"


def test_total_failure_becomes_a_tool_failure(client, recorder):
    """mock: VECTOR_SEARCH_FAILED with no usable chunk must not look empty."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(status_event("VECTOR_SEARCH_FAILED", "search backend unavailable")),
    )
    tool = GoodMemSearchTool(client=client, space_ids=["s1"])
    result = tool._run(query="anything")

    assert isinstance(result, ToolFailure)
    assert result.code == "retrieval_failed"
    assert "VECTOR_SEARCH_FAILED" in result.message


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
        GoodMemSearchTool(client=client, space_ids=["s1"])._run(query="q")
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
        GoodMemSearchTool(client=client, space_ids=["s1"])._run(query="q")
    )

    assert payload["total_results"] == 1, "unknown code must not discard chunks"
    assert payload["partial"] is True, "but it is not silently ignored either"
    assert payload["statuses"][0]["code"] == "UNKNOWN"
    assert payload["statuses"][0]["unrecognized"] is True


# ---------------------------------------------------------------- P5
def test_search_makes_exactly_one_request(client, recorder):
    """live: an empty space cost 12.2s of polling in 0.1.1 vs 0.3s without."""
    recorder.route("POST", ":retrieve", ndjson())
    tool = GoodMemSearchTool(client=client, space_ids=["s1"])
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
        space_id="s", upload_dir=str(allowed), base_url="https://x", api_key="k"
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
        space_id="s", upload_dir=str(allowed), base_url="https://x", api_key="k"
    )
    result = tool._run(file_name="link.txt")
    assert isinstance(result, ToolFailure)


def test_upload_is_disabled_until_a_directory_is_configured(tmp_path):
    tool = GoodMemUploadFileTool(space_id="s", base_url="https://x", api_key="k")
    result = tool._run(file_name="anything.txt")
    assert isinstance(result, ToolFailure)
    assert "upload_dir" in result.message


def test_create_memory_takes_no_file_path():
    schema = GoodMemCreateMemoryTool.model_fields["args_schema"].default
    assert "file_path" not in schema.model_fields


# ---------------------------------------------------------------- P21
def test_api_key_is_not_serialized():
    """local: 0.1.1 exposed the key in model_dump() and repr()."""
    tool = GoodMemSearchTool(space_ids=["s"], base_url="https://x", api_key="sk-SECRET")
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
        space_ids=["configured-space"],
        filter="CAST(val('$.a') AS TEXT) = 'b'",
    )
    tool._run(query="q")
    body = recorder.body()
    assert body["spaceKeys"][0]["spaceId"] == "configured-space"
    assert body["spaceKeys"][0]["filter"] == "CAST(val('$.a') AS TEXT) = 'b'"


# ---------------------------------------------------------------- P29
def test_real_vector_scores_are_negative_and_survive_untouched(client, recorder):
    """live capture: a real vector relevanceScore is -0.5345.

    CrewAI documents score as 'higher is better, typically 0-1' and defaults
    score_threshold to 0.6. Applying that to a raw vector score would discard
    every result, so it is only applied when a reranker produced the scores.
    """
    recorder.route(
        "POST", ":retrieve", ndjson(memory_event("m1"), chunk_event("c1", "text", "m1"))
    )
    storage = GoodMemKnowledgeStorage(client=client, space_id="s1")
    with pytest.warns(UserWarning, match="score_threshold is ignored"):
        results = storage.search(["q"], limit=5, score_threshold=0.6)

    assert len(results) == 1, "a negative vector score is not a reason to drop a hit"
    assert results[0]["score"] == REAL_VECTOR_SCORE
    assert results[0]["metadata"]["score_kind"] == "vector"


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
    storage = GoodMemKnowledgeStorage(client=client, space_id="s1", reranker_id="r1")
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
        GoodMemSearchTool(client=client, space_ids=["s1"])._run(query="q")
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
        GoodMemSearchTool(client=client, space_ids=["s1"])._run(query="q")
    )
    assert payload["results"][0]["metadata"]["title"] == "Quarterly report"


def test_get_memory_returns_readable_text_in_one_request(client, recorder):
    """0.1.1 made a second call for content and returned success:true with a
    contentError when it failed. The SDK returns bytes, so the text also has
    to be decoded rather than handed to the agent as base64."""
    recorder.route(
        "GET",
        "/v1/memories/m1",
        # "aGVsbG8gd29ybGQ=" is how the server really sends "hello world".
        httpx.Response(200, json=memory_json("m1", content_b64="aGVsbG8gd29ybGQ=")),
    )
    payload = json.loads(GoodMemGetMemoryTool(client=client)._run(memory_id="m1"))

    assert payload["content"] == "hello world", "readable text, not base64"
    assert "original_content" not in payload
    assert len(recorder.requests) == 1, "content must not cost a second request"


def test_get_memory_describes_binary_content_instead_of_dumping_it(client, recorder):
    recorder.route(
        "GET",
        "/v1/memories/m2",
        httpx.Response(
            200,
            json=memory_json(
                "m2", content_b64="JVBERi0xLjQK", content_type="application/pdf"
            ),
        ),
    )
    payload = json.loads(GoodMemGetMemoryTool(client=client)._run(memory_id="m2"))
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
    tool = GoodMemSearchTool(client=client, space_ids=["s1"])
    tool._run(query="q")
    tool._run(query="q again")
    assert len(recorder.requests) == 2, "the injected client stays usable"


def test_environment_cannot_redirect_an_injected_client(client, recorder, monkeypatch):
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://attacker.example")
    monkeypatch.setenv("GOODMEM_API_KEY", "other-key")
    recorder.route("POST", ":retrieve", ndjson())
    GoodMemSearchTool(client=client, space_ids=["s1"])._run(query="q")
    assert recorder.requests[0].url.host == "goodmem.test"
