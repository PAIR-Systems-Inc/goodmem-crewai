"""Opt-in LLM post-processing on ``GoodMemSearchTool`` (``llm_id``).

The developer sets ``llm_id``; the model never can. GoodMem then runs that
LLM over the retrieved passages and streams back an ``abstractReply``, which
the tool returns as ``abstract_reply`` next to the passages. Against 0.2.1
there was no way to ask for one: ``llm_id`` was not a field, nothing reached
the request and no answer came back.

Every stream below is the live server's (GoodMem, 2026-09-29, qwen3-8b via
OpenRouter), trimmed to the events that matter. Nothing inside the
integration or the SDK is mocked.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
import pytest

from goodmem_crewai import GoodMemKnowledgeStorage, GoodMemSearchTool

from .conftest import REAL_VECTOR_SCORE, chunk_event, memory_event, ndjson


SPACE = "01a0ace4-678d-7459-aa91-b6ccd46d97d8"
RERANKER = "019cfd1c-c033-7517-b7de-f73941a0464c"
LLM = "019cfd9f-0963-76f9-b069-4cde19a64ba8"
MISSING_LLM = "00000000-0000-0000-0000-000000000000"
MEMORY = "01a0eb6d-364e-727b-99f6-edfa30a146c9"
RESULT_SET = "01a0eb6d-3ef3-72d6-ad3c-bccb4b712a7a"

# The live answer, verbatim.
REPLY_TEXT = (
    "The internal audit codeword mentioned in the retrieved data is **ZEPHYR-7**. "
    "This specific detail is explicitly stated in the content provided. No "
    "additional context or alternative codewords are included in the retrieved "
    "records."
)
PASSAGE = "The internal audit codeword is ZEPHYR-7. Revenue rose in the north."

# Live, for an LLM id that does not exist: NOT_FOUND before the hits and
# SUMMARIZATION_FAILED after them, both carrying the id in details.
LLM_NOT_FOUND = {
    "status": {
        "code": "NOT_FOUND",
        "message": (
            f"LLM validation failed: Exception: LLM not found: {MISSING_LLM} "
            f"(ID: {MISSING_LLM}). Verify the LLM exists and is accessible."
        ),
        "details": {"llm_id": MISSING_LLM},
    }
}
LLM_CLIENT_FAILED = {
    "status": {
        "code": "SUMMARIZATION_FAILED",
        "message": (
            "Failed to create LLM inference client: Exception: LLM not found: "
            f"{MISSING_LLM}"
        ),
        "details": {"llm_id": MISSING_LLM},
    }
}
# Live, for an existing LLM whose provider refused the call (out of credits).
LLM_PROVIDER_FAILED = {
    "status": {
        "code": "SUMMARIZATION_FAILED",
        "message": (
            "StatusOr{status=INTERNAL: Summary unavailable due to LLM processing "
            "error: com.openai.errors.RateLimitException: 429: You have no "
            "credits remaining.}"
        ),
    }
}


def _hits() -> list[dict[str, Any]]:
    return [
        memory_event(MEMORY, {"title": "audit note"}),
        chunk_event("c1", PASSAGE, MEMORY, REAL_VECTOR_SCORE),
    ]


def _answer() -> dict[str, Any]:
    return {
        "abstractReply": {
            "text": REPLY_TEXT,
            "relevanceScore": 0.0,
            "resultSetId": RESULT_SET,
        }
    }


def _config(recorder) -> dict[str, Any]:
    return recorder.body()["postProcessor"]["config"]


# ------------------------------------------------------------------ request
def test_llm_id_is_sent_in_the_post_processor_config(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits(), _answer()))
    GoodMemSearchTool(client=client, space_ids=[SPACE], k=3, llm_id=LLM)._run(
        query="audit codeword"
    )

    config = _config(recorder)
    assert config["llm_id"] == LLM
    assert "reranker_id" not in config
    # The post-processor keeps k results rather than the server's default 10.
    assert config["max_results"] == 3


def test_llm_id_sits_next_to_reranker_id(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits(), _answer()))
    GoodMemSearchTool(
        client=client, space_ids=[SPACE], reranker_id=RERANKER, llm_id=LLM
    )._run(query="audit codeword")

    config = _config(recorder)
    assert (config["reranker_id"], config["llm_id"]) == (RERANKER, LLM)


def test_without_llm_id_the_request_is_unchanged(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits()))
    GoodMemSearchTool(client=client, space_ids=[SPACE])._run(query="q")
    assert "postProcessor" not in recorder.body()

    GoodMemSearchTool(client=client, space_ids=[SPACE], reranker_id=RERANKER)._run(
        query="q"
    )
    assert "llm_id" not in _config(recorder)


def test_an_uppercase_llm_id_is_sent_lowercase(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits(), _answer()))
    GoodMemSearchTool(client=client, space_ids=[SPACE], llm_id=LLM.upper())._run(
        query="q"
    )
    assert _config(recorder)["llm_id"] == LLM


@pytest.mark.parametrize(
    "bad",
    ["qwen3-8b", "../llms", f"{LLM}\n", f" {LLM}", ""],
    ids=repr,
)
def test_a_non_uuid_llm_id_is_refused_without_a_request(client, recorder, bad):
    recorder.route("POST", ":retrieve", ndjson(*_hits(), _answer()))
    result = GoodMemSearchTool(client=client, space_ids=[SPACE], llm_id=bad)._run(
        query="q"
    )

    assert isinstance(result, ToolFailure), result
    assert result.reason is ToolFailureReason.INVALID_INPUT
    assert result.code == "invalid_input"
    assert "llm_id must be a UUID" in result.message
    assert recorder.requests == []


def test_the_model_cannot_set_llm_id():
    """Developer configuration, like reranker_id: not in the model's schema."""
    assert "llm_id" in GoodMemSearchTool.model_fields
    schema = GoodMemSearchTool.model_fields["args_schema"].default
    assert list(schema.model_fields) == ["query"]


# ----------------------------------------------------------------- response
def test_the_llm_answer_is_returned_with_the_passages(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits(), _answer()))
    payload = json.loads(
        GoodMemSearchTool(client=client, space_ids=[SPACE], llm_id=LLM)._run(
            query="What is the audit codeword?"
        )
    )

    assert _config(recorder)["llm_id"] == LLM, "the answer was asked for"
    assert payload["abstract_reply"]["text"] == REPLY_TEXT
    assert payload["partial"] is False
    assert "statuses" not in payload
    assert [h["chunk_text"] for h in payload["results"]] == [PASSAGE]
    # An LLM does not rerank: the score is the server's vector score, as is.
    assert payload["results"][0]["score"] == REAL_VECTOR_SCORE
    assert payload["results"][0]["score_kind"] == "vector"


@pytest.mark.parametrize(
    ("statuses", "codes"),
    [
        pytest.param(
            (LLM_NOT_FOUND, LLM_CLIENT_FAILED),
            ["NOT_FOUND", "SUMMARIZATION_FAILED"],
            id="llm-not-found",
        ),
        pytest.param(
            (LLM_PROVIDER_FAILED,), ["SUMMARIZATION_FAILED"], id="provider-429"
        ),
    ],
)
def test_a_failed_llm_keeps_the_hits_and_marks_partial(
    client, recorder, statuses, codes
):
    """Retrieval status contract: a failed LLM is a problem status. The hits
    are kept, the result is partial with the statuses, nothing is raised."""
    first, *rest = statuses
    recorder.route("POST", ":retrieve", ndjson(first, *_hits(), *rest))
    result = GoodMemSearchTool(
        client=client, space_ids=[SPACE], llm_id=MISSING_LLM
    )._run(query="audit codeword")

    assert not isinstance(result, ToolFailure), result
    payload = json.loads(result)
    assert payload["partial"] is True
    assert [s["code"] for s in payload["statuses"]] == codes
    assert all(s["message"] for s in payload["statuses"])
    assert payload["total_results"] == 1
    assert payload["results"][0]["chunk_text"] == PASSAGE
    assert "abstract_reply" not in payload
    assert _config(recorder)["llm_id"] == MISSING_LLM


def test_a_failed_llm_does_not_relabel_reranked_scores(client, recorder):
    """The LLM's NOT_FOUND is about the LLM, not the reranker: scores the
    reranker produced stay reranker scores."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            LLM_NOT_FOUND,
            memory_event(MEMORY),
            chunk_event("c1", PASSAGE, MEMORY, 0.91),
            LLM_CLIENT_FAILED,
        ),
    )
    payload = json.loads(
        GoodMemSearchTool(
            client=client, space_ids=[SPACE], reranker_id=RERANKER, llm_id=MISSING_LLM
        )._run(query="q")
    )
    assert payload["results"][0]["score_kind"] == "reranker"
    assert payload["results"][0]["score"] == 0.91
    assert payload["partial"] is True


# ---------------------------------------------------------- knowledge storage
def test_knowledge_storage_does_not_take_an_llm():
    """CrewAI puts only each SearchResult's ``content`` into the prompt
    (``extract_knowledge_context``), and ``search()`` runs one retrieval per
    query string. An LLM answer there would be paid for on every task and
    then discarded, so the option lives on the tool only."""
    assert "llm_id" not in GoodMemKnowledgeStorage.model_fields


def test_knowledge_storage_never_requests_an_llm(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(*_hits()))
    GoodMemKnowledgeStorage(client=client, space_id=SPACE, reranker_id=RERANKER).search(
        ["q"], score_threshold=float("-inf")
    )
    assert "llm_id" not in _config(recorder)
