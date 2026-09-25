"""Regression tests for defects reproduced against 0.2.1.

Every test drives the real ``goodmem`` SDK over a mock transport (or a local
HTTP server), so the SDK's own signatures, request building and response
parsing are part of what is tested.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, get_type_hints
import warnings

from crewai.tools.tool_failure import ToolFailure
from goodmem import Goodmem
import httpx
import pytest

from crewai_goodmem import (
    GoodMemKnowledgeStorage,
    GoodMemListEmbeddersTool,
    GoodMemListRerankersTool,
    GoodMemSearchTool,
)
from crewai_goodmem._typing import GoodmemClient
from crewai_goodmem.filters import from_mapping

from .conftest import chunk_event, memory_event, ndjson, status_event


OWNER = "019cfcff-37c5-76d0-bd46-8525e29a9c82"
USER = "019cfcff-37c7-75ef-be71-06c83dae99c3"


def _model_json(kind: str, index: int) -> dict[str, Any]:
    """An embedder or reranker with every field the SDK requires."""
    payload: dict[str, Any] = {
        f"{kind}Id": f"019cfd1c-c033-7517-b7de-f73941a0{index:04x}",
        "displayName": f"{kind}-{index}",
        "providerType": "OPENAI",
        "endpointUrl": "https://api.example.test/v1",
        "modelIdentifier": f"model-{index}",
        "supportedModalities": ["TEXT"],
        "labels": {},
        "ownerId": OWNER,
        "createdAt": 1789607045012,
        "updatedAt": 1789607045012,
        "createdById": USER,
        "updatedById": USER,
    }
    if kind == "embedder":
        payload["dimensionality"] = 1536
        payload["distributionType"] = "DENSE"
    return payload


# ------------------------------------------- list embedders / list rerankers
# goodmem 0.1.34/0.1.35: embedders.list() and rerankers.list() take no
# max_items (they are not paginated) and return a plain list. 0.2.1 passed
# max_items=..., so both tools failed on every call with
# "EmbeddersAPI.list() got an unexpected keyword argument 'max_items'".
_LIST_TOOLS = [
    (GoodMemListEmbeddersTool, "embedder", "/v1/embedders", "embedders"),
    (GoodMemListRerankersTool, "reranker", "/v1/rerankers", "rerankers"),
]


@pytest.mark.parametrize(("tool_cls", "kind", "path", "key"), _LIST_TOOLS)
def test_list_model_tools_work_with_their_defaults(
    client, recorder, tool_cls, kind, path, key
):
    items = [_model_json(kind, i) for i in range(3)]
    recorder.route("GET", path, httpx.Response(200, json={key: items}))

    result = tool_cls(client=client)._run()

    assert not isinstance(result, ToolFailure), result
    payload = json.loads(result)
    assert payload["returned"] == 3
    assert payload["truncated"] is False
    assert [i[f"{kind}_id"] for i in payload[key]] == [i[f"{kind}Id"] for i in items]
    assert recorder.paths() == [path]
    assert recorder.requests[0].url.query == b"", "no paging parameter is sent"


@pytest.mark.parametrize(("tool_cls", "kind", "path", "key"), _LIST_TOOLS)
def test_list_model_tools_cap_at_max_items(client, recorder, tool_cls, kind, path, key):
    items = [_model_json(kind, i) for i in range(5)]
    recorder.route("GET", path, httpx.Response(200, json={key: items}))

    payload = json.loads(tool_cls(client=client, max_items=2)._run())
    assert payload["returned"] == 2
    assert payload["truncated"] is True
    assert len(recorder.requests) == 1

    recorder.requests.clear()
    payload = json.loads(tool_cls(client=client, max_items=None)._run())
    assert payload["returned"] == 5
    assert payload["truncated"] is False


@pytest.mark.parametrize("api", ["embedders", "rerankers"])
def test_typed_list_signature_matches_the_sdk(api):
    """The Protocol mypy checks against must not accept arbitrary keywords.

    ``list(**kwargs: Any)`` let ``list(max_items=...)`` pass mypy --strict
    while the SDK rejected it at runtime. Every keyword the Protocol accepts
    must be one the SDK method accepts.
    """
    protocol = get_type_hints(getattr(GoodmemClient, api).fget)["return"]
    typed = inspect.signature(protocol.list).parameters
    assert not any(p.kind is p.VAR_KEYWORD for p in typed.values())
    assert "max_items" not in typed

    sdk_client = Goodmem(base_url="https://goodmem.test", api_key="k")
    try:
        sdk = inspect.signature(getattr(sdk_client, api).list).parameters
    finally:
        sdk_client.close()
    assert set(typed) - {"self"} <= set(sdk)


# ------------------------------------------------ Q4a: reranker fallback hits
SPACE = "01a0ace4-678d-7459-aa91-b6ccd46d97d8"
RERANKER = "019cfd1c-c033-7517-b7de-f73941a0464c"
MISSING_RERANKER = "00000000-0000-0000-0000-000000000000"
MEMORY = "01a0b060-0618-749c-b081-329aedc62cfe"

# Measured live (GoodMem v1.0.320, 2026-09-25) with a reranker id that does
# not exist: the server reports NOT_FOUND and RERANKING_FAILED and still sends
# the vector search's hits, with vector (negative inner product) scores.
FALLBACK_SCORES = [-0.5947084426879883, -0.2715, -0.2513674199581146]


def _reranker_not_found() -> dict[str, Any]:
    return {
        "status": {
            "code": "NOT_FOUND",
            "message": (
                "Reranker validation failed: Exception: Reranker not found: "
                f"{MISSING_RERANKER} (ID: {MISSING_RERANKER}). Verify the "
                "reranker exists and is accessible."
            ),
            "details": {"reranker_id": MISSING_RERANKER},
        }
    }


def _fallback_stream(*statuses: dict[str, Any]) -> httpx.Response:
    return ndjson(
        *statuses,
        memory_event(MEMORY, {"title": "refunds", "category": "policy"}),
        chunk_event(
            "c1", "Refunds above $500 need approval.", MEMORY, FALLBACK_SCORES[0]
        ),
        chunk_event(
            "c2", "Expense reports are due on day 5.", MEMORY, FALLBACK_SCORES[1]
        ),
        chunk_event(
            "c3", "On-call hands over on Wednesday.", MEMORY, FALLBACK_SCORES[2]
        ),
        status_event(
            "FEATURE_DISABLED",
            "Abstract reply generation disabled: no LLM configured.",
            feature="summarization",
        ),
    )


LIVE_FALLBACK = (
    _reranker_not_found(),
    status_event(
        "RERANKING_FAILED",
        "Failed to create reranker client: Exception: Reranker not found: "
        f"{MISSING_RERANKER}",
    ),
)


def test_search_tool_labels_reranker_fallback_hits_as_vector(client, recorder):
    """live D01: 0.2.1 labelled all three fallback hits score_kind='reranker'."""
    recorder.route("POST", ":retrieve", _fallback_stream(*LIVE_FALLBACK))
    tool = GoodMemSearchTool(
        client=client, space_ids=[SPACE], reranker_id=MISSING_RERANKER
    )
    payload = json.loads(tool._run(query="refund approval"))

    assert payload["total_results"] == 3, "fallback hits are never discarded"
    assert [h["score_kind"] for h in payload["results"]] == ["vector"] * 3
    assert [h["score"] for h in payload["results"]] == FALLBACK_SCORES
    assert payload["partial"] is True
    assert [s["code"] for s in payload["statuses"]] == [
        "NOT_FOUND",
        "RERANKING_FAILED",
    ]
    # The reranker was still asked for; only the labelling follows the result.
    assert recorder.body()["postProcessor"]["config"]["reranker_id"] == (
        MISSING_RERANKER
    )


def test_storage_keeps_reranker_fallback_hits_under_default_threshold(client, recorder):
    """live D02: 0.2.1 applied CrewAI's default score_threshold=0.6 to these
    vector scores as if they were reranker scores, dropped all three, and
    warned that "this reranker's scores ranged -0.595..-0.251"."""
    recorder.route("POST", ":retrieve", _fallback_stream(*LIVE_FALLBACK))
    storage = GoodMemKnowledgeStorage(
        client=client, space_id=SPACE, reranker_id=MISSING_RERANKER
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = storage.search(["refund approval"])  # default threshold 0.6

    assert [r["id"] for r in results] == ["c1", "c2", "c3"]
    assert [r["score"] for r in results] == [-s for s in FALLBACK_SCORES]
    assert [r["metadata"]["raw_score"] for r in results] == FALLBACK_SCORES
    assert {r["metadata"]["score_kind"] for r in results} == {"vector"}
    for result in results:
        assert result["metadata"]["goodmem_partial"] is True
        assert [s["code"] for s in result["metadata"]["goodmem_statuses"]] == [
            "NOT_FOUND",
            "RERANKING_FAILED",
        ]
    messages = [str(w.message) for w in caught]
    assert not any("reranked result" in m for m in messages), messages
    assert not any("returned no results" in m for m in messages), messages


@pytest.mark.parametrize(
    "statuses",
    [
        pytest.param((_reranker_not_found(),), id="reranker-not-found-only"),
        pytest.param(
            (status_event("RERANKING_FAILED", "reranker timed out"),),
            id="reranking-failed-only",
        ),
    ],
)
def test_either_reranker_failure_status_means_vector_scores(client, recorder, statuses):
    recorder.route("POST", ":retrieve", _fallback_stream(*statuses))
    tool = GoodMemSearchTool(client=client, space_ids=[SPACE], reranker_id=RERANKER)
    payload = json.loads(tool._run(query="q"))
    assert [h["score_kind"] for h in payload["results"]] == ["vector"] * 3
    assert payload["partial"] is True


def test_unrelated_not_found_does_not_relabel_reranked_hits(client, recorder):
    """A NOT_FOUND about something else (one of several spaces) says nothing
    about the reranker: hits it ranked stay reranker-scored and thresholded."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            status_event("NOT_FOUND", "Space not found", space_id=SPACE),
            memory_event(MEMORY),
            chunk_event("c1", "strong", MEMORY, 0.91),
            chunk_event("c2", "weak", MEMORY, 0.10),
        ),
    )
    storage = GoodMemKnowledgeStorage(
        client=client, space_id=SPACE, reranker_id=RERANKER
    )
    results = storage.search(["q"], score_threshold=0.6)
    assert [r["id"] for r in results] == ["c1"]
    assert results[0]["metadata"]["score_kind"] == "reranker"
    assert results[0]["metadata"]["goodmem_partial"] is True


# ------------------------------------------------ typed metadata_filter values
# live: {'flag': True} was built as CAST(val('$.flag') AS TEXT) = 'True' and
# matched 0 rows; CAST(val('$.flag') AS BOOLEAN) = true matched the 1 row.
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "CAST(val('$.flag') AS BOOLEAN) = true"),
        (False, "CAST(val('$.flag') AS BOOLEAN) = false"),
        (2026, "CAST(val('$.flag') AS NUMERIC) = 2026"),
        (-3, "CAST(val('$.flag') AS NUMERIC) = -3"),
        (2.5, "CAST(val('$.flag') AS NUMERIC) = 2.5"),
        (1e-05, "CAST(val('$.flag') AS NUMERIC) = 0.00001"),
        (1e20, "CAST(val('$.flag') AS NUMERIC) = 100000000000000000000"),
        ("True", "CAST(val('$.flag') AS TEXT) = 'True'"),
        ("o'brien", r"CAST(val('$.flag') AS TEXT) = 'o\'brien'"),
    ],
)
def test_metadata_filter_casts_by_value_type(value, expected):
    assert from_mapping({"flag": value}) == expected


@pytest.mark.parametrize(
    "value",
    [None, [1], {"a": 1}, (1,), b"x", float("nan"), float("inf"), object()],
    ids=["None", "list", "dict", "tuple", "bytes", "nan", "inf", "object"],
)
def test_metadata_filter_refuses_values_it_cannot_compare(value):
    with pytest.raises(ValueError, match="flag"):
        from_mapping({"flag": value})


def test_boolean_metadata_filter_reaches_the_server_as_a_boolean(client, recorder):
    recorder.route("POST", ":retrieve", ndjson())
    GoodMemSearchTool(
        client=client, space_ids=[SPACE], metadata_filter={"flag": True, "n": 1}
    )._run(query="q")
    assert recorder.body()["spaceKeys"][0]["filter"] == (
        "(CAST(val('$.flag') AS BOOLEAN) = true) AND (CAST(val('$.n') AS NUMERIC) = 1)"
    )


def test_search_tool_refuses_an_unfilterable_value_without_a_request(client, recorder):
    """0.2.1 sent CAST(val('$.flag') AS TEXT) = 'None', a filter that can
    never match, and reported an ordinary empty result."""
    recorder.route("POST", ":retrieve", ndjson())
    result = GoodMemSearchTool(
        client=client, space_ids=[SPACE], metadata_filter={"flag": None}
    )._run(query="q")
    assert isinstance(result, ToolFailure)
    assert result.code == "invalid_input"
    assert recorder.requests == []
