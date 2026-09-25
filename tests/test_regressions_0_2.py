"""Regression tests for defects reproduced against 0.2.1.

Every test drives the real ``goodmem`` SDK over a mock transport (or a local
HTTP server), so the SDK's own signatures, request building and response
parsing are part of what is tested.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, get_type_hints

from crewai.tools.tool_failure import ToolFailure
from goodmem import Goodmem
import httpx
import pytest

from crewai_goodmem import GoodMemListEmbeddersTool, GoodMemListRerankersTool
from crewai_goodmem._typing import GoodmemClient


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
