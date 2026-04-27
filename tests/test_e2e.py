"""End-to-end tests against a live GoodMem server.

Skipped unless ``GOODMEM_BASE_URL`` is set. Each test cleans up the spaces and
memories it creates in a ``finally`` block so failures don't leave server state
behind.
"""

from __future__ import annotations

import json
import os
import uuid

from dotenv import load_dotenv
import pytest

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemCreateSpaceTool,
    GoodMemDeleteMemoryTool,
    GoodMemDeleteSpaceTool,
    GoodMemGetMemoryTool,
    GoodMemGetSpaceTool,
    GoodMemListEmbeddersTool,
    GoodMemListMemoriesTool,
    GoodMemListSpacesTool,
    GoodMemRetrieveMemoriesTool,
    GoodMemUpdateSpaceTool,
    wait_for_memories_completed,
)


load_dotenv()


def _verify_ssl() -> bool:
    return os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("GOODMEM_BASE_URL"),
        reason="GOODMEM_BASE_URL not set; skipping live e2e tests",
    ),
    pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning"),
]


@pytest.fixture(scope="module")
def embedder_id() -> str:
    result = json.loads(GoodMemListEmbeddersTool(verify_ssl=_verify_ssl())._run())
    assert result["success"], result
    embedders = result["embedders"]
    assert embedders, "no embedders registered on the GoodMem server"
    return embedders[0]["embedderId"]


def test_list_embedders_returns_at_least_one() -> None:
    result = json.loads(GoodMemListEmbeddersTool(verify_ssl=_verify_ssl())._run())
    assert result["success"]
    assert result["totalEmbedders"] >= 1


def test_space_lifecycle(embedder_id: str) -> None:
    create = GoodMemCreateSpaceTool(verify_ssl=_verify_ssl())
    get = GoodMemGetSpaceTool(verify_ssl=_verify_ssl())
    update = GoodMemUpdateSpaceTool(verify_ssl=_verify_ssl())
    list_spaces = GoodMemListSpacesTool(verify_ssl=_verify_ssl())
    delete = GoodMemDeleteSpaceTool(verify_ssl=_verify_ssl())

    name = f"crewai-goodmem-e2e-{uuid.uuid4().hex[:8]}"
    space_id: str | None = None
    try:
        # create
        result = json.loads(create._run(name=name, embedder_id=embedder_id))
        assert result["success"], result
        assert result["reused"] is False
        space_id = result["spaceId"]
        assert space_id

        # get_space — round-trip the space we just created
        fetched = json.loads(get._run(space_id=space_id))
        assert fetched["success"], fetched
        assert fetched["space"]["spaceId"] == space_id
        assert fetched["space"]["name"] == name

        # update_space — rename and merge a label, confirm both stuck
        renamed = f"{name}-renamed"
        updated = json.loads(
            update._run(
                space_id=space_id,
                name=renamed,
                merge_labels_json='{"e2e": "true"}',
            )
        )
        assert updated["success"], updated
        assert updated["space"]["name"] == renamed
        assert updated["space"].get("labels", {}).get("e2e") == "true"

        # list_spaces — the renamed space must appear in the listing
        listed = json.loads(list_spaces._run())
        assert listed["success"], listed
        assert any(s["spaceId"] == space_id for s in listed["spaces"]), (
            f"space {space_id} missing from list_spaces output"
        )
    finally:
        if space_id:
            delete._run(space_id=space_id)


def test_memory_lifecycle(embedder_id: str) -> None:
    create_space = GoodMemCreateSpaceTool(verify_ssl=_verify_ssl())
    delete_space = GoodMemDeleteSpaceTool(verify_ssl=_verify_ssl())
    create_memory = GoodMemCreateMemoryTool(verify_ssl=_verify_ssl())
    list_memories = GoodMemListMemoriesTool(verify_ssl=_verify_ssl())
    get_memory = GoodMemGetMemoryTool(verify_ssl=_verify_ssl())
    retrieve = GoodMemRetrieveMemoriesTool(verify_ssl=_verify_ssl())
    delete_memory = GoodMemDeleteMemoryTool(verify_ssl=_verify_ssl())

    name = f"crewai-goodmem-e2e-{uuid.uuid4().hex[:8]}"
    space_id: str | None = None
    memory_id: str | None = None
    try:
        space_id = json.loads(create_space._run(name=name, embedder_id=embedder_id))[
            "spaceId"
        ]

        memory_id = json.loads(
            create_memory._run(
                space_id=space_id,
                text_content="The quick brown fox jumps over the lazy dog.",
                metadata={"source": "e2e-test"},
            )
        )["memoryId"]
        assert memory_id

        statuses = wait_for_memories_completed(
            [memory_id], verify_ssl=_verify_ssl(), timeout=60.0
        )
        assert statuses[memory_id] == "COMPLETED"

        listed = json.loads(list_memories._run(space_id=space_id))
        assert listed["success"]
        assert listed["totalMemories"] >= 1

        record = json.loads(get_memory._run(memory_id=memory_id, include_content=True))
        assert record["success"]
        assert record["memory"]["memoryId"] == memory_id

        retrieved = json.loads(
            retrieve._run(query="quick brown fox", space_ids=[space_id])
        )
        assert retrieved["success"]
        assert retrieved["totalResults"] >= 1
    finally:
        if memory_id:
            delete_memory._run(memory_id=memory_id)
        if space_id:
            delete_space._run(space_id=space_id)
