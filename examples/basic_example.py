"""End-to-end example: store a document, then search it two ways.

Run against a real GoodMem server:

    export GOODMEM_BASE_URL=https://localhost:8080
    export GOODMEM_API_KEY=...
    export GOODMEM_EMBEDDER_ID=...
    export GOODMEM_VERIFY_SSL=false        # only for a self-signed dev server
    python examples/basic_example.py

Creates a temporary space and deletes it at the end, reporting any teardown
failure rather than hiding it.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from crewai.tools.tool_failure import ToolFailure
from goodmem import Goodmem

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemKnowledgeStorage,
    GoodMemSearchTool,
    GoodMemUpdateSpaceTool,
)


HANDBOOK = [
    "Refunds above $500 require a manager's approval before they are issued.",
    "Expense reports are due on the fifth business day of the following month.",
    "The on-call rotation hands over at 10:00 UTC every Wednesday.",
]


def main() -> int:
    base_url = os.environ.get("GOODMEM_BASE_URL")
    api_key = os.environ.get("GOODMEM_API_KEY")
    embedder_id = os.environ.get("GOODMEM_EMBEDDER_ID")
    if not (base_url and api_key and embedder_id):
        print("Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID.")
        return 2
    verify = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"

    # One client, shared by every tool below, closed at the end.
    client = Goodmem(base_url=base_url, api_key=api_key, verify=verify)
    space = client.spaces.create(
        name=f"crewai-example-{uuid.uuid4().hex[:8]}",
        space_embedders=[{"embedderId": embedder_id}],
    )
    space_id = space.space_id
    print(f"created space {space_id}")

    try:
        # --- store, waiting for the write to be searchable -------------
        create = GoodMemCreateMemoryTool(client=client, space_id=space_id)
        for text in HANDBOOK:
            result = create._run(text_content=text, metadata={"category": "policy"})
            if isinstance(result, ToolFailure):
                print(f"  store failed: {result.as_agent_message()}")
                return 1
            print(
                f"  stored {json.loads(result)['memory_id']} ({json.loads(result)['status']})"
            )

        # --- search as an agent tool would ----------------------------
        search = GoodMemSearchTool(client=client, space_ids=[space_id], k=3)
        payload = json.loads(search._run(query="who approves a large refund?"))
        print(
            f"\nsearch returned {payload['total_results']} passage(s), partial={payload['partial']}"
        )
        for hit in payload["results"]:
            print(
                f"  [{hit['score_kind']} {hit['score']:.4f}] {hit['chunk_text'].strip()[:80]}"
            )

        # A filter that matches nothing is an answer, not a failure.
        filtered = GoodMemSearchTool(
            client=client,
            space_ids=[space_id],
            metadata_filter={"category": "nonexistent"},
        )
        print(
            "filtered to a missing category ->",
            json.loads(filtered._run(query="refund"))["total_results"],
            "results",
        )

        # --- the same data through CrewAI's Knowledge interface -------
        storage = GoodMemKnowledgeStorage(client=client, space_id=space_id)
        results = storage.search(["when are expense reports due?"], limit=2)
        print(f"\nknowledge storage returned {len(results)} result(s)")
        for item in results:
            print(f"  {item['id'][:8]}… {item['content'].strip()[:80]}")

        # --- labels still work; public_read no longer exists ----------
        updated = GoodMemUpdateSpaceTool(client=client)._run(
            space_id=space_id, merge_labels={"example": "true"}
        )
        if isinstance(updated, ToolFailure):
            print(f"update failed: {updated.as_agent_message()}")
            return 1
        print("\nlabels now:", json.loads(updated).get("labels"))
        return 0
    finally:
        client.spaces.delete(id=space_id)
        remaining = [s.space_id for s in client.spaces.list(max_items=1000)]
        if space_id in remaining:
            print(f"WARNING: space {space_id} was not deleted", file=sys.stderr)
        else:
            print(f"deleted space {space_id}")
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
