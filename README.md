# crewai-goodmem

GoodMem knowledge storage and RAG tools for [CrewAI](https://crewai.com).

[GoodMem](https://goodmem.ai) is a self-hostable RAG service that handles
embedding, chunking, storage and retrieval on the server. This package plugs it
into CrewAI two ways: as a knowledge backend for CrewAI's own `Knowledge`, and
as tools an agent can call directly.

## Install

```bash
pip install crewai-goodmem
```

Python 3.11–3.13, CrewAI 1.15+. Runtime dependencies are `crewai` and the
official `goodmem` SDK.

Set `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`, or pass `base_url`/`api_key`, or
inject a configured `Goodmem` client.

## As a knowledge backend

```python
from crewai import Agent, Crew, Task
from crewai.knowledge import Knowledge
from crewai_goodmem import GoodMemKnowledgeStorage

storage = GoodMemKnowledgeStorage(space_id="<space-id>", reranker_id="<reranker-id>")
knowledge = Knowledge(collection_name="handbook", sources=[], storage=storage)

storage.save(["Refunds over $500 need a manager's approval."])

agent = Agent(
    role="Support Lead",
    goal="Answer policy questions from the handbook.",
    backstory="You cite the passage you relied on.",
    knowledge=knowledge,
)
```

## As an agent tool

The search tool takes only a query from the model. Which spaces it searches,
how many results it returns, whether it reranks and any metadata filter are
set by you, so a model cannot redirect the search mid-run.

```python
from crewai_goodmem import GoodMemSearchTool

search = GoodMemSearchTool(
    space_ids=["<space-id>"],
    k=5,
    reranker_id="<reranker-id>",          # optional
    metadata_filter={"category": "policy"},  # optional, escaped for you
)

agent = Agent(role="Researcher", goal="Answer from the knowledge base",
              backstory="You cite sources.", tools=[search])
```

Results carry `partial` and `statuses`. If part of a search failed — a reranker
was unavailable, one space was unreachable — you get the usable passages *and*
the fact that they are incomplete. A search that produced nothing usable
returns a `ToolFailure` rather than an empty success.

## Tools

| Tool | Purpose |
| --- | --- |
| `GoodMemSearchTool` | Semantic search; the model passes only a query |
| `GoodMemCreateMemoryTool` | Store text; waits for indexing by default |
| `GoodMemUploadFileTool` | Store a file from a configured `upload_dir` (opt-in) |
| `GoodMemGetMemoryTool` | Fetch a memory with readable content |
| `GoodMemListMemoriesTool` / `GoodMemDeleteMemoryTool` | Manage memories |
| `GoodMemListSpacesTool` / `GoodMemGetSpaceTool` | Find spaces |
| `GoodMemCreateSpaceTool` / `GoodMemUpdateSpaceTool` / `GoodMemDeleteSpaceTool` | Manage spaces |
| `GoodMemListEmbeddersTool` / `GoodMemListRerankersTool` | Discover model IDs |

Space, memory and file tools carry the authority of the configured API key.
Give them only to crews that need it.

## Waiting for indexing

Searching is not a way to wait for a write. `GoodMemCreateMemoryTool` waits for
its own memory by default; use `wait_for_memories(ids)` to wait on specific IDs.

## Scores

CrewAI's `SearchResult.score` is documented as higher-is-better. GoodMem's
vector score is a negative inner product — the best match is the *lowest*
number (a live capture ranked `-0.6154` above `-0.3873`) — so it is negated to
fit that convention; a reranker score already runs the right way and is passed
through. The untouched server value is kept as `metadata["raw_score"]`, and
`metadata["score_kind"]` (`"vector"` or `"reranker"`) names the scale.

Neither scale is 0–1. `score_threshold` is therefore applied only when a
reranker produced the scores; without one it is ignored with a warning.

## Development

```bash
uv sync --extra dev
uv run pytest tests/          # offline, SDK driven over a mock transport
GOODMEM_BASE_URL=… GOODMEM_API_KEY=… GOODMEM_EMBEDDER_ID=… uv run pytest -m e2e
```

Apache-2.0.
