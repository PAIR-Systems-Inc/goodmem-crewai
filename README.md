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
returns empty results with `partial: true` and the statuses, so the model can
tell a failed search from a miss; it is never raised. `GoodMemKnowledgeStorage`
returns a bare list, so in that case it emits a warning and a log line instead.

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

Every id, whether a model passes it or you configure it, must be a UUID;
anything else is refused before a request is made (a tool returns a
`ToolFailure` with reason `INVALID_INPUT`, `GoodMemKnowledgeStorage` and
`wait_for_memories` raise `ValueError`), because the SDK puts ids into the URL
path unescaped and an id such as `../spaces/<id>` would otherwise send the call
to a different resource.

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
`score_kind` follows what the server did, not what was configured: if the
reranker fails (`RERANKING_FAILED`, or `NOT_FOUND` for the reranker), the
server still returns the vector search's hits, and they are kept as `"vector"`
results — negated, not thresholded, and flagged partial with the statuses.

Even with a reranker, the scale is **model-dependent**: on the same documents
Voyage `rerank-2.5` scored `0.27..0.93` and Jina `jina-reranker-v3` scored
`-0.14..0.43`. CrewAI's default `score_threshold=0.6` keeps the top results on
the first and removes everything on the second, so if a threshold drops every
result the storage warns and names the observed range rather than returning a
silent empty list. Calibrate the threshold for the reranker you use.

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest -m "not e2e"    # offline: the SDK over a mock transport and a local HTTP server
GOODMEM_BASE_URL=… GOODMEM_API_KEY=… GOODMEM_EMBEDDER_ID=… GOODMEM_RERANKER_ID=… GOODMEM_VERIFY_SSL=false uv run pytest -m e2e
```

`GOODMEM_RERANKER_ID` is optional — the reranker tests skip without it.
`GOODMEM_VERIFY_SSL=false` is for a local server with a self-signed certificate.

Apache-2.0.
