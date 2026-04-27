# crewai-goodmem

GoodMem RAG tools for [CrewAI](https://crewai.com) agents.

[GoodMem](https://goodmem.ai) is a self-hostable RAG service that handles
embedding, chunking, storage, and retrieval on the server. This package
exposes its REST surface as eleven `crewai.tools.BaseTool` classes so any
CrewAI agent can store documents, run semantic search, and manage memory
spaces directly.

## Install

```bash
pip install crewai-goodmem
# or
uv add crewai-goodmem
```

Python 3.10–3.13 supported. The only runtime dependencies are `crewai` and
`requests`.

## Quickstart

```python
from crewai import Agent, Crew, Task
from crewai_goodmem import GoodMemRetrieveMemoriesTool

retrieve = GoodMemRetrieveMemoriesTool()  # reads GOODMEM_BASE_URL / GOODMEM_API_KEY

researcher = Agent(
    role="Knowledge Researcher",
    goal="Answer questions from the team knowledge base.",
    backstory="You always cite the retrieved memory chunks.",
    tools=[retrieve],
)

task = Task(
    description="Search space '<your-space-id>' for 'agent skills' and answer.",
    expected_output="A grounded answer.",
    agent=researcher,
)

print(Crew(agents=[researcher], tasks=[task]).kickoff())
```

A full multi-scenario script lives at [`examples/basic_example.py`](examples/basic_example.py).

## Configuration

Every tool reads the same three environment variables, with constructor
arguments taking precedence:

| Variable | Required | Description |
|---|---|---|
| `GOODMEM_BASE_URL` | Yes | Base URL of your GoodMem server (e.g. `https://localhost:8080`, `https://api.goodmem.ai`). |
| `GOODMEM_API_KEY`  | Yes | API key issued by your GoodMem instance (starts with `gm_`). |
| `GOODMEM_VERIFY_SSL` | No | Set to `false` to skip TLS verification when pointing at a server with a self-signed certificate. |

Pass them at construction time instead if you prefer:

```python
tool = GoodMemRetrieveMemoriesTool(
    base_url="https://localhost:8080",
    api_key="gm_...",
    verify_ssl=False,
)
```

## Available tools

Eleven tools cover the complete GoodMem v1 REST surface — embedders, spaces
(CRUD), memories (CRUD), and semantic retrieval.

| Tool | Operation |
|---|---|
| `GoodMemListEmbeddersTool` | `GET /v1/embedders` |
| `GoodMemListSpacesTool` | `GET /v1/spaces` |
| `GoodMemGetSpaceTool` | `GET /v1/spaces/{id}` |
| `GoodMemCreateSpaceTool` | `POST /v1/spaces` (reuses if name exists) |
| `GoodMemUpdateSpaceTool` | `PUT /v1/spaces/{id}` |
| `GoodMemDeleteSpaceTool` | `DELETE /v1/spaces/{id}` |
| `GoodMemCreateMemoryTool` | `POST /v1/memories` |
| `GoodMemListMemoriesTool` | `GET /v1/spaces/{space_id}/memories` |
| `GoodMemRetrieveMemoriesTool` | `POST /v1/memories:retrieve` (NDJSON stream) |
| `GoodMemGetMemoryTool` | `GET /v1/memories/{id}` + `/content` |
| `GoodMemDeleteMemoryTool` | `DELETE /v1/memories/{id}` |

### GoodMemCreateSpaceTool

Create a new space, or reuse an existing one with the same name.

| Argument | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Unique name for the space |
| `embedder_id` | Yes | — | ID of the embedder model for vector embeddings |
| `chunk_size` | No | 256 | Characters per chunk when splitting documents |
| `chunk_overlap` | No | 25 | Overlapping characters between consecutive chunks |
| `keep_strategy` | No | `KEEP_END` | Separator placement: `KEEP_END`, `KEEP_START`, `DISCARD` |
| `length_measurement` | No | `CHARACTER_COUNT` | `CHARACTER_COUNT` or `TOKEN_COUNT` |

### GoodMemUpdateSpaceTool

Update a space's name, labels, or `publicRead` flag. Only fields you pass
are changed.

| Argument | Required | Description |
|---|---|---|
| `space_id` | Yes | UUID of the space to update |
| `name` | No | New name for the space |
| `public_read` | No | Toggle unauthenticated read access |
| `replace_labels_json` | No | JSON string that replaces all existing labels |
| `merge_labels_json` | No | JSON string that merges into existing labels |

`replace_labels_json` and `merge_labels_json` are mutually exclusive.

### GoodMemCreateMemoryTool

Store a document or plain text as a memory. Binary content is base64-encoded
on upload.

| Argument | Required | Description |
|---|---|---|
| `space_id` | Yes | Space to store the memory in |
| `text_content` | No | Plain text content (used when no file is provided) |
| `file_path` | No | Local file path (PDF, DOCX, image, etc.). Takes priority over `text_content` |
| `metadata` | No | Key-value metadata as a JSON object |

### GoodMemListMemoriesTool

List memories in a space with optional filtering and sorting.

| Argument | Required | Default | Description |
|---|---|---|---|
| `space_id` | Yes | — | UUID of the space to list from |
| `status_filter` | No | — | One of `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `include_content` | No | `false` | Include each memory's original content |
| `sort_by` | No | — | `created_at` or `updated_at` |
| `sort_order` | No | — | `ASCENDING` or `DESCENDING` |

### GoodMemRetrieveMemoriesTool

Perform semantic search across one or more spaces. Supports reranking, LLM
summarization, chronological resort, and server-side metadata filtering via
SQL-style JSONPath expressions.

| Argument | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | — | Natural language search query |
| `space_ids` | Yes | — | List of space IDs to search across |
| `max_results` | No | 5 | Maximum number of results |
| `include_memory_definition` | No | `true` | Fetch full memory metadata alongside chunks |
| `wait_for_indexing` | No | `true` | Retry polling when no results found |
| `max_wait_seconds` | No | 10 | Maximum seconds to poll when `wait_for_indexing` is `true` |
| `poll_interval` | No | 2 | Seconds between polling attempts |
| `reranker_id` | No | — | Reranker model ID for improved ordering |
| `llm_id` | No | — | LLM ID for contextual response generation |
| `relevance_threshold` | No | — | Minimum score (0–1). Used with reranker/LLM |
| `llm_temperature` | No | — | LLM creativity (0–2). Used when `llm_id` is set |
| `chronological_resort` | No | `false` | Reorder by creation time instead of relevance |
| `metadata_filter` | No | — | SQL-style JSONPath expression applied server-side. Applied to every space in `space_ids` |

### GoodMemGetMemoryTool

Retrieve a memory by ID with metadata and optionally the original content.
Text content is returned verbatim (`contentEncoding: "text"`); binary
content is base64-encoded (`contentEncoding: "base64"`).

| Argument | Required | Default | Description |
|---|---|---|---|
| `memory_id` | Yes | — | UUID of the memory to fetch |
| `include_content` | No | `true` | Also fetch the original document content |

### GoodMemDeleteMemoryTool

Permanently delete a memory and its chunks and embeddings.

| Argument | Required | Description |
|---|---|---|
| `memory_id` | Yes | UUID of the memory to delete |

### GoodMemListSpacesTool / GoodMemListEmbeddersTool / GoodMemGetSpaceTool / GoodMemDeleteSpaceTool

`List` tools take no arguments. `GetSpace` and `DeleteSpace` take a single
`space_id`.

## Waiting for indexing

Memory ingestion is asynchronous: `GoodMemCreateMemoryTool` returns once the
record exists, but the chunks aren't searchable until the server finishes
embedding them. Instead of guessing with a fixed sleep, poll the actual
`processingStatus`:

```python
from crewai_goodmem import wait_for_memories_completed

statuses = wait_for_memories_completed(
    [memory_id_1, memory_id_2],
    timeout=60.0,
)
# {'mem-1': 'COMPLETED', 'mem-2': 'COMPLETED'}
```

The function returns once every id has reached a terminal status
(`COMPLETED` or `FAILED`), or raises `TimeoutError` if the deadline elapses.
Configure with `timeout` (max seconds) and `interval` (poll frequency).

## Metadata filtering

Tag memories at creation time and filter server-side at retrieval time:

```python
from crewai_goodmem import GoodMemCreateMemoryTool, GoodMemRetrieveMemoriesTool
import json

create = GoodMemCreateMemoryTool()
retrieve = GoodMemRetrieveMemoriesTool()

create._run(
    space_id=space_id,
    text_content="Shipped the CSV export feature.",
    metadata={"category": "feat"},
)

result = retrieve._run(
    query="new features",
    space_ids=[space_id],
    metadata_filter="CAST(val('$.category') AS TEXT) = 'feat'",
)
print(json.loads(result)["results"])
```

The filter is a SQL-style JSONPath expression evaluated by GoodMem. When
set, the same filter is applied to every space in `space_ids`.

## Development

```bash
uv venv
uv sync --extra dev

uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest               # mocked unit tests only

# live tests require a running GoodMem server
GOODMEM_BASE_URL=https://localhost:8080 \
GOODMEM_API_KEY=gm_... \
uv run pytest -m e2e
```

Conventional Commits are enforced via pre-commit. Install with:

```bash
uv run pre-commit install
```

## Links

- [GoodMem documentation](https://docs.goodmem.com/)
- [GoodMem CrewAI integration docs](https://docs.goodmem.com/integrations/crewai)
- [CrewAI documentation](https://docs.crewai.com/)

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
