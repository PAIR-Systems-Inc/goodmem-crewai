# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-09-25

### Security

- **An id can no longer redirect a call to a different resource.** The
  `goodmem` SDK puts ids into URL paths unescaped and httpx resolves dot
  segments before sending, so `GoodMemDeleteMemoryTool(memory_id="../spaces/<id>")`
  sent `DELETE /v1/spaces/<id>`, deleting the whole space, and reported
  `{"deleted": true}`. Measured on 0.2.0 against a local recording server:
  `a/../../spaces/<id>` and `<id>/../../spaces/<id>` did the same;
  `%2e%2e/spaces/<id>` and `..%2Fspaces%2F<id>` were sent as written for the
  server to decode; on list-memories `<id>#frag` and `<id>?x=1` fetched the
  space itself instead. The get/update/delete space, list/get/delete memory
  tools, `wait_for_memories` and `GoodMemKnowledgeStorage.reset()` were all
  affected; for `reset()` the configured `space_id` decided whose memories were
  listed and then deleted. Every id must now be a UUID and is sent in lowercase.
  Anything else is refused before a request is made: tools return a
  `ToolFailure` with reason `INVALID_INPUT`, while `GoodMemKnowledgeStorage`
  and `wait_for_memories` raise `ValueError`. Configured ids sent in a request
  body (search spaces, reranker, embedder, target space) are checked the same
  way, and so are ids the server lists before they are used in a path:
  `reset()` checks every listed id before it deletes any.
- The model-facing id arguments are declared as UUIDs (`pattern`) in the tool
  schema, so the model is told. The check inside the tool is what refuses;
  pydantic does not enforce the pattern, because CrewAI would then record the
  refusal as a bare exception instead of `INVALID_INPUT`.

### Fixed

- **`GoodMemListEmbeddersTool` and `GoodMemListRerankersTool` failed on every
  call.** They passed `max_items` to `embedders.list()` / `rerankers.list()`,
  which in `goodmem` 0.1.34/0.1.35 are not paginated and take no such keyword,
  so every call returned `ToolFailure("... got an unexpected keyword argument
  'max_items'")` without making a request. They now fetch the full list and
  keep at most `max_items`, reporting `truncated` exactly. The typing Protocol
  for these two APIs now spells out the SDK's `list()` signature instead of
  `**kwargs`, so mypy rejects the bad keyword.
- **A failed reranker no longer makes every result disappear.** Whether hits
  were reranked was decided from configuration (`reranker_id` set). With a
  reranker id that does not exist, the server reports `NOT_FOUND` and
  `RERANKING_FAILED` and still returns the vector search's hits (live scores
  `-0.5947`, `-0.2715`, `-0.2514`). `GoodMemSearchTool` labelled them
  `score_kind="reranker"`, and `GoodMemKnowledgeStorage` left them un-negated,
  applied CrewAI's default `score_threshold=0.6` to them, returned `[]`, and
  warned that "this reranker's scores ranged -0.595..-0.251". It is now
  decided from the response: when either status is present the hits are
  vector results (negated in the storage, not thresholded), and
  `partial`/`goodmem_partial` stay true with the statuses.
- **A boolean `metadata_filter` value matched nothing.** Every value was
  turned into text, so `{"flag": True}` became
  `CAST(val('$.flag') AS TEXT) = 'True'`, which matched 0 rows live where
  `CAST(val('$.flag') AS BOOLEAN) = true` matched 1. Values are now cast by
  type: `bool` as `BOOLEAN` (`true`/`false`), `int`/`float` as `NUMERIC`
  (plain decimal, no exponent), `str` as escaped `TEXT`. `None`, lists, dicts,
  NaN/infinity and other objects raise `ValueError` instead of building a
  filter that can never match; `GoodMemSearchTool` returns that as an
  `INVALID_INPUT` `ToolFailure` without making a request. The new
  `filters.equals()` builds one typed comparison.
- **The declared CrewAI floor could not import the package.** `crewai>=1.15`
  admitted 1.15.0–1.15.8, which do not ship `crewai.tools.tool_failure`
  (first in 1.15.9); installing at the lowest allowed versions gave
  crewai 1.15.0 and `ModuleNotFoundError` on `import crewai_goodmem`. The
  requirement is now `crewai>=1.15.9`. A new CI job installs the built wheel
  with `uv pip install --resolution lowest`, checks that exactly the declared
  floors (crewai 1.15.9, goodmem 0.1.34) were installed and that the wheel is
  what imports, and runs the offline suite there. The `goodmem>=0.1.34` floor
  passes that job unchanged.
- **The README quickstarts did not run.** The knowledge-backend snippet failed
  on its second line, `from crewai.knowledge import Knowledge`
  (`crewai/knowledge/__init__.py` is empty; the class is exported as
  `from crewai import Knowledge`), and the agent-tool snippet used `Agent`
  without importing it. Both are fixed, and `tests/test_readme.py` (its own
  CI step) executes every Python snippet in the README against a local mock
  server and then uses what it built: `knowledge.query()` and `search.run()`
  must each reach GoodMem. The Development section no longer tells you to set
  `GOODMEM_RERANKER_ID` for "the reranker tests": no test reads it.
- **The PyPI "Documentation" link was dead.** `https://docs.goodmem.com/integrations/crewai`
  fails the TLS handshake (`tlsv1 unrecognized name`). It now points at
  `https://docs.goodmem.ai/docs/integrations/agent-frameworks/crewai/`.

### Changed

- `reranker_id=""` used to mean "no reranker" without saying so. It is now
  refused like any other id that is not a UUID; pass `None` for no reranker.

## [0.2.0] — 2026-09-17

0.2 is a deliberate API break. The integration now uses the official `goodmem`
SDK and adds `GoodMemKnowledgeStorage`, so GoodMem can back CrewAI's own
`Knowledge` instead of only being reachable through standalone tools.
Requires CrewAI 1.15+.

### Fixed

- **A failed search no longer looks like a successful one.** Retrieval statuses
  were parsed and discarded. Asking for reranking with an unavailable reranker
  returned unreranked chunks and reported success; a `VECTOR_SEARCH_FAILED`
  covering one of two spaces was indistinguishable from a complete search, and
  a truncated NDJSON line was skipped in silence. Results now carry `partial`
  and `statuses`; a search that produced nothing usable returns empty results
  with `partial: true` rather than an empty success, and is never raised.
- **`FEATURE_DISABLED` is informational by its code alone.** The server
  defines it as "feature disabled due to missing configuration", so it never
  means a requested feature was lost; the details are not inspected.
- **Statuses from a newer server no longer break retrieval.** The SDK decodes
  codes it does not know as `None`. Those are reported as `UNKNOWN` and mark
  results partial; they never discard chunks and never raise.
- **`public_read` is gone.** GoodMem removed the field from spaces, and sending
  it failed the whole update with HTTP 400.
- **Searching no longer polls.** `wait_for_indexing` retried empty searches for
  up to 10 seconds, so querying an empty space cost 12.2s instead of 0.3s.
  Create-memory waits for its own memory instead, and `wait_for_memories`
  waits on specific IDs. Searching is not a way to wait for indexing.
- **Listing follows pagination.** Space, embedder and memory listings read only
  the server's first page and reported that count as the total. Create-space
  also used that truncated list to decide a name already existed.
- **Create-space no longer returns a space with the wrong embedder.** Asking
  for a name that already existed returned the existing space, silently
  ignoring the requested embedder and reporting success. Creation now creates,
  and a name collision is reported as a conflict.
- **File uploads are confined to a configured directory.** `file_path` was
  model-chosen and unrestricted: an agent could read any file the process
  could, including the caller's own GoodMem credentials, and store it. Uploads
  moved to a separate opt-in `GoodMemUploadFileTool` that requires
  `upload_dir` and rejects paths and symlinks resolving outside it.
- **API keys are no longer serialized.** The key was a plain string field and
  appeared in `model_dump()` and `repr()`. It is a `SecretStr` excluded from
  dumps.
- **Get-memory returns readable text in one request.** It made a second call
  for content and returned `success: true` with the content missing when that
  call failed. Content now arrives with the metadata, is decoded using the
  declared charset, and binary content is described rather than dumped as
  base64.
- **Chunks are joined to their sources.** Chunks and memory definitions were
  returned as two unrelated arrays for the model to correlate. They are joined
  by UUID regardless of event order, deduplicated by chunk ID so two passages
  from one document remain two results.
- **Errors reach the framework.** Failures were `{"success": false}` JSON
  strings, which CrewAI's failure tracking cannot see. Tools return
  `ToolFailure`, so failures appear on `TaskOutput.tool_failures` and the event
  bus, and `tool_failure_policy` applies.

### Changed

- **Search exposes only `query` to the model.** It previously accepted twelve
  arguments including `poll_interval`, `llm_temperature`, `relevance_threshold`
  and arbitrary `space_ids`. Spaces, reranking and filters are configured by
  the developer on the tool.
- `relevance_threshold` was documented as a 0-1 score. Real GoodMem vector
  scores are negative inner products (a live capture returned `-0.5345`; the
  best match is the lowest number). `GoodMemKnowledgeStorage` now presents
  `score` under CrewAI's higher-is-better convention by negating vector
  scores — reranker scores are left as they are — and keeps the server value
  as `metadata["raw_score"]` with `metadata["score_kind"]` naming the scale.
  Neither scale is 0-1, so `score_threshold` is applied only when a reranker
  produced the scores, with a warning otherwise. Reranker scales are also
  model-dependent (Voyage `rerank-2.5` 0.27..0.93 vs Jina `jina-reranker-v3`
  -0.14..0.43 on the same documents), so a threshold that removes every
  result warns and names the observed range.
- Chunking configuration is set by the developer, not chosen by the model.
- `crewai_goodmem.filters` builds metadata filter expressions with escaping
  the server actually accepts (backslash; SQL `''` doubling is rejected) and
  refuses field names and control characters it cannot encode safely.
- `wait_for_memories_completed` is now `wait_for_memories`.

### Added

- `GoodMemKnowledgeStorage`, a `BaseKnowledgeStorage` implementation, so
  `Knowledge(storage=...)` can search and store through GoodMem. Async methods
  run the synchronous SDK on a worker thread. `reset()` requires
  `allow_reset=True` rather than silently deleting a space's contents — or
  silently doing nothing.
- `GoodMemListRerankersTool`, so an agent can discover reranker IDs.
- Connection injection: pass `client=Goodmem(...)` to share a pool or supply
  custom TLS. An injected client is authoritative; environment variables
  cannot redirect it.

### Migration from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `GoodMemRetrieveMemoriesTool(query=…, space_ids=[…], …)` | `GoodMemSearchTool(space_ids=[…], k=…)`; the model passes only `query` |
| `wait_for_indexing` on search | Removed. Create waits for its own memory; use `wait_for_memories` for specific IDs |
| `public_read` on update-space | Removed; use authorization grants |
| `file_path` on create-memory | `GoodMemUploadFileTool` with `upload_dir` |
| `{"success": false, "error": …}` | `ToolFailure`, handled by `tool_failure_policy` |
| `wait_for_memories_completed(...)` | `wait_for_memories(...)` |
| Create-space reused a same-named space | Creation creates; a collision is a conflict |
| `requests` | the official `goodmem` SDK |

[Unreleased]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/releases/tag/v0.1.0
