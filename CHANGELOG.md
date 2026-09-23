# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  and `statuses`, and a search that produced nothing usable returns a
  `ToolFailure`.
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
  produced the scores, with a warning otherwise.
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

[Unreleased]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/releases/tag/v0.1.0
