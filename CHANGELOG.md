# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-04-28

### Changed
- Drop Python 3.10 support. The transitive `onnxruntime` dependency (via
  `crewai → chromadb`) no longer ships 3.10 wheels in current versions, so
  installs failed in practice. Supported versions are now 3.11 / 3.12 / 3.13.

### Fixed
- README install banner correctly states the supported Python range.
- Pin `UV_PYTHON` per CI matrix entry so `uv run` doesn't silently rebuild
  the venv against `.python-version` instead of the matrix Python.

## [0.1.0] — 2026-04-27

### Added
- Initial release.
- Eleven CrewAI tools covering the GoodMem v1 REST surface: embedders, spaces
  (CRUD), memories (CRUD), and semantic retrieval with optional reranking, LLM
  summarization, and SQL-style JSONPath metadata filtering.
- `wait_for_memories_completed(memory_ids, *, timeout, interval, ...)` helper
  that polls each memory's `processingStatus` until it reaches a terminal
  state, replacing blind `time.sleep` waits in the example and live test.
- Apache 2.0 license, type stubs (PEP 561), and full type annotations.
- Mocked unit-test suite covering every tool.

[Unreleased]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PAIR-Systems-Inc/goodmem-crewai/releases/tag/v0.1.0
