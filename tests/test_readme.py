"""Every Python snippet in README.md runs as written.

Each ```python block is executed on its own, in a fresh namespace, against
the local recording server. The only changes are the ones a reader makes:
``GOODMEM_BASE_URL``/``GOODMEM_API_KEY`` point at the server and the
``<space-id>``/``<reranker-id>`` placeholders become UUIDs. The objects a
snippet builds are then used once, so a snippet that constructs something
unusable fails here too.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pytest

from .conftest import RecordingServer


README = Path(__file__).resolve().parent.parent / "README.md"
_FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)

PLACEHOLDERS = {
    "<space-id>": "01a0ace4-678d-7459-aa91-b6ccd46d97d8",
    "<reranker-id>": "019cfd1c-c033-7517-b7de-f73941a0464c",
}


def _snippets() -> list[tuple[int, str]]:
    text = README.read_text()
    return [
        (text.count("\n", 0, match.start()) + 2, match.group(1))
        for match in _FENCE.finditer(text)
    ]


SNIPPETS = _snippets()


def test_readme_has_the_snippets_this_file_runs():
    fences = README.read_text().count("```python")
    assert fences == len(SNIPPETS) >= 2


def _run(source: str, line: int) -> dict[str, Any]:
    for placeholder, value in PLACEHOLDERS.items():
        source = source.replace(placeholder, value)
    namespace: dict[str, Any] = {"__name__": "__readme__"}
    exec(compile(source, f"README.md:{line}", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def readme_env(http_server: RecordingServer, monkeypatch) -> RecordingServer:
    monkeypatch.setenv("GOODMEM_BASE_URL", http_server.base_url)
    monkeypatch.setenv("GOODMEM_API_KEY", "readme-test-key")
    # Keep CrewAI offline: nothing here should reach a network but the server.
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    return http_server


@pytest.mark.parametrize(
    ("line", "source"), SNIPPETS, ids=[f"README.md:{line}" for line, _ in SNIPPETS]
)
def test_readme_snippet_runs(readme_env: RecordingServer, line: int, source: str):
    namespace = _run(source, line)

    if "knowledge" in namespace:
        # The knowledge backend snippet: saving went to GoodMem, and querying
        # CrewAI's Knowledge goes through the GoodMem storage.
        assert ("POST", "/v1/memories:batchCreate") in readme_env.seen()
        namespace["knowledge"].query(["refund approval"], score_threshold=0.0)
        body = readme_env.body()
        assert body["message"] == "refund approval"
        assert body["spaceKeys"][0]["spaceId"] == PLACEHOLDERS["<space-id>"]
        assert "agent" in namespace
    if "search" in namespace:
        # The agent tool snippet: the configured tool really searches.
        namespace["search"].run(query="refund approval")
        body = readme_env.body()
        assert body["message"] == "refund approval"
        assert body["spaceKeys"][0]["spaceId"] == PLACEHOLDERS["<space-id>"]
        assert "agent" in namespace
    if "knowledge" not in namespace and "search" not in namespace:
        pytest.fail(f"README.md:{line}: no snippet check covers this block")
