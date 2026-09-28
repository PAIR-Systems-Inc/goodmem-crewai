"""Package metadata that users read on PyPI agrees with reality."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

# Hosts the project URLs may use. docs.goodmem.com does not complete a TLS
# handshake ("tlsv1 unrecognized name"); the docs live on docs.goodmem.ai.
KNOWN_HOSTS = {"github.com", "docs.goodmem.ai"}


def test_project_urls_use_live_hosts():
    hosts = {name: urlsplit(url).hostname for name, url in PROJECT["urls"].items()}
    assert {n: h for n, h in hosts.items() if h not in KNOWN_HOSTS} == {}
    assert PROJECT["urls"]["Documentation"] == (
        "https://docs.goodmem.ai/docs/integrations/agent-frameworks/crewai/"
    )


def test_readme_states_the_declared_floors():
    floors = dict(
        re.fullmatch(r"([a-z]+)>=([0-9.]+)", dep).groups()
        for dep in PROJECT["dependencies"]
    )
    readme = (ROOT / "README.md").read_text()
    assert f"CrewAI {floors['crewai']}+" in readme
    assert f"`goodmem` SDK ({floors['goodmem']}+)" in readme
