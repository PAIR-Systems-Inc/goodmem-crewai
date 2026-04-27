"""GoodMem + CrewAI multi-agent Crew example.

Three scenarios exercising every GoodMem tool end to end:

    Scenario 1 -- Persistent project context across sessions
        An agent stores project context over several turns. A fresh agent
        instance handles the recall question so the answer can only come
        from GoodMem. Ends with an operator-level inspection that lists
        the memories in the space and fetches the latest record with its
        original content.

    Scenario 2 -- Two-agent team knowledge pipeline
        A Scribe writes team notes into a shared space. A separate Analyst
        answers questions by retrieving from that space.

    Scenario 3 -- Structured team activity log with metadata filtering
        Writes activity entries tagged with 'feat', 'fix', 'chore', or
        'docs' in their metadata, then filters by category server-side
        with a JSONPath expression.

Prerequisites:
    - A running GoodMem server (see https://docs.goodmem.ai)
    - An OpenAI API key (or any CrewAI-supported LLM provider)
    - At least one embedder registered on your GoodMem server

Usage::

    # Put the keys in a .env file at the repo root, or export them directly.
    export OPENAI_API_KEY="sk-..."
    export GOODMEM_API_KEY="gm_..."
    export GOODMEM_BASE_URL="https://localhost:8080"

    # Run from the repo root so .env is discoverable.
    python examples/basic_example.py

Set ``GOODMEM_VERIFY_SSL=false`` to skip TLS verification when pointing at a
server with a self-signed certificate.
"""

from __future__ import annotations

import json
import os
import sys

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
import requests
import urllib3

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemCreateSpaceTool,
    GoodMemDeleteMemoryTool,
    GoodMemDeleteSpaceTool,
    GoodMemGetMemoryTool,
    GoodMemListEmbeddersTool,
    GoodMemListMemoriesTool,
    GoodMemRetrieveMemoriesTool,
    wait_for_memories_completed,
)


Tools = dict[str, BaseTool]


load_dotenv()


REQUIRED_ENV_VARS = ("GOODMEM_API_KEY", "GOODMEM_BASE_URL", "OPENAI_API_KEY")


def check_env_vars() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        sys.exit(f"Error: missing required environment variables: {', '.join(missing)}")


SPACE_NAME = "crewai-goodmem-crew-example"

SCENARIO_1_TURNS = [
    "I'm building a CrewAI-based customer support assistant for our SaaS product.",
    "The team uses Python 3.12 with FastAPI and Postgres.",
    "For tests we use pytest with at least 80% coverage required.",
    "Remind me what our coverage requirement is.",
]

TEAM_NOTES = [
    "Q2 goal: reduce customer support response time to under 2 hours.",
    "Our main services are auth-service, billing-service, and notifications-service.",
    "Known issue: notifications-service occasionally drops messages during high load.",
    "Team retro: the CI pipeline is too slow; we should parallelize tests.",
]

SCENARIO_2_QUESTION = "What do we know about our services and current priorities?"

TAGGED_FACTS = [
    ("Added user profile editing to the dashboard.", "feat"),
    ("Built the CSV export feature.", "feat"),
    ("Resolved slow login on the mobile app.", "fix"),
    ("Fixed crash when opening large attachments.", "fix"),
    ("Upgraded Python version across services.", "chore"),
    ("Updated the API reference for billing endpoints.", "docs"),
]

SCENARIO_3_QUESTION = "Show me the new features we've shipped."


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def subsection(title: str) -> None:
    print(f"\n{'- ' * 30}")
    print(f"  {title}")
    print(f"{'- ' * 30}")


def build_tools(verify_ssl: bool) -> Tools:
    kw = {"verify_ssl": verify_ssl}
    return {
        "list_embedders": GoodMemListEmbeddersTool(**kw),
        "create_space": GoodMemCreateSpaceTool(**kw),
        "delete_space": GoodMemDeleteSpaceTool(**kw),
        "create_memory": GoodMemCreateMemoryTool(**kw),
        "list_memories": GoodMemListMemoriesTool(**kw),
        "retrieve": GoodMemRetrieveMemoriesTool(**kw),
        "get_memory": GoodMemGetMemoryTool(**kw),
        "delete_memory": GoodMemDeleteMemoryTool(**kw),
    }


def _wait_for_space_indexed(tools: Tools, space_id: str) -> None:
    """Block until every memory in ``space_id`` has finished indexing."""
    listed = json.loads(tools["list_memories"]._run(space_id=space_id))
    memories = listed.get("memories", []) if listed.get("success") else []
    memory_ids = [m["memoryId"] for m in memories if m.get("memoryId")]
    if memory_ids:
        wait_for_memories_completed(
            memory_ids, verify_ssl=tools["list_memories"].verify_ssl
        )


def setup_space(tools: Tools, space_name: str) -> str:
    try:
        embedders_result = json.loads(tools["list_embedders"]._run())
    except requests.ConnectionError:
        sys.exit(
            "Error: could not connect to the GoodMem server at "
            f"{os.environ['GOODMEM_BASE_URL']}."
        )

    if not embedders_result.get("success"):
        sys.exit(f"Error listing embedders: {embedders_result.get('error')}")
    embedders = embedders_result["embedders"]
    if not embedders:
        sys.exit("Error: No embedders registered on the GoodMem server.")
    embedder_id = embedders[0]["embedderId"]
    print(f"  Using embedder: {embedders[0].get('displayName', embedder_id)}")

    space_result = json.loads(
        tools["create_space"]._run(name=space_name, embedder_id=embedder_id)
    )
    if not space_result.get("success"):
        sys.exit(f"Error creating space: {space_result.get('error')}")
    space_id: str = space_result["spaceId"]
    reused = space_result.get("reused", False)
    print(f"  Space '{space_name}' ({'reused' if reused else 'created'}): {space_id}")
    return space_id


def cleanup(tools: Tools, space_ids: list[str]) -> None:
    for space_id in space_ids:
        if not space_id:
            continue
        try:
            listed = json.loads(tools["list_memories"]._run(space_id=space_id))
            memories = listed.get("memories", []) if listed.get("success") else []
        except Exception:
            memories = []
        print(f"  Space {space_id}: deleting {len(memories)} memories...")
        for memory in memories:
            memory_id = memory.get("memoryId") or memory.get("id")
            if memory_id:
                try:
                    tools["delete_memory"]._run(memory_id=memory_id)
                except Exception:
                    pass
        print(f"  Deleting space {space_id}...")
        try:
            tools["delete_space"]._run(space_id=space_id)
        except Exception:
            pass
    print("  Cleanup complete.")


def _build_scribe(tools: Tools, space_id: str) -> Agent:
    """Build a scribe agent bound to a GoodMem space."""
    return Agent(
        role="Engineering Assistant",
        goal=(
            "Remember everything the engineer shares by storing each fact "
            f"verbatim in the GoodMem space '{space_id}', and answer "
            "follow-up questions by retrieving from that space."
        ),
        backstory=(
            "You are the team's long-term memory. You must always call the "
            "GoodMem tools rather than answering from your own weights. "
            f"When the user shares context, call GoodMemCreateMemory with "
            f"space_id='{space_id}'. When asked a question, call "
            f"GoodMemRetrieveMemories with space_ids=['{space_id}']."
        ),
        tools=[tools["create_memory"], tools["retrieve"]],
        verbose=False,
        allow_delegation=False,
    )


def scenario_1(tools: Tools, space_id: str) -> None:
    """Persistent project context across sessions."""
    section("Scenario 1: Persistent project context across sessions")

    scribe = _build_scribe(tools, space_id)
    recall_index = len(SCENARIO_1_TURNS) - 1

    for i, message in enumerate(SCENARIO_1_TURNS):
        print(f"\n  Turn {i + 1}")
        print(f"  User:  {message}")
        if i == recall_index:
            # Wait until every memory the agent just stored is indexed, then
            # spin up a fresh agent so the recall answer can only come from
            # GoodMem.
            print("  (waiting for indexing...)")
            _wait_for_space_indexed(tools, space_id)
            scribe = _build_scribe(tools, space_id)

        task = Task(
            description=(
                f"Turn {i + 1}. User says: {message!r}. "
                "If this is a fact, store it in GoodMem. If this is a "
                "question, search GoodMem and answer from the retrieved "
                "memories only."
            ),
            expected_output=(
                "A short confirmation if storing, or a grounded answer if retrieving."
            ),
            agent=scribe,
        )
        crew = Crew(
            agents=[scribe],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        print(f"  Agent: {result}")

    # Operator-level inspection: list the memories the agent wrote and pull
    # the most recent one's raw record.
    subsection("Operator inspection")
    listed = json.loads(tools["list_memories"]._run(space_id=space_id))
    memories = listed.get("memories", []) if listed.get("success") else []
    print(f"  {len(memories)} memories stored in space {space_id}")

    if memories:
        latest_id = memories[0].get("memoryId") or memories[0].get("id")
        record = json.loads(
            tools["get_memory"]._run(memory_id=latest_id, include_content=True)
        )
        if record.get("success"):
            meta = record.get("memory", {})
            content = record.get("content", "")
            preview = content[:100] if isinstance(content, str) else ""
            print(f"  Memory ID:    {meta.get('memoryId')}")
            print(f"  Status:       {meta.get('processingStatus')}")
            print(f"  Content type: {meta.get('contentType')}")
            print(f"  Content:      {preview}...")


def scenario_2(tools: Tools) -> str:
    """Two-agent pipeline sharing a dedicated GoodMem space."""
    section("Scenario 2: Two-agent team knowledge pipeline")

    team_space_name = f"{SPACE_NAME}-team"
    team_space_id = setup_space(tools, team_space_name)

    scribe = Agent(
        role="Team Scribe",
        goal=(
            "Store every team note the user gives you in the GoodMem space "
            f"with ID '{team_space_id}'."
        ),
        backstory=(
            "You are a team Scribe. Store every note verbatim using "
            f"GoodMemCreateMemory with space_id='{team_space_id}'. Confirm "
            "storage briefly."
        ),
        tools=[tools["create_memory"]],
        verbose=False,
        allow_delegation=False,
    )

    for note_index, note in enumerate(TEAM_NOTES, start=1):
        print(f"\n  Scribe note {note_index}: {note}")
        task = Task(
            description=f"Store this team note in GoodMem verbatim: {note!r}",
            expected_output="A short confirmation that the note was stored.",
            agent=scribe,
        )
        Crew(
            agents=[scribe],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

    print("\n  Waiting for indexing to complete...")
    _wait_for_space_indexed(tools, team_space_id)

    analyst = Agent(
        role="Team Analyst",
        goal=(
            "Synthesize accurate answers grounded in the team notes stored in "
            f"GoodMem space '{team_space_id}'."
        ),
        backstory=(
            "You are a team Analyst. Use GoodMemRetrieveMemories with "
            f"space_ids=['{team_space_id}'] to find relevant notes and "
            "answer the user's question from those notes only."
        ),
        tools=[tools["retrieve"]],
        verbose=False,
        allow_delegation=False,
    )

    print(f"\n  User (to Analyst): {SCENARIO_2_QUESTION}")
    task = Task(
        description=(
            f"Search the GoodMem space '{team_space_id}' to answer: "
            f"{SCENARIO_2_QUESTION}"
        ),
        expected_output=("A synthesized answer citing the retrieved team notes."),
        agent=analyst,
    )
    result = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    ).kickoff()
    print(f"  Analyst: {result}")

    return team_space_id


def scenario_3(tools: Tools) -> str:
    """Metadata-tagged memories with server-side JSONPath filtering."""
    section("Scenario 3: Structured team activity log (metadata filtering)")

    tagged_space_name = f"{SPACE_NAME}-tagged"
    tagged_space_id = setup_space(tools, tagged_space_name)

    # Write tagged memories directly so metadata is deterministic for the demo.
    print(f"\n  Ingesting {len(TAGGED_FACTS)} tagged memories...")
    memory_ids: list[str] = []
    for content, category in TAGGED_FACTS:
        result = json.loads(
            tools["create_memory"]._run(
                space_id=tagged_space_id,
                text_content=content,
                metadata={"category": category},
            )
        )
        memory_ids.append(result["memoryId"])
        print(f"    [{category:>8}] {content}")

    print("  Waiting for indexing to complete...")
    wait_for_memories_completed(
        memory_ids, verify_ssl=tools["create_memory"].verify_ssl
    )

    release_manager = Agent(
        role="Release Manager",
        goal=(
            "Report the team's feature-only activity by filtering GoodMem "
            f"memories in space '{tagged_space_id}' by the 'category' "
            "metadata field."
        ),
        backstory=(
            "You are a release manager. The team activity log lives in "
            f"GoodMem space '{tagged_space_id}'. Each memory has a 'category' "
            "field in its metadata (one of 'feat', 'fix', 'chore', 'docs'). "
            "To answer category-specific questions, call "
            "GoodMemRetrieveMemories with a metadata_filter using a SQL-style "
            "JSONPath expression against the 'category' field. For example, "
            "for 'feat' entries pass metadata_filter as: "
            "CAST(val('$.category') AS TEXT) = 'feat'. Report each returned "
            "result by its chunkText. Do not invent entries."
        ),
        tools=[tools["retrieve"]],
        verbose=False,
        allow_delegation=False,
    )

    print(f"\n  User: {SCENARIO_3_QUESTION}")
    task = Task(
        description=(
            f"Search the GoodMem space '{tagged_space_id}' with "
            "metadata_filter=\"CAST(val('$.category') AS TEXT) = 'feat'\" "
            f"to answer: {SCENARIO_3_QUESTION}"
        ),
        expected_output=(
            "A bulleted list of feature entries, each quoting the retrieved "
            "chunkText verbatim."
        ),
        agent=release_manager,
    )
    result = Crew(
        agents=[release_manager],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    ).kickoff()
    print(f"  Release Manager: \n{result}")

    return tagged_space_id


def main() -> None:
    check_env_vars()

    print("=" * 60)
    print("  GoodMem + CrewAI Crew Example")
    print("=" * 60)

    verify_ssl = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    tools = build_tools(verify_ssl)

    subsection("Setup: Discovering embedder and creating space")
    space_id = setup_space(tools, SPACE_NAME)
    team_space_id: str | None = None
    tagged_space_id: str | None = None

    try:
        scenario_1(tools, space_id)
        team_space_id = scenario_2(tools)
        tagged_space_id = scenario_3(tools)
    finally:
        subsection("Cleanup")
        spaces_to_clean = [space_id]
        if team_space_id:
            spaces_to_clean.append(team_space_id)
        if tagged_space_id:
            spaces_to_clean.append(tagged_space_id)
        cleanup(tools, spaces_to_clean)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
