"""GoodMem as a CrewAI knowledge backend."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
import warnings

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from goodmem import MemoryCreationRequest
from pydantic import Field, PrivateAttr

from crewai_goodmem._connection import GoodMemConnection
from crewai_goodmem._results import GoodMemRetrievalError, classify, hits_from_events
from crewai_goodmem.filters import combine, from_mapping


if TYPE_CHECKING:
    from crewai.rag.types import SearchResult


class GoodMemIngestionError(RuntimeError):
    """A write failed partway. ``created_memory_ids`` records what was accepted."""

    def __init__(self, message: str, *, created_memory_ids: list[str] | None = None):
        super().__init__(message)
        self.created_memory_ids = created_memory_ids or []


class GoodMemKnowledgeStorage(GoodMemConnection, BaseKnowledgeStorage):
    """Back CrewAI ``Knowledge`` with a GoodMem space.

    Pass an instance as ``Knowledge(storage=...)`` to search and store through
    GoodMem instead of CrewAI's bundled vector store::

        storage = GoodMemKnowledgeStorage(space_id="…", reranker_id="…")
        knowledge = Knowledge(collection_name="docs", sources=[], storage=storage)

    Scores are reported exactly as GoodMem returns them. A vector score is an
    opaque similarity that may be negative and is not on a 0-1 scale, so
    ``score_threshold`` is only applied when ``reranker_id`` is configured and
    the scores are genuine relevance values. Results keep the server's
    ordering; they are not re-sorted client-side.

    Async methods run the synchronous SDK on a worker thread, so they do not
    block the event loop.
    """

    space_id: str
    space_ids: list[str] = Field(default_factory=list)
    reranker_id: str | None = None
    filter: str | None = Field(
        default=None,
        description="A GoodMem filter expression applied to every configured space.",
    )
    fetch_k: int | None = Field(default=None, gt=0)
    wait_for_indexing: bool = True
    indexing_timeout: float = 120.0
    allow_reset: bool = False

    _warned_threshold: bool = PrivateAttr(default=False)

    def _targets(self) -> list[str]:
        return [self.space_id, *[s for s in self.space_ids if s != self.space_id]]

    # ------------------------------------------------------------ searching
    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        expression = combine(
            self.filter,
            from_mapping(metadata_filter) if metadata_filter else None,
        )
        reranked = bool(self.reranker_id)

        if not reranked and score_threshold and not self._warned_threshold:
            self._warned_threshold = True
            warnings.warn(
                "score_threshold is ignored without a reranker: GoodMem vector "
                "scores are opaque similarities (possibly negative), not 0-1 "
                "relevance. Configure reranker_id to filter by score.",
                stacklevel=2,
            )

        merged: dict[str, dict[str, Any]] = {}
        all_statuses: list[dict[str, Any]] = []
        with self._session() as client:
            for text in query:
                if not text or not text.strip():
                    continue
                kwargs: dict[str, Any] = {
                    "message": text,
                    "requested_size": self.fetch_k or limit,
                    "fetch_memory": True,
                    "stream": False,
                }
                if expression is None:
                    kwargs["space_ids"] = self._targets()
                else:
                    kwargs["space_keys"] = [
                        {"spaceId": sid, "filter": expression}
                        for sid in self._targets()
                    ]
                if reranked:
                    kwargs["reranker_id"] = self.reranker_id
                    kwargs["max_results"] = limit

                events = list(client.memories.retrieve(**kwargs))
                statuses, degraded = classify(events)
                all_statuses.extend(statuses)
                hits = hits_from_events(events, reranked=reranked)

                # A failed search must not look like an empty one.
                if degraded and not hits:
                    raise GoodMemRetrievalError(
                        "; ".join(
                            f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}"
                            for s in statuses
                        )
                        or "Retrieval failed",
                        statuses=statuses,
                    )

                for hit in hits:
                    if reranked and score_threshold is not None:
                        score = hit["score"]
                        if score is not None and score < score_threshold:
                            continue
                    # Keep the first occurrence: the server already ranked each
                    # query's results, and comparing raw scores would assume a
                    # direction. A vector score is better when it is *lower* on
                    # some metrics, so "keep the larger score" can pick the
                    # worse duplicate.
                    merged.setdefault(hit["chunk_id"], hit)

        results: list[SearchResult] = []
        for hit in list(merged.values())[:limit]:
            metadata = dict(hit["metadata"])
            metadata.update(
                memory_id=hit["memory_id"],
                space_id=hit["space_id"],
                source=hit["source"],
                score_kind=hit["score_kind"],
            )
            if all_statuses:
                # The caller gets the results AND the fact that they may be
                # incomplete, rather than one silently standing in for both.
                metadata["goodmem_statuses"] = all_statuses
            # A dict literal, not SearchResult(...): it is a TypedDict used
            # only for checking, and importing it at runtime is unnecessary.
            results.append(
                {
                    "id": hit["chunk_id"],
                    "content": hit["chunk_text"],
                    "metadata": metadata,
                    "score": hit["score"],
                }
            )
        return results

    async def asearch(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self.search, query, limit, metadata_filter, score_threshold
        )

    # ------------------------------------------------------------- writing
    def save(self, documents: list[str]) -> None:
        texts = [d for d in documents if d and d.strip()]
        if not texts:
            return

        accepted: list[str] = []
        with self._session() as client:
            response = client.memories.batch_create(
                requests=[
                    MemoryCreationRequest(
                        space_id=self.space_id,
                        original_content=text,
                        content_type="text/plain",
                    )
                    for text in texts
                ]
            )
            failures = []
            for position, result in enumerate(response.results or []):
                # On success the id is carried on the nested memory, not on
                # the result's own memory_id field, which stays null.
                memory_id = result.memory_id or getattr(
                    result.memory, "memory_id", None
                )
                if result.success and memory_id:
                    accepted.append(memory_id)
                else:
                    index = (
                        result.request_index
                        if result.request_index is not None
                        else position
                    )
                    failures.append(f"#{index}: {result.error or 'unknown error'}")
            if failures:
                raise GoodMemIngestionError(
                    f"{len(failures)} of {len(texts)} documents failed: "
                    + "; ".join(failures),
                    created_memory_ids=accepted,
                )

            if self.wait_for_indexing:
                try:
                    self._wait(client, accepted)
                except Exception as exc:
                    raise GoodMemIngestionError(
                        f"Documents were written but indexing was not confirmed: {exc}",
                        created_memory_ids=accepted,
                    ) from exc

    async def asave(self, documents: list[str]) -> None:
        await asyncio.to_thread(self.save, documents)

    def _wait(self, client: Any, memory_ids: list[str]) -> None:
        import time

        deadline = time.monotonic() + self.indexing_timeout
        pending = list(dict.fromkeys(memory_ids))
        while pending:
            still: list[str] = []
            for memory_id in pending:
                status = client.memories.get(id=memory_id).processing_status
                if status == "FAILED":
                    raise RuntimeError(f"memory {memory_id} failed processing")
                if status != "COMPLETED":
                    still.append(memory_id)
            if not still:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{len(still)} memories still indexing after "
                    f"{self.indexing_timeout}s: {still}"
                )
            pending = still
            time.sleep(0.5)

    # ------------------------------------------------------------ resetting
    def reset(self) -> None:
        """Delete every memory in the configured space.

        Destructive and irreversible, so it requires ``allow_reset=True``
        rather than silently deleting a space's contents — or silently doing
        nothing, which would be worse.
        """
        if not self.allow_reset:
            raise PermissionError(
                "reset() would permanently delete every memory in space "
                f"{self.space_id}. Construct the storage with allow_reset=True "
                "to permit it."
            )
        with self._session() as client:
            ids = [m.memory_id for m in client.memories.list(space_id=self.space_id)]
            for memory_id in ids:
                client.memories.delete(id=memory_id)

    async def areset(self) -> None:
        await asyncio.to_thread(self.reset)
