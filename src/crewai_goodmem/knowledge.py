"""GoodMem as a CrewAI knowledge backend."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
import warnings

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from goodmem import MemoryCreationRequest
from pydantic import Field, PrivateAttr

from crewai_goodmem._connection import GoodMemConnection
from crewai_goodmem._ids import require_uuid
from crewai_goodmem._results import classify, hits_from_events
from crewai_goodmem.filters import combine, from_mapping


if TYPE_CHECKING:
    from crewai.rag.types import SearchResult


logger = logging.getLogger(__name__)


class GoodMemIngestionError(RuntimeError):
    """A write failed partway. ``created_memory_ids`` records what was accepted."""

    def __init__(self, message: str, *, created_memory_ids: list[str] | None = None):
        super().__init__(message)
        self.created_memory_ids = created_memory_ids or []


def _host_score(hit: dict[str, Any]) -> float:
    """Present a hit's score under CrewAI's higher-is-better convention.

    A GoodMem vector score is a negative inner product: the closest match is
    the most negative number (a live capture ranked -0.6154 above -0.3873).
    Negating it yields a plain similarity that sorts the way ``SearchResult``
    documents. A reranker score already does, and is left alone. The server
    value stays available as ``metadata["raw_score"]``.
    """
    # The SDK types relevance_score as a required float; there is no None case.
    score = float(hit["score"])
    return -score if hit["score_kind"] == "vector" else score


class GoodMemKnowledgeStorage(GoodMemConnection, BaseKnowledgeStorage):
    """Back CrewAI ``Knowledge`` with a GoodMem space.

    Pass an instance as ``Knowledge(storage=...)`` to search and store through
    GoodMem instead of CrewAI's bundled vector store::

        storage = GoodMemKnowledgeStorage(space_id="…", reranker_id="…")
        knowledge = Knowledge(collection_name="docs", sources=[], storage=storage)

    ``score`` follows CrewAI's convention that higher is better. GoodMem's
    vector score is a negative inner product -- the best match is the *lowest*
    number -- so it is negated here; a reranker score already runs the right
    way and is passed through. The untouched server value is kept as
    ``metadata["raw_score"]`` and ``metadata["score_kind"]`` says which scale
    it is. Neither is 0-1, so ``score_threshold`` is only applied when
    ``reranker_id`` is configured and the scores are genuine relevance values.
    Results keep the server's ordering; they are not re-sorted client-side.

    Async methods run the synchronous SDK on a worker thread, so they do not
    block the event loop.

    Every configured id must be a UUID. One that is not raises ``ValueError``
    before any request is made: the SDK places ids in URL paths unescaped.
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
        primary = require_uuid(self.space_id, "space_id")
        others = [
            require_uuid(sid, f"space_ids[{i}]") for i, sid in enumerate(self.space_ids)
        ]
        return [primary, *[s for s in others if s != primary]]

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
        targets = self._targets()
        reranker_id = (
            require_uuid(self.reranker_id, "reranker_id")
            if self.reranker_id is not None
            else None
        )
        reranked = bool(reranker_id)

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
                    kwargs["space_ids"] = targets
                else:
                    kwargs["space_keys"] = [
                        {"spaceId": sid, "filter": expression} for sid in targets
                    ]
                if reranked:
                    kwargs["reranker_id"] = reranker_id
                    kwargs["max_results"] = limit

                events = list(client.memories.retrieve(**kwargs))
                # degraded is exactly bool(statuses); the flag is derived below.
                statuses, _ = classify(events)
                all_statuses.extend(statuses)
                hits = hits_from_events(events, reranked=reranked)

                dropped = 0
                for hit in hits:
                    if reranked and score_threshold is not None:
                        score = hit["score"]
                        if score is not None and score < score_threshold:
                            dropped += 1
                            continue
                    # Keep the first occurrence: the server already ranked each
                    # query's results, and comparing raw scores would assume a
                    # direction. A vector score is better when it is *lower* on
                    # some metrics, so "keep the larger score" can pick the
                    # worse duplicate.
                    merged.setdefault(hit["chunk_id"], hit)

                if reranked and hits and dropped == len(hits):
                    # CrewAI's interface defaults score_threshold to 0.6. That is
                    # a sensible cut on a 0-1 reranker (Voyage rerank-2.5) and
                    # removes everything on one whose scale is not 0-1 (Jina
                    # jina-reranker-v3 measured -0.14..0.43 live). Say so rather
                    # than return an empty list that reads as "no matches".
                    scores = [h["score"] for h in hits if h["score"] is not None]
                    warnings.warn(
                        f"score_threshold={score_threshold} removed all {len(hits)} "
                        f"reranked result(s); this reranker's scores ranged "
                        f"{min(scores):.3f}..{max(scores):.3f}. Reranker score scales "
                        "are model-dependent and not necessarily 0-1; calibrate the "
                        "threshold for the reranker in use.",
                        stacklevel=2,
                    )

        results: list[SearchResult] = []
        for hit in list(merged.values())[:limit]:
            metadata = dict(hit["metadata"])
            metadata.update(
                memory_id=hit["memory_id"],
                space_id=hit["space_id"],
                source=hit["source"],
                score_kind=hit["score_kind"],
                raw_score=hit["score"],
                # True when the server reported a real problem during this
                # search: the caller gets the results AND the fact that they
                # may be incomplete, rather than one silently standing in for
                # both. (Retrieval status contract, Q4a.)
                goodmem_partial=bool(all_statuses),
            )
            if all_statuses:
                metadata["goodmem_statuses"] = all_statuses

            # A dict literal, not SearchResult(...): it is a TypedDict used
            # only for checking, and importing it at runtime is unnecessary.
            results.append(
                {
                    "id": hit["chunk_id"],
                    "content": hit["chunk_text"],
                    "metadata": metadata,
                    "score": _host_score(hit),
                }
            )
        if all_statuses and not results:
            # Contract Q4b: a failed search returns empty rather than raising.
            # A bare list has nowhere to carry the flag, so it is emitted as a
            # warning and logged, with the statuses, so the failure is visible
            # somewhere. CrewAI itself only reads `content`.
            summary = "; ".join(
                f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}"
                for s in all_statuses
            )
            warnings.warn(
                f"GoodMem search returned no results and reported a problem: {summary}",
                stacklevel=2,
            )
            logger.warning(
                "GoodMem search failed with no results; statuses=%s", all_statuses
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
        space_id = require_uuid(self.space_id, "space_id")

        accepted: list[str] = []
        with self._session() as client:
            response = client.memories.batch_create(
                requests=[
                    MemoryCreationRequest(
                        space_id=space_id,
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
        # Server-issued ids, but each one becomes a URL path all the same.
        pending = list(dict.fromkeys(require_uuid(m, "memory_id") for m in memory_ids))
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
        space_id = require_uuid(self.space_id, "space_id")
        with self._session() as client:
            # Listed ids are checked before the first delete, so a bad one
            # stops the reset rather than leaving it half done.
            ids = [
                require_uuid(m.memory_id, "memory_id")
                for m in client.memories.list(space_id=space_id)
            ]
            for memory_id in ids:
                client.memories.delete(id=memory_id)

    async def areset(self) -> None:
        await asyncio.to_thread(self.reset)
