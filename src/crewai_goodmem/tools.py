"""CrewAI tools for GoodMem, built on the official ``goodmem`` SDK."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, ClassVar, Literal

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
from goodmem.errors import (
    AuthenticationError,
    ConflictError,
    GoodMemError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field

from crewai_goodmem._connection import GoodMemConnection
from crewai_goodmem._results import abstract_reply, classify, hits_from_events
from crewai_goodmem._uploads import GoodMemUploadError, resolve_upload_path
from crewai_goodmem.filters import combine, from_mapping


_GOODMEM_ENV_VARS = [
    EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
    EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
]

_TERMINAL = frozenset({"COMPLETED", "FAILED"})


def _failure(
    exc: Exception,
    context: str,
    *,
    extra_message: str = "",
    extra_details: dict[str, Any] | None = None,
) -> ToolFailure:
    """Map an SDK error onto CrewAI's failure type, keeping the server's text.

    The framework records these on ``TaskOutput.tool_failures`` and the event
    bus; a JSON string saying ``success: false`` is invisible to it.
    """
    if isinstance(exc, (GoodMemUploadError, ValueError)):
        reason, code, retryable = (
            ToolFailureReason.INVALID_INPUT,
            "invalid_input",
            False,
        )
    elif isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        reason, code, retryable = ToolFailureReason.TOOL_REPORTED, "unauthorized", False
    elif isinstance(exc, NotFoundError):
        reason, code, retryable = ToolFailureReason.TOOL_REPORTED, "not_found", False
    elif isinstance(exc, ConflictError):
        reason, code, retryable = ToolFailureReason.TOOL_REPORTED, "conflict", False
    elif isinstance(exc, RateLimitError):
        reason, code, retryable = ToolFailureReason.USAGE_LIMIT, "rate_limited", True
    elif isinstance(exc, GoodMemError):
        reason, code, retryable = ToolFailureReason.TOOL_REPORTED, "goodmem_error", True
    else:
        reason, code, retryable = ToolFailureReason.EXCEPTION, "unexpected", False

    body = getattr(exc, "body", None)
    details: dict[str, Any] = {}
    if body:
        details["server_response"] = str(body)[:1000]
    if extra_details:
        details.update(extra_details)
    # ToolFailure is frozen, so everything is decided before construction.
    return ToolFailure(
        message=f"{context}: {exc}{extra_message}",
        reason=reason,
        code=code,
        retryable=retryable,
        details=details,
    )


class _GoodMemBaseTool(GoodMemConnection, BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_GOODMEM_ENV_VARS))


# ============================================================ search (P28)
class SearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="What to look for, in natural language.",
    )


class GoodMemSearchTool(_GoodMemBaseTool):
    """Semantic search over spaces the developer configured.

    The agent supplies only a query. Spaces, result count, reranking and
    metadata filters are set here, by you, so a model cannot redirect the
    search to another space or change retrieval behaviour mid-run.
    """

    name: str = "GoodMemSearch"
    description: str = (
        "Search stored knowledge for passages relevant to a natural-language "
        "query. Returns matching passages with their sources."
    )
    args_schema: type[BaseModel] = SearchSchema

    space_ids: list[str] = Field(..., min_length=1)
    k: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(default=None, gt=0)
    reranker_id: str | None = None
    filter: str | None = None
    metadata_filter: dict[str, Any] | None = None

    def _run(self, query: str) -> Any:
        if not query.strip():
            return _failure(ValueError("query must not be empty"), "Search rejected")

        expression = combine(
            self.filter,
            from_mapping(self.metadata_filter) if self.metadata_filter else None,
        )
        reranked = bool(self.reranker_id)
        kwargs: dict[str, Any] = {
            "message": query,
            "requested_size": self.fetch_k or self.k,
            "fetch_memory": True,
            "stream": False,
        }
        if expression is None:
            kwargs["space_ids"] = list(self.space_ids)
        else:
            kwargs["space_keys"] = [
                {"spaceId": sid, "filter": expression} for sid in self.space_ids
            ]
        if reranked:
            kwargs["reranker_id"] = self.reranker_id
            kwargs["max_results"] = self.k

        try:
            with self._session() as client:
                events = list(client.memories.retrieve(**kwargs))
        except Exception as exc:
            return _failure(exc, "Search failed")

        statuses, degraded = classify(events)
        hits = hits_from_events(events, reranked=reranked)[: self.k]

        # A search that failed outright must not be mistaken for one that
        # simply found nothing.
        if degraded and not hits:
            return ToolFailure(
                message="Search failed: "
                + "; ".join(f"{s.get('code')}: {s.get('message')}" for s in statuses),
                reason=ToolFailureReason.TOOL_REPORTED,
                code="retrieval_failed",
                retryable=True,
                details={"statuses": json.dumps(statuses)[:1000]},
            )

        payload: dict[str, Any] = {
            "query": query,
            "results": hits,
            "total_results": len(hits),
            # True when some part of the search did not complete: the results
            # below are usable but incomplete.
            "partial": degraded,
        }
        if statuses:
            payload["statuses"] = statuses
        if reply := abstract_reply(events):
            payload["abstract_reply"] = reply
        return json.dumps(payload, default=str)


# ============================================================ discovery
class _Empty(BaseModel):
    pass


class GoodMemListSpacesTool(_GoodMemBaseTool):
    name: str = "GoodMemListSpaces"
    description: str = (
        "List GoodMem spaces, returning each space's ID, name and embedders."
    )
    args_schema: type[BaseModel] = _Empty
    max_items: int | None = 100

    def _run(self) -> Any:
        try:
            with self._session() as client:
                # Follows pagination through the SDK rather than reading only
                # the server's first page.
                spaces = [
                    s.model_dump(exclude_none=True)
                    for s in client.spaces.list(max_items=self.max_items)
                ]
        except Exception as exc:
            return _failure(exc, "Failed to list spaces")
        return json.dumps(
            {
                "spaces": spaces,
                "returned": len(spaces),
                "truncated": self.max_items is not None
                and len(spaces) >= self.max_items,
            },
            default=str,
        )


class GoodMemListEmbeddersTool(_GoodMemBaseTool):
    name: str = "GoodMemListEmbedders"
    description: str = "List embedders available for creating GoodMem spaces."
    args_schema: type[BaseModel] = _Empty
    max_items: int | None = 100

    def _run(self) -> Any:
        try:
            with self._session() as client:
                items = [
                    e.model_dump(exclude_none=True)
                    for e in client.embedders.list(max_items=self.max_items)
                ]
        except Exception as exc:
            return _failure(exc, "Failed to list embedders")
        return json.dumps({"embedders": items, "returned": len(items)}, default=str)


class GoodMemListRerankersTool(_GoodMemBaseTool):
    name: str = "GoodMemListRerankers"
    description: str = "List rerankers available to improve search result ordering."
    args_schema: type[BaseModel] = _Empty
    max_items: int | None = 100

    def _run(self) -> Any:
        try:
            with self._session() as client:
                items = [
                    r.model_dump(exclude_none=True)
                    for r in client.rerankers.list(max_items=self.max_items)
                ]
        except Exception as exc:
            return _failure(exc, "Failed to list rerankers")
        return json.dumps({"rerankers": items, "returned": len(items)}, default=str)


# ============================================================ spaces
class GetSpaceSchema(BaseModel):
    space_id: str = Field(..., description="The UUID of the space.")


class GoodMemGetSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemGetSpace"
    description: str = "Fetch one GoodMem space by ID, with its embedders and labels."
    args_schema: type[BaseModel] = GetSpaceSchema

    def _run(self, space_id: str) -> Any:
        try:
            with self._session() as client:
                return json.dumps(
                    client.spaces.get(id=space_id).model_dump(exclude_none=True),
                    default=str,
                )
        except Exception as exc:
            return _failure(exc, "Failed to get space")


class CreateSpaceSchema(BaseModel):
    name: str = Field(..., description="A name for the new space.")


class GoodMemCreateSpaceTool(_GoodMemBaseTool):
    """Create a space using the embedder and chunking this tool was given.

    Creation always creates. It does not look up a space by name and quietly
    hand back a different one: a name collision is reported as a conflict so
    the caller decides. Chunking is configured here by the developer, not
    chosen by the model.
    """

    name: str = "GoodMemCreateSpace"
    description: str = (
        "Create a new GoodMem space to store memories in. Fails if a space "
        "with that name already exists."
    )
    args_schema: type[BaseModel] = CreateSpaceSchema

    embedder_id: str = Field(..., description="Embedder to attach (developer-set).")
    chunking_config: dict[str, Any] | None = None

    def _run(self, name: str) -> Any:
        try:
            with self._session() as client:
                kwargs: dict[str, Any] = {
                    "name": name,
                    "space_embedders": [{"embedderId": self.embedder_id}],
                }
                if self.chunking_config:
                    kwargs["default_chunking_config"] = self.chunking_config
                space = client.spaces.create(**kwargs)
            return json.dumps(space.model_dump(exclude_none=True), default=str)
        except ConflictError as exc:
            return ToolFailure(
                message=(
                    f"A space named {name!r} already exists. Use GoodMemListSpaces "
                    "to find its ID instead of creating a duplicate."
                ),
                reason=ToolFailureReason.TOOL_REPORTED,
                code="conflict",
                retryable=False,
                details={"server_response": str(getattr(exc, "body", ""))[:500]},
            )
        except Exception as exc:
            return _failure(exc, "Failed to create space")


class UpdateSpaceSchema(BaseModel):
    space_id: str = Field(..., description="The UUID of the space to update.")
    name: str | None = Field(default=None, description="New name for the space.")
    merge_labels: dict[str, str] | None = Field(
        default=None,
        description="Labels to merge into the existing set (at most 20 entries).",
    )
    replace_labels: dict[str, str] | None = Field(
        default=None,
        description="Labels replacing the existing set (at most 20 entries).",
    )


class GoodMemUpdateSpaceTool(_GoodMemBaseTool):
    """Update a space's mutable fields: name and labels.

    ``public_read`` is gone. GoodMem removed the field from spaces entirely and
    now rejects it; shared access is granted through authorization grants.
    """

    name: str = "GoodMemUpdateSpace"
    description: str = "Rename a GoodMem space or change its labels."
    args_schema: type[BaseModel] = UpdateSpaceSchema

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        merge_labels: dict[str, str] | None = None,
        replace_labels: dict[str, str] | None = None,
    ) -> Any:
        if merge_labels and replace_labels:
            return _failure(
                ValueError("Use merge_labels or replace_labels, not both."),
                "Update rejected",
            )
        if name is None and not merge_labels and not replace_labels:
            return _failure(
                ValueError(
                    "Nothing to update: pass name, merge_labels or replace_labels."
                ),
                "Update rejected",
            )
        request: dict[str, Any] = {}
        if name is not None:
            request["name"] = name
        if merge_labels:
            request["mergeLabels"] = merge_labels
        if replace_labels:
            request["replaceLabels"] = replace_labels
        try:
            with self._session() as client:
                space = client.spaces.update(id=space_id, request=request)
            return json.dumps(space.model_dump(exclude_none=True), default=str)
        except Exception as exc:
            return _failure(exc, "Failed to update space")


class DeleteSpaceSchema(BaseModel):
    space_id: str = Field(..., description="The UUID of the space to delete.")


class GoodMemDeleteSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemDeleteSpace"
    description: str = (
        "Permanently delete a GoodMem space and every memory in it. Cannot be undone."
    )
    args_schema: type[BaseModel] = DeleteSpaceSchema

    def _run(self, space_id: str) -> Any:
        try:
            with self._session() as client:
                client.spaces.delete(id=space_id)
            return json.dumps({"deleted": True, "space_id": space_id})
        except Exception as exc:
            return _failure(exc, "Failed to delete space")


# ============================================================ memories
class CreateMemorySchema(BaseModel):
    text_content: str = Field(..., description="The text to remember.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata. A 'title' helps some embedders.",
    )


class GoodMemCreateMemoryTool(_GoodMemBaseTool):
    """Store text in the space this tool was configured with.

    Waits for the new memory to finish indexing by default, so a later search
    does not have to poll for it. On a wait failure the memory ID is still
    reported, so the caller can check its status instead of writing it again.
    """

    name: str = "GoodMemCreateMemory"
    description: str = "Store a piece of text so it can be found by later searches."
    args_schema: type[BaseModel] = CreateMemorySchema

    space_id: str = Field(..., description="Target space (developer-set).")
    wait: bool = True
    indexing_timeout: float = 120.0

    def _run(self, text_content: str, metadata: dict[str, Any] | None = None) -> Any:
        if not text_content.strip():
            return _failure(ValueError("text_content must not be empty"), "Rejected")
        memory_id = None
        try:
            with self._session() as client:
                memory = client.memories.create(
                    space_id=self.space_id,
                    original_content=text_content,
                    content_type="text/plain",
                    metadata=metadata or None,
                )
                memory_id = memory.memory_id
                status = memory.processing_status
                if self.wait:
                    status = _wait_one(client, memory_id, self.indexing_timeout)
            return json.dumps(
                {"memory_id": memory_id, "space_id": self.space_id, "status": status},
                default=str,
            )
        except Exception as exc:
            # If the write was accepted and only the wait failed, say so, so
            # the caller checks status instead of writing a duplicate.
            return _failure(
                exc,
                "Failed to store memory",
                extra_message=(
                    f" (memory {memory_id} was created; check its processing "
                    "status rather than storing it again)"
                    if memory_id
                    else ""
                ),
                extra_details={"memory_id": memory_id} if memory_id else None,
            )


class UploadFileSchema(BaseModel):
    file_name: str = Field(
        ...,
        description="Name of a file inside the configured upload directory.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata."
    )


class GoodMemUploadFileTool(_GoodMemBaseTool):
    """Upload a file from one configured directory. Opt-in, and off by default.

    ``upload_dir`` must be set explicitly. Paths that resolve outside it,
    including via symlinks, are refused before the file is opened, so the
    model cannot name an arbitrary path on the host.
    """

    name: str = "GoodMemUploadFile"
    description: str = (
        "Store a file from the approved upload directory so its contents can "
        "be found by later searches."
    )
    args_schema: type[BaseModel] = UploadFileSchema

    space_id: str = Field(..., description="Target space (developer-set).")
    upload_dir: str | None = Field(
        default=None,
        description="Directory whose files agents may upload. Required.",
    )
    wait: bool = True
    indexing_timeout: float = 180.0

    _MIME: ClassVar[dict[str, str]] = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    def _run(self, file_name: str, metadata: dict[str, Any] | None = None) -> Any:
        try:
            path = resolve_upload_path(file_name, self.upload_dir)
        except GoodMemUploadError as exc:
            return _failure(exc, "Upload refused")

        content_type = self._MIME.get(path.suffix.lower(), "application/octet-stream")
        raw = path.read_bytes()
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("title", path.name)
        kwargs: dict[str, Any] = {
            "space_id": self.space_id,
            "content_type": content_type,
            "metadata": merged_metadata,
        }
        if content_type.startswith("text/"):
            kwargs["original_content"] = raw.decode("utf-8", errors="replace")
        else:
            kwargs["original_content_b64"] = base64.b64encode(raw).decode("ascii")

        memory_id = None
        try:
            with self._session() as client:
                memory = client.memories.create(**kwargs)
                memory_id = memory.memory_id
                status = memory.processing_status
                if self.wait:
                    status = _wait_one(client, memory_id, self.indexing_timeout)
            return json.dumps(
                {
                    "memory_id": memory_id,
                    "file": path.name,
                    "content_type": content_type,
                    "status": status,
                },
                default=str,
            )
        except Exception as exc:
            return _failure(
                exc,
                "Failed to upload file",
                extra_details={"memory_id": memory_id} if memory_id else None,
            )


class ListMemoriesSchema(BaseModel):
    space_id: str = Field(..., description="The UUID of the space.")
    status_filter: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] | None = (
        Field(
            default=None, description="Only return memories in this processing state."
        )
    )


class GoodMemListMemoriesTool(_GoodMemBaseTool):
    name: str = "GoodMemListMemories"
    description: str = "List the memories stored in a GoodMem space."
    args_schema: type[BaseModel] = ListMemoriesSchema
    max_items: int | None = 100
    include_content: bool = False

    def _run(
        self,
        space_id: str,
        status_filter: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
        | None = None,
    ) -> Any:
        try:
            with self._session() as client:
                page = client.memories.list(
                    space_id=space_id,
                    status_filter=status_filter,
                    include_content=self.include_content or None,
                    max_items=self.max_items,
                )
                memories = [m.model_dump(exclude_none=True) for m in page]
        except Exception as exc:
            return _failure(exc, "Failed to list memories")
        return json.dumps(
            {
                "memories": memories,
                "returned": len(memories),
                "truncated": self.max_items is not None
                and len(memories) >= self.max_items,
            },
            default=str,
        )


class GetMemorySchema(BaseModel):
    memory_id: str = Field(..., description="The UUID of the memory.")
    include_content: bool = Field(
        default=True, description="Include the stored content, not only metadata."
    )


_TEXTUAL_TYPES = ("text/", "application/json", "application/xml", "+json", "+xml")


def _charset_of(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "charset" and value:
            return value.strip().strip('"')
    return "utf-8"


class GoodMemGetMemoryTool(_GoodMemBaseTool):
    """Fetch one memory. Content arrives in the same request as the metadata.

    The SDK hands back ``original_content`` as raw bytes. Textual content is
    decoded using the declared charset and returned as readable text; binary
    content is described rather than dumped into the agent's context as
    base64, which it cannot use anyway.
    """

    name: str = "GoodMemGetMemory"
    description: str = "Fetch a stored memory by ID, with its metadata and content."
    args_schema: type[BaseModel] = GetMemorySchema

    def _run(self, memory_id: str, include_content: bool = True) -> Any:
        try:
            with self._session() as client:
                memory = client.memories.get(
                    id=memory_id, include_content=include_content or None
                )
        except Exception as exc:
            return _failure(exc, "Failed to get memory")

        payload = memory.model_dump(exclude_none=True)
        # model_dump() re-encodes the content as base64; the attribute holds
        # the decoded bytes, which is what we actually want to render.
        payload.pop("original_content", None)
        raw = memory.original_content
        content_type = memory.content_type or ""

        if raw is not None:
            if isinstance(raw, str):
                raw = raw.encode()
            if any(marker in content_type for marker in _TEXTUAL_TYPES):
                charset = _charset_of(content_type)
                try:
                    payload["content"] = raw.decode(charset)
                except (LookupError, UnicodeDecodeError) as exc:
                    # Say what went wrong rather than silently mangling text.
                    payload["content_error"] = (
                        f"Content could not be decoded as {charset}: {exc}"
                    )
            else:
                payload["content_omitted"] = (
                    f"{len(raw)} bytes of {content_type or 'binary'} content is not "
                    "included; fetch it through the SDK if you need the bytes."
                )
        return json.dumps(payload, default=str)


class DeleteMemorySchema(BaseModel):
    memory_id: str = Field(..., description="The UUID of the memory to delete.")


class GoodMemDeleteMemoryTool(_GoodMemBaseTool):
    name: str = "GoodMemDeleteMemory"
    description: str = "Permanently delete a stored memory. Cannot be undone."
    args_schema: type[BaseModel] = DeleteMemorySchema

    def _run(self, memory_id: str) -> Any:
        try:
            with self._session() as client:
                client.memories.delete(id=memory_id)
            return json.dumps({"deleted": True, "memory_id": memory_id})
        except Exception as exc:
            return _failure(exc, "Failed to delete memory")


# ============================================================ helpers
def _wait_one(client: Any, memory_id: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while True:
        status = str(client.memories.get(id=memory_id).processing_status)
        if status in _TERMINAL:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"memory {memory_id} was still {status} after {timeout}s"
            )
        time.sleep(0.5)


def wait_for_memories(
    memory_ids: list[str],
    *,
    connection: GoodMemConnection | None = None,
    timeout: float = 120.0,
) -> dict[str, str]:
    """Wait for specific memory IDs to finish indexing.

    Use this after writing, instead of searching repeatedly and hoping results
    appear. Searching is not a way to wait.
    """
    if not memory_ids:
        return {}
    conn = connection or GoodMemConnection()
    out: dict[str, str] = {}
    with conn._session() as client:
        for memory_id in dict.fromkeys(memory_ids):
            out[memory_id] = _wait_one(client, memory_id, timeout)
    return out
