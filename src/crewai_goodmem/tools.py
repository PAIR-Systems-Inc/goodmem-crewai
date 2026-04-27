from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
import requests


_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mime_from_extension(ext: str) -> str | None:
    return _MIME_TYPES.get(ext.lower().lstrip("."))


def _headers(api_key: str, *, include_content_type: bool = True) -> dict[str, str]:
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _resolve_config(
    *,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str]:
    url = base_url or os.getenv("GOODMEM_BASE_URL", "")
    key = api_key or os.getenv("GOODMEM_API_KEY", "")
    if not url:
        raise ValueError(
            "GoodMem base URL is required. Pass base_url to the tool "
            "constructor or set the GOODMEM_BASE_URL environment variable."
        )
    if not key:
        raise ValueError(
            "GoodMem API key is required. Pass api_key to the tool "
            "constructor or set the GOODMEM_API_KEY environment variable."
        )
    return url.rstrip("/"), key


def _error_payload(prefix: str, exc: requests.RequestException) -> dict[str, Any]:
    error_detail = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            error_detail = response.text
        except Exception:
            error_detail = ""
    return {
        "success": False,
        "error": f"{prefix}: {exc}",
        "details": error_detail,
    }


def _list_embedders(
    base_url: str, api_key: str, *, verify_ssl: bool = True
) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{base_url}/v1/embedders",
        headers=_headers(api_key, include_content_type=False),
        verify=verify_ssl,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, list) else body.get("embedders", [])


def _list_spaces(
    base_url: str, api_key: str, *, verify_ssl: bool = True
) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{base_url}/v1/spaces",
        headers=_headers(api_key, include_content_type=False),
        verify=verify_ssl,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, list) else body.get("spaces", [])


_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})


def wait_for_memories_completed(
    memory_ids: list[str],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    timeout: float = 60.0,
    interval: float = 0.5,
) -> dict[str, str]:
    """Block until every memory has reached a terminal processing status.

    Polls ``GET /v1/memories/{id}`` for each id until its ``processingStatus``
    is ``COMPLETED`` or ``FAILED``, then returns the final status map. Use
    this instead of a fixed sleep after creating memories that you intend to
    search immediately.

    Args:
        memory_ids: Memory IDs to wait on. Duplicates are ignored. An empty
            list returns ``{}`` without making any requests.
        base_url: GoodMem base URL. Falls back to the ``GOODMEM_BASE_URL``
            environment variable.
        api_key: GoodMem API key. Falls back to the ``GOODMEM_API_KEY``
            environment variable.
        verify_ssl: Whether to verify the server's TLS certificate.
        timeout: Maximum seconds to wait before raising ``TimeoutError``.
        interval: Seconds to sleep between polling rounds.

    Returns:
        A dict mapping each memory id to its final ``processingStatus``
        (``COMPLETED`` or ``FAILED``).

    Raises:
        TimeoutError: If any memory is still pending when ``timeout`` elapses.
            The error message lists the still-pending ids.
        ValueError: If credentials cannot be resolved.
        requests.RequestException: If the server returns an HTTP error.
    """
    if not memory_ids:
        return {}

    url, key = _resolve_config(base_url=base_url, api_key=api_key)
    headers = _headers(key, include_content_type=False)

    statuses: dict[str, str] = {}
    pending = list(dict.fromkeys(memory_ids))
    deadline = time.monotonic() + timeout

    while True:
        still_pending: list[str] = []
        for memory_id in pending:
            resp = requests.get(
                f"{url}/v1/memories/{memory_id}",
                headers=headers,
                verify=verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            status = resp.json().get("processingStatus", "")
            if status in _TERMINAL_STATUSES:
                statuses[memory_id] = status
            else:
                still_pending.append(memory_id)

        if not still_pending:
            return statuses

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Memories did not reach a terminal status within {timeout}s. "
                f"Still pending: {still_pending}"
            )

        pending = still_pending
        time.sleep(interval)


class ListEmbeddersSchema(BaseModel):
    pass


class ListSpacesSchema(BaseModel):
    pass


class GetSpaceSchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The UUID of the space to fetch.",
    )


class CreateSpaceSchema(BaseModel):
    name: str = Field(
        ...,
        description=(
            "A unique name for the space. If a space with this name already "
            "exists, its ID is returned instead of creating a duplicate."
        ),
    )
    embedder_id: str = Field(
        ...,
        description=(
            "The ID of the embedder model that converts text into vector "
            "representations for similarity search."
        ),
    )
    chunk_size: int = Field(
        default=256,
        description="Number of characters per chunk when splitting documents.",
    )
    chunk_overlap: int = Field(
        default=25,
        description="Number of overlapping characters between consecutive chunks.",
    )
    keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = Field(
        default="KEEP_END",
        description=(
            "Where to attach the separator when splitting. One of KEEP_END, "
            "KEEP_START, DISCARD."
        ),
    )
    length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = Field(
        default="CHARACTER_COUNT",
        description="How chunk size is measured. One of CHARACTER_COUNT, TOKEN_COUNT.",
    )


class UpdateSpaceSchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The UUID of the space to update.",
    )
    name: str | None = Field(
        default=None,
        description="New name for the space.",
    )
    public_read: bool | None = Field(
        default=None,
        description="Whether to allow unauthenticated read access.",
    )
    replace_labels_json: str | None = Field(
        default=None,
        description=(
            "A JSON string of labels that replace all existing labels, e.g. "
            '\'{"env": "prod"}\'. Mutually exclusive with merge_labels_json.'
        ),
    )
    merge_labels_json: str | None = Field(
        default=None,
        description=(
            "A JSON string of labels that merge into existing labels, e.g. "
            '\'{"team": "ml"}\'. Mutually exclusive with replace_labels_json.'
        ),
    )


class DeleteSpaceSchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The UUID of the space to delete.",
    )


class CreateMemorySchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The ID of the space to store the memory in.",
    )
    text_content: str | None = Field(
        default=None,
        description=(
            "Plain text content to store as memory. If both file_path and "
            "text_content are provided, file_path takes priority."
        ),
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Local file path to store as memory (PDF, DOCX, images, etc.). "
            "Content type is auto-detected from the file extension."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value metadata as a JSON object.",
    )


class ListMemoriesSchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The UUID of the space to list memories from.",
    )
    status_filter: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] | None = (
        Field(
            default=None,
            description="Filter by processing status.",
        )
    )
    include_content: bool = Field(
        default=False,
        description="Include each memory's original content in addition to metadata.",
    )
    sort_by: Literal["created_at", "updated_at"] | None = Field(
        default=None,
        description="Field to sort by.",
    )
    sort_order: Literal["ASCENDING", "DESCENDING"] | None = Field(
        default=None,
        description="Sort direction.",
    )


class RetrieveMemoriesSchema(BaseModel):
    query: str = Field(
        ...,
        description="A natural language query used to find semantically similar memory chunks.",
    )
    space_ids: list[str] = Field(
        ...,
        description="List of space IDs to search across.",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return.",
    )
    include_memory_definition: bool = Field(
        default=True,
        description="Fetch full memory metadata alongside matched chunks.",
    )
    wait_for_indexing: bool = Field(
        default=True,
        description=(
            "Retry when no results are found, useful when memories were just "
            "added and may still be processing."
        ),
    )
    max_wait_seconds: float = Field(
        default=10.0,
        description="Maximum seconds to poll for results when wait_for_indexing is True.",
    )
    poll_interval: float = Field(
        default=2.0,
        description="Seconds to sleep between polling attempts when wait_for_indexing is True.",
    )
    reranker_id: str | None = Field(
        default=None,
        description="Optional reranker model ID to improve result ordering.",
    )
    llm_id: str | None = Field(
        default=None,
        description="Optional LLM ID to generate contextual responses alongside retrieved chunks.",
    )
    relevance_threshold: float | None = Field(
        default=None,
        description="Minimum score (0-1) for including results. Only used with reranker or LLM.",
    )
    llm_temperature: float | None = Field(
        default=None,
        description="Creativity setting for LLM generation (0-2). Only used when llm_id is set.",
    )
    chronological_resort: bool = Field(
        default=False,
        description="Reorder results by creation time instead of relevance score.",
    )
    metadata_filter: str | None = Field(
        default=None,
        description=(
            "A SQL-style JSONPath expression applied server-side to narrow results "
            "by metadata. Example: CAST(val('$.category') AS TEXT) = 'feat'. When "
            "set, the same filter is applied to every space in space_ids."
        ),
    )


class GetMemorySchema(BaseModel):
    memory_id: str = Field(
        ...,
        description="The UUID of the memory to fetch.",
    )
    include_content: bool = Field(
        default=True,
        description="Fetch the original document content in addition to metadata.",
    )


class DeleteMemorySchema(BaseModel):
    memory_id: str = Field(
        ...,
        description="The UUID of the memory to delete.",
    )


_GOODMEM_ENV_VARS = [
    EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
    EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
]


class _GoodMemBaseTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_GOODMEM_ENV_VARS))
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True


class GoodMemListEmbeddersTool(_GoodMemBaseTool):
    name: str = "GoodMemListEmbedders"
    description: str = (
        "List all available embedders in GoodMem. Returns embedder IDs, "
        "display names, and model identifiers that are required when creating "
        "a new space."
    )
    args_schema: type[BaseModel] = ListEmbeddersSchema

    def _run(self) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            embedders = _list_embedders(url, key, verify_ssl=self.verify_ssl)
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to list embedders", exc))

        return json.dumps(
            {
                "success": True,
                "embedders": [
                    {
                        "embedderId": e.get("embedderId") or e.get("id"),
                        "displayName": (
                            e.get("displayName")
                            or e.get("name")
                            or e.get("modelIdentifier")
                            or "Unnamed"
                        ),
                        "modelIdentifier": (
                            e.get("modelIdentifier") or e.get("model") or "unknown"
                        ),
                    }
                    for e in embedders
                ],
                "totalEmbedders": len(embedders),
            }
        )


class GoodMemListSpacesTool(_GoodMemBaseTool):
    name: str = "GoodMemListSpaces"
    description: str = (
        "List all available spaces in GoodMem. Returns space IDs, names, and "
        "attached embedders that can be used with other GoodMem tools."
    )
    args_schema: type[BaseModel] = ListSpacesSchema

    def _run(self) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            spaces = _list_spaces(url, key, verify_ssl=self.verify_ssl)
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to list spaces", exc))

        return json.dumps(
            {
                "success": True,
                "spaces": [
                    {
                        "spaceId": s.get("spaceId") or s.get("id"),
                        "name": s.get("name") or "Unnamed",
                        "spaceEmbedders": s.get("spaceEmbedders", []),
                    }
                    for s in spaces
                ],
                "totalSpaces": len(spaces),
            }
        )


class GoodMemGetSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemGetSpace"
    description: str = (
        "Fetch a GoodMem space by its ID. Returns the full space configuration, "
        "including its embedders, chunking settings, and labels."
    )
    args_schema: type[BaseModel] = GetSpaceSchema

    def _run(self, space_id: str) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.get(
                f"{url}/v1/spaces/{space_id}",
                headers=_headers(key, include_content_type=False),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            return json.dumps(
                {
                    "success": True,
                    "space": resp.json(),
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to get space", exc))


class GoodMemCreateSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemCreateSpace"
    description: str = (
        "Create a new GoodMem space (container for memories) with an embedder "
        "and chunking configuration. If a space with the same name already "
        "exists it is reused."
    )
    args_schema: type[BaseModel] = CreateSpaceSchema

    def _run(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = "KEEP_END",
        length_measurement: Literal[
            "CHARACTER_COUNT", "TOKEN_COUNT"
        ] = "CHARACTER_COUNT",
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            spaces = _list_spaces(url, key, verify_ssl=self.verify_ssl)
            for s in spaces:
                if s.get("name") == name:
                    actual_embedder_id = embedder_id
                    space_embedders = s.get("spaceEmbedders") or []
                    if space_embedders:
                        actual_embedder_id = space_embedders[0].get(
                            "embedderId", embedder_id
                        )
                    return json.dumps(
                        {
                            "success": True,
                            "spaceId": s.get("spaceId") or s.get("id"),
                            "name": s["name"],
                            "embedderId": actual_embedder_id,
                            "message": "Space already exists, reusing existing space",
                            "reused": True,
                        }
                    )
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in (401, 403):
                return json.dumps(
                    _error_payload(
                        "Authentication failed while checking existing spaces",
                        exc,
                    )
                )
            # Other errors (connection/5xx/etc.): fall through to create.

        body: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                },
            },
        }

        try:
            resp = requests.post(
                f"{url}/v1/spaces",
                headers=_headers(key),
                json=body,
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {
                    "success": True,
                    "spaceId": data.get("spaceId"),
                    "name": data.get("name"),
                    "embedderId": embedder_id,
                    "chunkingConfig": body["defaultChunkingConfig"],
                    "message": "Space created successfully",
                    "reused": False,
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to create space", exc))


class GoodMemUpdateSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemUpdateSpace"
    description: str = (
        "Update a GoodMem space. Supports changing the name, toggling public "
        "read access, and replacing or merging metadata labels."
    )
    args_schema: type[BaseModel] = UpdateSpaceSchema

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels_json: str | None = None,
        merge_labels_json: str | None = None,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        if replace_labels_json and merge_labels_json:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Cannot use both replace_labels_json and "
                        "merge_labels_json at the same time."
                    ),
                }
            )

        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if public_read is not None:
            body["publicRead"] = public_read
        if replace_labels_json:
            try:
                body["replaceLabels"] = json.loads(replace_labels_json)
            except json.JSONDecodeError as exc:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"replace_labels_json is not valid JSON: {exc}",
                    }
                )
        if merge_labels_json:
            try:
                body["mergeLabels"] = json.loads(merge_labels_json)
            except json.JSONDecodeError as exc:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"merge_labels_json is not valid JSON: {exc}",
                    }
                )

        try:
            resp = requests.put(
                f"{url}/v1/spaces/{space_id}",
                headers=_headers(key),
                json=body,
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            return json.dumps(
                {
                    "success": True,
                    "space": resp.json(),
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to update space", exc))


class GoodMemDeleteSpaceTool(_GoodMemBaseTool):
    name: str = "GoodMemDeleteSpace"
    description: str = (
        "Permanently delete a GoodMem space and all its memories, chunks, and "
        "vector embeddings. This operation cannot be undone."
    )
    args_schema: type[BaseModel] = DeleteSpaceSchema

    def _run(self, space_id: str) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.delete(
                f"{url}/v1/spaces/{space_id}",
                headers=_headers(key, include_content_type=False),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            return json.dumps(
                {
                    "success": True,
                    "spaceId": space_id,
                    "message": "Space deleted successfully",
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to delete space", exc))


class GoodMemCreateMemoryTool(_GoodMemBaseTool):
    name: str = "GoodMemCreateMemory"
    description: str = (
        "Store a document or plain text as a memory in a GoodMem space. "
        "Provide either a local file path (PDF, DOCX, images, etc.) or plain "
        "text content. The memory is chunked and embedded for semantic search."
    )
    args_schema: type[BaseModel] = CreateMemorySchema

    def _run(
        self,
        space_id: str,
        text_content: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        body: dict[str, Any] = {"spaceId": space_id}

        if file_path:
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            mime = _mime_from_extension(ext) or "application/octet-stream"
            try:
                with open(file_path, "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Cannot read file '{file_path}': {exc}",
                    }
                )

            body["contentType"] = mime
            if mime.startswith("text/"):
                body["originalContent"] = raw.decode("utf-8", errors="replace")
            else:
                body["originalContentB64"] = base64.b64encode(raw).decode("ascii")
        elif text_content:
            body["contentType"] = "text/plain"
            body["originalContent"] = text_content
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": "No content provided. Supply either file_path or text_content.",
                }
            )

        if metadata:
            body["metadata"] = metadata

        try:
            resp = requests.post(
                f"{url}/v1/memories",
                headers=_headers(key),
                json=body,
                verify=self.verify_ssl,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(
                {
                    "success": True,
                    "memoryId": data.get("memoryId"),
                    "spaceId": data.get("spaceId"),
                    "status": data.get("processingStatus", "PENDING"),
                    "contentType": body.get("contentType"),
                    "message": "Memory created successfully",
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to create memory", exc))


class GoodMemListMemoriesTool(_GoodMemBaseTool):
    name: str = "GoodMemListMemories"
    description: str = (
        "List memories in a GoodMem space with optional filtering by processing "
        "status and sort order. Returns metadata and optionally each memory's "
        "original content."
    )
    args_schema: type[BaseModel] = ListMemoriesSchema

    def _run(
        self,
        space_id: str,
        status_filter: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
        | None = None,
        include_content: bool = False,
        sort_by: Literal["created_at", "updated_at"] | None = None,
        sort_order: Literal["ASCENDING", "DESCENDING"] | None = None,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        params: dict[str, str] = {}
        if include_content:
            params["includeContent"] = "true"
        if status_filter:
            params["statusFilter"] = status_filter
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order

        try:
            resp = requests.get(
                f"{url}/v1/spaces/{space_id}/memories",
                headers=_headers(key, include_content_type=False),
                params=params,
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            memories = body if isinstance(body, list) else body.get("memories", [])
            return json.dumps(
                {
                    "success": True,
                    "memories": memories,
                    "totalMemories": len(memories),
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to list memories", exc))


class GoodMemRetrieveMemoriesTool(_GoodMemBaseTool):
    name: str = "GoodMemRetrieveMemories"
    description: str = (
        "Perform semantic search across one or more GoodMem spaces to find "
        "relevant memory chunks. Supports optional reranking, LLM-generated "
        "responses, chronological resort, and server-side metadata filters "
        "(SQL-style JSONPath expressions against memory metadata)."
    )
    args_schema: type[BaseModel] = RetrieveMemoriesSchema

    def _run(
        self,
        query: str,
        space_ids: list[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        max_wait_seconds: float = 10.0,
        poll_interval: float = 2.0,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool = False,
        metadata_filter: str | None = None,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        space_keys: list[dict[str, Any]] = [
            {"spaceId": sid} for sid in space_ids if sid
        ]
        if not space_keys:
            return json.dumps(
                {
                    "success": False,
                    "error": "At least one space ID must be provided.",
                }
            )
        if metadata_filter:
            for space_key in space_keys:
                space_key["filter"] = metadata_filter

        body: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        if reranker_id or llm_id or chronological_resort:
            config: dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True
            body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        start = time.monotonic()

        try:
            while True:
                retrieve_headers = {
                    **_headers(key),
                    "Accept": "application/x-ndjson",
                }
                resp = requests.post(
                    f"{url}/v1/memories:retrieve",
                    headers=retrieve_headers,
                    json=body,
                    verify=self.verify_ssl,
                    timeout=90,
                )
                resp.raise_for_status()

                results: list[dict[str, Any]] = []
                memories: list[dict[str, Any]] = []
                result_set_id = ""
                abstract_reply: dict[str, Any] | None = None

                for line in resp.text.strip().split("\n"):
                    json_str = line.strip()
                    if not json_str:
                        continue
                    if json_str.startswith("data:"):
                        json_str = json_str[5:].strip()
                    if json_str.startswith("event:") or json_str == "":
                        continue
                    try:
                        item = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    if item.get("resultSetBoundary"):
                        result_set_id = item["resultSetBoundary"].get("resultSetId", "")
                    elif item.get("memoryDefinition"):
                        memories.append(item["memoryDefinition"])
                    elif item.get("abstractReply"):
                        abstract_reply = item["abstractReply"]
                    elif item.get("retrievedItem"):
                        chunk_data = item["retrievedItem"].get("chunk", {})
                        chunk = chunk_data.get("chunk", {})
                        results.append(
                            {
                                "chunkId": chunk.get("chunkId"),
                                "chunkText": chunk.get("chunkText"),
                                "memoryId": chunk.get("memoryId"),
                                "relevanceScore": chunk_data.get("relevanceScore"),
                                "memoryIndex": chunk_data.get("memoryIndex"),
                            }
                        )

                last_result = {
                    "success": True,
                    "resultSetId": result_set_id,
                    "results": results,
                    "memories": memories,
                    "totalResults": len(results),
                    "query": query,
                }
                if abstract_reply:
                    last_result["abstractReply"] = abstract_reply

                if results or not wait_for_indexing:
                    return json.dumps(last_result)

                if time.monotonic() - start >= max_wait_seconds:
                    last_result["message"] = (
                        f"No results found after waiting {max_wait_seconds}s for "
                        "indexing. Memories may still be processing."
                    )
                    return json.dumps(last_result)

                time.sleep(poll_interval)

        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to retrieve memories", exc))


class GoodMemGetMemoryTool(_GoodMemBaseTool):
    name: str = "GoodMemGetMemory"
    description: str = (
        "Fetch a specific memory by its ID, including metadata, processing "
        "status, and optionally the original content. Binary content is "
        "returned base64-encoded."
    )
    args_schema: type[BaseModel] = GetMemorySchema

    def _run(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.get(
                f"{url}/v1/memories/{memory_id}",
                headers=_headers(key, include_content_type=False),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to get memory", exc))

        result: dict[str, Any] = {
            "success": True,
            "memory": resp.json(),
        }

        if include_content:
            try:
                content_resp = requests.get(
                    f"{url}/v1/memories/{memory_id}/content",
                    headers=_headers(key, include_content_type=False),
                    verify=self.verify_ssl,
                    timeout=30,
                )
                content_resp.raise_for_status()
                content_type = content_resp.headers.get("Content-Type", "")
                if "text" in content_type or "json" in content_type:
                    result["content"] = content_resp.text
                    result["contentEncoding"] = "text"
                else:
                    result["content"] = base64.b64encode(content_resp.content).decode(
                        "ascii"
                    )
                    result["contentEncoding"] = "base64"
                result["contentType"] = content_type
            except requests.RequestException as content_exc:
                result["contentError"] = f"Failed to fetch content: {content_exc}"

        return json.dumps(result)


class GoodMemDeleteMemoryTool(_GoodMemBaseTool):
    name: str = "GoodMemDeleteMemory"
    description: str = (
        "Permanently delete a memory and all its associated chunks and vector "
        "embeddings from GoodMem."
    )
    args_schema: type[BaseModel] = DeleteMemorySchema

    def _run(self, memory_id: str) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.delete(
                f"{url}/v1/memories/{memory_id}",
                headers=_headers(key, include_content_type=False),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            return json.dumps(
                {
                    "success": True,
                    "memoryId": memory_id,
                    "message": "Memory deleted successfully",
                }
            )
        except requests.RequestException as exc:
            return json.dumps(_error_payload("Failed to delete memory", exc))
