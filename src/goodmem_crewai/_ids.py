"""GoodMem ids are UUIDs. Anything else is refused before a request is made.

The SDK puts ids into URL paths unescaped (``f"/v1/memories/{id}"``) and httpx
resolves dot segments before sending, so a memory id of ``../spaces/<id>``
turns ``DELETE /v1/memories/<id>`` into ``DELETE /v1/spaces/<id>``. The server
also decodes ``%2e%2e`` into the same traversal, so neither encoding the id
nor the server can be relied on. Every GoodMem id (memory, space, embedder,
reranker, ...) is a UUID, so this module requires exactly that shape rather
than trying to recognise what is dangerous.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field


UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UUID = re.compile(UUID_PATTERN)

# A model-visible id argument. The JSON schema tells the model the id is a
# UUID; pydantic deliberately does not enforce it. CrewAI validates arguments
# before ``_run`` and records a schema failure as a bare exception, whereas
# require_uuid() in ``_run`` reports it as INVALID_INPUT. The refusal itself
# never depends on the model reading the schema.
UuidStr = Annotated[
    str, Field(json_schema_extra={"format": "uuid", "pattern": UUID_PATTERN})
]


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lowercase canonical UUID, or raise ``ValueError``.

    ``field`` names the argument in the message, so the caller (a model or a
    developer) knows which id to correct.
    """
    # fullmatch, not match: `$` also matches just before a trailing newline.
    if isinstance(value, str) and _UUID.fullmatch(value):
        return value.lower()
    shown = repr(value)
    if len(shown) > 80:
        shown = shown[:77] + "..."
    raise ValueError(
        f"{field} must be a UUID like 123e4567-e89b-12d3-a456-426614174000; got {shown}"
    )
