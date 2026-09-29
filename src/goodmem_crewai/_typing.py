"""Typing for the SDK client.

The SDK attaches its API groups (``spaces``, ``memories``, …) to the client
dynamically, so they carry no annotations and a type checker cannot see them.
These Protocols describe only the surface this package uses. They add no
runtime layer: the objects passed around are the SDK's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class _SpacesAPI(Protocol):
    def list(self, **kwargs: Any) -> Any: ...
    def get(self, *, id: str) -> Any: ...
    def create(self, **kwargs: Any) -> Any: ...
    def update(self, *, id: str, request: Any) -> Any: ...
    def delete(self, *, id: str) -> None: ...


class _MemoriesAPI(Protocol):
    def list(self, **kwargs: Any) -> Any: ...
    def get(self, **kwargs: Any) -> Any: ...
    def create(self, **kwargs: Any) -> Any: ...
    def delete(self, *, id: str) -> None: ...
    def retrieve(self, **kwargs: Any) -> Any: ...
    # Sequence, not list: `def list` above shadows the builtin in this scope.
    def batch_create(self, *, requests: Sequence[Any]) -> Any: ...


class _RegistryAPI(Protocol):
    """``embedders`` and ``rerankers``.

    Their ``list()`` is not paginated: it takes only these filters and returns
    every match as a plain list. The signature is spelled out, not
    ``**kwargs``, so a keyword the SDK does not accept (``max_items``, which
    only the paginated ``spaces``/``memories`` lists take) fails type checking
    instead of failing every call at runtime.
    """

    def list(
        self,
        *,
        label: dict[str, str] | None = None,
        owner_id: str | None = None,
    ) -> Sequence[Any]: ...


class GoodmemClient(Protocol):
    """The parts of ``goodmem.Goodmem`` this package calls."""

    @property
    def spaces(self) -> _SpacesAPI: ...
    @property
    def memories(self) -> _MemoriesAPI: ...
    @property
    def embedders(self) -> _RegistryAPI: ...
    @property
    def rerankers(self) -> _RegistryAPI: ...

    def close(self) -> None: ...
