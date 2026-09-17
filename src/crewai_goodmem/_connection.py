"""Connection settings and SDK client ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from typing import Any, cast

from goodmem import Goodmem
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from crewai_goodmem._typing import GoodmemClient


DEFAULT_TIMEOUT = 60.0


class GoodMemConnection(BaseModel):
    """Shared connection configuration for every GoodMem component.

    Either inject a caller-owned ``client`` (the connection pool and TLS
    configuration are then yours, and this class never closes it), or supply
    ``base_url``/``api_key`` and let each operation open and close its own
    client. There is no process-wide client cache.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str | None = None
    # Excluded from dumps and masked in repr: a tool instance is frequently
    # serialized into traces and agent state.
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    verify_ssl: bool | str = True
    timeout: float = DEFAULT_TIMEOUT
    client: Any = Field(default=None, exclude=True, repr=False)

    def _resolve(self) -> tuple[str, str]:
        url = self.base_url or os.getenv("GOODMEM_BASE_URL", "")
        key = (
            self.api_key.get_secret_value()
            if self.api_key is not None
            else os.getenv("GOODMEM_API_KEY", "")
        )
        if not url:
            raise ValueError(
                "GoodMem base URL is required. Pass base_url, inject a client, "
                "or set GOODMEM_BASE_URL."
            )
        if not key:
            raise ValueError(
                "GoodMem API key is required. Pass api_key, inject a client, "
                "or set GOODMEM_API_KEY."
            )
        return url.rstrip("/"), key

    @contextmanager
    def _session(self) -> Iterator[GoodmemClient]:
        """Yield an SDK client, closing it only when this object created it."""
        if self.client is not None:
            # An injected client keeps its own server, credentials and TLS
            # settings. Environment variables must not redirect it.
            yield self.client
            return

        url, key = self._resolve()
        client = Goodmem(
            base_url=url,
            api_key=key,
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        try:
            # The SDK builds its API groups at runtime, so the static type
            # does not advertise them; GoodmemClient describes what we call.
            yield cast(GoodmemClient, client)
        finally:
            client.close()
