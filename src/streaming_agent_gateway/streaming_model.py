"""SDK-independent streaming model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Input accepted by every model implementation."""

    run_id: UUID
    prompt: str
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelDelta:
    """A provider-neutral incremental model output."""

    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    usage: "ModelUsage | None" = None


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Optional usage information reported after streaming completes."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ModelStreamError(Exception):
    """A normalized model failure that can be translated into a RunEvent."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelCancelled(ModelStreamError):
    """The model stopped because the run cancellation signal was set."""

    def __init__(self, message: str = "model stream cancelled") -> None:
        super().__init__("model_cancelled", message)


class StreamingModel(ABC):
    """Minimal async streaming contract, independent of any provider SDK."""

    @abstractmethod
    def stream(
        self,
        request: ModelRequest,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[ModelDelta]:
        """Yield deltas until completion or cooperative cancellation.

        Implementations should check ``cancel_event.is_set()`` between provider
        reads and raise ``ModelCancelled`` after stopping the provider request.
        """

    async def collect(
        self,
        request: ModelRequest,
        cancel_event: asyncio.Event,
    ) -> list[ModelDelta]:
        """Consume the stream while preserving the same cancellation contract."""

        deltas: list[ModelDelta] = []
        async for delta in self.stream(request, cancel_event):
            deltas.append(delta)
        return deltas


def raise_if_cancelled(cancel_event: asyncio.Event) -> None:
    """Shared cooperative-cancellation check for model adapters."""

    if cancel_event.is_set():
        raise ModelCancelled()
