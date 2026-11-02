"""Application-facing submit, subscribe, and cancel operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from .agent_loop import AgentLoop, RunNotActiveError
from .events import RunEvent
from .run_store import RunNotFoundError, RunSnapshot, RunStore
from .streaming_model import ModelRequest


class RunGateway:
    """Coordinate background runs and event subscriptions."""

    def __init__(self, agent_loop: AgentLoop, store: RunStore) -> None:
        self._agent_loop = agent_loop
        self._store = store

    async def submit(self, request: ModelRequest) -> UUID:
        """Start a run and return immediately with its stable ID."""

        await self._agent_loop.start(request)
        return request.run_id

    async def cancel(self, run_id: UUID) -> None:
        """Request cancellation of an active run."""

        await self._agent_loop.cancel(run_id)

    async def subscribe(
        self,
        run_id: UUID,
        after_sequence: int = 0,
        poll_interval: float = 0.01,
    ) -> AsyncIterator[RunEvent]:
        """Replay and then follow a run until its terminal event.

        ``after_sequence`` is exclusive, matching the SSE ``Last-Event-ID``
        cursor. Closing this generator only disconnects the subscriber; it does
        not cancel the underlying run.
        """

        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")

        cursor = after_sequence
        while True:
            events = self._store.replay(run_id, after_sequence=cursor)
            for event in events:
                cursor = event.sequence
                yield event
                if event.is_terminal:
                    return
            run_snapshot = self._store.get(run_id)
            if run_snapshot.is_terminal:
                for event in self._store.replay(run_id, after_sequence=cursor):
                    cursor = event.sequence
                    yield event
                    if event.is_terminal:
                        return
                return
            await asyncio.sleep(poll_interval)

    def snapshot(self, run_id: UUID) -> RunSnapshot:
        """Return the current event-backed state for diagnostics."""

        return self._store.get(run_id)

    def list_runs(self) -> tuple[RunSnapshot, ...]:
        """Return all known runs in newest-first order."""

        return self._store.list_runs()

    def usage(self, run_id: UUID):
        """Return persisted token usage when the configured store supports it."""

        usage = getattr(self._store, "usage", None)
        return usage(run_id) if usage is not None else None

    def active_run_count(self) -> int:
        return self._agent_loop.active_run_count()

    def active_run_ids(self) -> tuple[UUID, ...]:
        return self._agent_loop.active_run_ids()


__all__ = ["RunGateway", "RunNotActiveError", "RunNotFoundError"]
