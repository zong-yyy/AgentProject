"""Thread-safe append-only storage for active runs and their events."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import UUID, uuid4

from .events import RunEvent, RunEventType, RunTerminalState
from .streaming_model import ModelUsage


class RunStoreError(Exception):
    """Base class for Run Store failures."""


class RunNotFoundError(RunStoreError):
    """The requested run does not exist in this store."""


class RunAlreadyTerminalError(RunStoreError):
    """An event was appended after the run reached a terminal state."""


class RunSequenceError(RunStoreError):
    """An externally supplied event did not match the next sequence."""


class RunStoreClosedError(RunStoreError):
    """An operation was attempted after the store was closed."""


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """A point-in-time view of a run without exposing mutable store state."""

    run_id: UUID
    events: tuple[RunEvent, ...]
    terminal_state: RunTerminalState | None

    @property
    def is_terminal(self) -> bool:
        return self.terminal_state is not None


class RunStore(Protocol):
    """Core storage operations required by the agent loop and gateway."""

    def create_run(self, run_id: UUID | None = None, **payload: Any) -> RunEvent: ...

    def append(
        self,
        run_id: UUID,
        event_type: RunEventType,
        payload: Mapping[str, Any] | None = None,
        *,
        terminal_state: RunTerminalState | None = None,
        error_code: str | None = None,
        checkpoint_id: str | None = None,
    ) -> RunEvent: ...

    def replay(
        self,
        run_id: UUID,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]: ...

    def get(self, run_id: UUID) -> RunSnapshot: ...

    def list_runs(self) -> tuple[RunSnapshot, ...]: ...

    def active_run_count(self) -> int: ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Optional capability for stores that persist recovery checkpoints."""

    def save_checkpoint(
        self,
        run_id: UUID,
        sequence: int,
        state: Mapping[str, Any],
        checkpoint_id: str | None = None,
    ) -> object: ...


@runtime_checkable
class TTFTStore(Protocol):
    """Optional capability for stores that persist time-to-first-token."""

    def record_ttft(self, run_id: UUID, ttft_ms: float) -> None: ...


@runtime_checkable
class UsageStore(Protocol):
    """Optional capability for stores that persist model token usage."""

    def record_usage(self, run_id: UUID, usage: ModelUsage) -> None: ...


@dataclass(slots=True)
class _StoredRun:
    events: list[RunEvent]
    terminal_state: RunTerminalState | None = None
    usage: ModelUsage | None = None


class InMemoryRunStore:
    """A process-local Run Store with atomic append and replay operations."""

    def __init__(self) -> None:
        self._runs: dict[UUID, _StoredRun] = {}
        self._lock = threading.RLock()

    def create_run(self, run_id: UUID | None = None, **payload: Any) -> RunEvent:
        """Create a run and append its first ``run.started`` event."""

        run_id = uuid4() if run_id is None else run_id
        with self._lock:
            if run_id in self._runs:
                raise RunStoreError(f"run already exists: {run_id}")
            self._runs[run_id] = _StoredRun(events=[])
            return self._append_locked(run_id, RunEventType.STARTED, payload)

    def append(
        self,
        run_id: UUID,
        event_type: RunEventType,
        payload: Mapping[str, Any] | None = None,
        *,
        terminal_state: RunTerminalState | None = None,
        error_code: str | None = None,
        checkpoint_id: str | None = None,
    ) -> RunEvent:
        """Atomically allocate the next sequence and append an event."""

        with self._lock:
            return self._append_locked(
                run_id,
                event_type,
                {} if payload is None else dict(payload),
                terminal_state=terminal_state,
                error_code=error_code,
                checkpoint_id=checkpoint_id,
            )

    def append_event(self, event: RunEvent) -> RunEvent:
        """Append a prebuilt event, validating its run and next sequence."""

        with self._lock:
            stored = self._get_locked(event.run_id)
            if stored.terminal_state is not None:
                raise RunAlreadyTerminalError(f"run is terminal: {event.run_id}")
            expected_sequence = len(stored.events) + 1
            if event.sequence != expected_sequence:
                raise RunSequenceError(
                    f"expected sequence {expected_sequence}, got {event.sequence}"
                )
            stored.events.append(event)
            if event.terminal_state is not None:
                stored.terminal_state = event.terminal_state
            return event

    def replay(self, run_id: UUID, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        """Return events strictly after a reconnect cursor."""

        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        with self._lock:
            stored = self._get_locked(run_id)
            return tuple(stored.events[after_sequence:])

    def get(self, run_id: UUID) -> RunSnapshot:
        """Return an immutable snapshot of the run."""

        with self._lock:
            stored = self._get_locked(run_id)
            return RunSnapshot(run_id, tuple(stored.events), stored.terminal_state)

    def list_runs(self) -> tuple[RunSnapshot, ...]:
        """Return run snapshots in newest-first creation order."""

        with self._lock:
            return tuple(
                RunSnapshot(run_id, tuple(stored.events), stored.terminal_state)
                for run_id, stored in reversed(self._runs.items())
            )

    def active_run_count(self) -> int:
        """Return the number of runs that have not reached a terminal state."""

        with self._lock:
            return sum(stored.terminal_state is None for stored in self._runs.values())

    def record_usage(self, run_id: UUID, usage: ModelUsage) -> None:
        if any(
            value is not None and (not isinstance(value, int) or value < 0)
            for value in (usage.input_tokens, usage.output_tokens, usage.total_tokens)
        ):
            raise RunStoreError("token usage must contain non-negative integer values")
        with self._lock:
            stored = self._get_locked(run_id)
            stored.usage = usage

    def usage(self, run_id: UUID) -> ModelUsage | None:
        with self._lock:
            return self._get_locked(run_id).usage

    def _append_locked(
        self,
        run_id: UUID,
        event_type: RunEventType,
        payload: Mapping[str, Any],
        *,
        terminal_state: RunTerminalState | None = None,
        error_code: str | None = None,
        checkpoint_id: str | None = None,
    ) -> RunEvent:
        stored = self._get_locked(run_id)
        if stored.terminal_state is not None:
            raise RunAlreadyTerminalError(f"run is terminal: {run_id}")
        event = RunEvent.create(
            run_id,
            len(stored.events) + 1,
            event_type,
            payload,
            terminal_state=terminal_state,
            error_code=error_code,
            checkpoint_id=checkpoint_id,
        )
        stored.events.append(event)
        if event.terminal_state is not None:
            stored.terminal_state = event.terminal_state
        return event

    def _get_locked(self, run_id: UUID) -> _StoredRun:
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise RunNotFoundError(f"run not found: {run_id}") from error
