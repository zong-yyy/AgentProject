"""Canonical events emitted during an agent run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID, uuid4


class RunEventType(StrEnum):
    """Events in the append-only run log."""

    STARTED = "run.started"
    DELTA = "run.delta"
    CHECKPOINT = "run.checkpoint"
    COMPLETED = "run.completed"
    CANCELLED = "run.cancelled"
    FAILED = "run.failed"
    RETRY = "run.retry"


class RunTerminalState(StrEnum):
    """The only terminal outcomes a run may have."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL_EVENT_TO_STATE = {
    RunEventType.COMPLETED: RunTerminalState.COMPLETED,
    RunEventType.CANCELLED: RunTerminalState.CANCELLED,
    RunEventType.FAILED: RunTerminalState.FAILED,
}


@dataclass(frozen=True, slots=True)
class RunEvent:
    """An immutable, replayable record in a run's event stream.

    ``sequence`` is scoped to ``run_id`` and starts at one. Consumers can use
    it to detect gaps after reconnecting. ``payload`` is intentionally generic
    so the event contract does not depend on a model SDK.
    """

    event_id: UUID
    run_id: UUID
    sequence: int
    type: RunEventType
    timestamp: datetime
    payload: Mapping[str, Any]
    terminal_state: RunTerminalState | None = None
    error_code: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("sequence must be greater than zero")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.type in _TERMINAL_EVENT_TO_STATE:
            expected_state = _TERMINAL_EVENT_TO_STATE[self.type]
            if self.terminal_state != expected_state:
                raise ValueError(
                    f"{self.type.value} requires terminal_state={expected_state.value}"
                )
        elif self.terminal_state is not None:
            raise ValueError("only terminal events may set terminal_state")
        if self.type is RunEventType.FAILED and not self.error_code:
            raise ValueError("run.failed requires error_code")
        if self.type is not RunEventType.FAILED and self.error_code is not None:
            raise ValueError("error_code is only valid for run.failed")

    @property
    def is_terminal(self) -> bool:
        return self.terminal_state is not None

    @classmethod
    def create(
        cls,
        run_id: UUID,
        sequence: int,
        event_type: RunEventType,
        payload: Mapping[str, Any] | None = None,
        *,
        terminal_state: RunTerminalState | None = None,
        error_code: str | None = None,
        checkpoint_id: str | None = None,
    ) -> "RunEvent":
        """Create an event with a UTC timestamp and a fresh event ID."""

        return cls(
            event_id=uuid4(),
            run_id=run_id,
            sequence=sequence,
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            payload={} if payload is None else dict(payload),
            terminal_state=terminal_state,
            error_code=error_code,
            checkpoint_id=checkpoint_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for SSE or persistence."""

        return {
            "event_id": str(self.event_id),
            "run_id": str(self.run_id),
            "sequence": self.sequence,
            "type": self.type.value,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "payload": dict(self.payload),
            "terminal_state": (
                self.terminal_state.value if self.terminal_state is not None else None
            ),
            "error_code": self.error_code,
            "checkpoint_id": self.checkpoint_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunEvent":
        """Rehydrate an event without coupling persistence to this class's internals."""

        terminal_state = value.get("terminal_state")
        return cls(
            event_id=UUID(str(value["event_id"])),
            run_id=UUID(str(value["run_id"])),
            sequence=int(value["sequence"]),
            type=RunEventType(value["type"]),
            timestamp=datetime.fromisoformat(str(value["timestamp"])),
            payload=dict(value.get("payload", {})),
            terminal_state=(
                RunTerminalState(terminal_state) if terminal_state is not None else None
            ),
            error_code=value.get("error_code"),
            checkpoint_id=value.get("checkpoint_id"),
        )


def validate_event_sequence(events: list[RunEvent]) -> None:
    """Validate the append-only ordering and single-terminal-state invariant."""

    if not events:
        return
    run_id = events[0].run_id
    terminal_count = 0
    for expected_sequence, event in enumerate(events, start=1):
        if event.run_id != run_id:
            raise ValueError("all events must belong to the same run")
        if event.sequence != expected_sequence:
            raise ValueError("event sequence must be contiguous and start at one")
        if event.is_terminal:
            terminal_count += 1
    if terminal_count > 1:
        raise ValueError("a run may have only one terminal event")
    if terminal_count == 1 and not events[-1].is_terminal:
        raise ValueError("no events may follow a terminal event")
