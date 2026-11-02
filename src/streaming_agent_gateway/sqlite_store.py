"""SQLite-backed Run Store for restart recovery and audit durability."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import sqlite3
import threading
from typing import Any, Iterator, Mapping
from uuid import UUID, uuid4

from .events import RunEvent, RunEventType, RunTerminalState
from .run_store import (
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunSequenceError,
    RunSnapshot,
    RunStoreClosedError,
    RunStoreError,
)
from .streaming_model import ModelUsage


@dataclass(frozen=True, slots=True)
class Checkpoint:
    run_id: UUID
    checkpoint_id: str
    sequence: int
    state: Mapping[str, Any]


class SQLiteRunStore:
    """A serialized SQLite Run Store with event and checkpoint recovery."""

    def __init__(self, path: str = ":memory:", busy_timeout_ms: int = 5000) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")
        self._connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=busy_timeout_ms / 1000,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def create_run(self, run_id: UUID | None = None, **payload: Any) -> RunEvent:
        run_id = uuid4() if run_id is None else run_id
        payload_json = self._encode_json(payload, "run payload")
        with self._write_transaction():
            try:
                self._connection.execute(
                    "INSERT INTO runs(run_id) VALUES (?)", (str(run_id),)
                )
            except sqlite3.IntegrityError as error:
                raise RunStoreError(f"run already exists: {run_id}") from error
            return self._append_locked(
                run_id,
                RunEventType.STARTED,
                payload,
                payload_json=payload_json,
            )

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
        normalized_payload = {} if payload is None else dict(payload)
        payload_json = self._encode_json(normalized_payload, "event payload")
        with self._write_transaction():
            return self._append_locked(
                run_id,
                event_type,
                normalized_payload,
                payload_json=payload_json,
                terminal_state=terminal_state,
                error_code=error_code,
                checkpoint_id=checkpoint_id,
            )

    def append_event(self, event: RunEvent) -> RunEvent:
        """Append a prebuilt event after validating its next sequence."""

        payload_json = self._encode_json(event.payload, "event payload")
        with self._write_transaction():
            terminal_state, current_sequence = self._run_state_locked(event.run_id)
            if terminal_state is not None:
                raise RunAlreadyTerminalError(f"run is terminal: {event.run_id}")
            expected_sequence = current_sequence + 1
            if event.sequence != expected_sequence:
                raise RunSequenceError(
                    f"expected sequence {expected_sequence}, got {event.sequence}"
                )
            self._insert_event_locked(event, payload_json)
            return event

    def replay(self, run_id: UUID, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            rows = self._connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                (str(run_id), after_sequence),
            ).fetchall()
            return tuple(self._row_to_event(row) for row in rows)

    def get(self, run_id: UUID) -> RunSnapshot:
        with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            rows = self._connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY sequence", (str(run_id),)
            ).fetchall()
            events = tuple(self._row_to_event(row) for row in rows)
            terminal = events[-1].terminal_state if events and events[-1].is_terminal else None
            return RunSnapshot(run_id, events, terminal)

    def list_runs(self) -> tuple[RunSnapshot, ...]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT run_id FROM runs ORDER BY rowid DESC"
            ).fetchall()
            return tuple(self.get(UUID(row["run_id"])) for row in rows)

    def active_run_count(self) -> int:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE terminal_state IS NULL"
            ).fetchone()
            return int(row["count"])

    def save_checkpoint(
        self,
        run_id: UUID,
        sequence: int,
        state: Mapping[str, Any],
        checkpoint_id: str | None = None,
    ) -> Checkpoint:
        state_copy = dict(state)
        state_json = self._encode_json(state_copy, "checkpoint state")
        checkpoint = Checkpoint(
            run_id=run_id,
            checkpoint_id=checkpoint_id or str(uuid4()),
            sequence=sequence,
            state=state_copy,
        )
        with self._write_transaction():
            self._require_run(run_id)
            event_exists = self._connection.execute(
                "SELECT 1 FROM run_events WHERE run_id = ? AND sequence = ?",
                (str(run_id), sequence),
            ).fetchone()
            if event_exists is None:
                raise RunSequenceError(
                    f"checkpoint sequence does not exist: {sequence}"
                )
            self._connection.execute(
                "INSERT INTO checkpoints(checkpoint_id, run_id, sequence, state_json) VALUES (?, ?, ?, ?)",
                (checkpoint.checkpoint_id, str(run_id), sequence, state_json),
            )
        return checkpoint

    def latest_checkpoint(self, run_id: UUID) -> Checkpoint | None:
        with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            row = self._connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (str(run_id),),
            ).fetchone()
            if row is None:
                return None
            return Checkpoint(
                run_id=run_id,
                checkpoint_id=row["checkpoint_id"],
                sequence=row["sequence"],
                state=json.loads(row["state_json"]),
            )

    def record_ttft(self, run_id: UUID, ttft_ms: float) -> None:
        with self._write_transaction():
            self._require_run(run_id)
            self._connection.execute(
                "INSERT INTO run_metrics(run_id, ttft_ms) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET ttft_ms = excluded.ttft_ms",
                (str(run_id), ttft_ms),
            )

    def ttft_ms(self, run_id: UUID) -> float | None:
        with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            row = self._connection.execute(
                "SELECT ttft_ms FROM run_metrics WHERE run_id = ?", (str(run_id),)
            ).fetchone()
            return None if row is None else float(row["ttft_ms"])

    def record_usage(self, run_id: UUID, usage: ModelUsage) -> None:
        values = (usage.input_tokens, usage.output_tokens, usage.total_tokens)
        if any(value is not None and (not isinstance(value, int) or value < 0) for value in values):
            raise RunStoreError("token usage must contain non-negative integer values")
        with self._write_transaction():
            self._require_run(run_id)
            self._connection.execute(
                "INSERT INTO run_metrics(run_id, input_tokens, output_tokens, total_tokens) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                "input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens, "
                "total_tokens = excluded.total_tokens",
                (str(run_id), *values),
            )

    def usage(self, run_id: UUID) -> ModelUsage | None:
        with self._lock:
            self._ensure_open()
            self._require_run(run_id)
            row = self._connection.execute(
                "SELECT input_tokens, output_tokens, total_tokens FROM run_metrics WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            if row is None or all(row[key] is None for key in ("input_tokens", "output_tokens", "total_tokens")):
                return None
            return ModelUsage(row["input_tokens"], row["output_tokens"], row["total_tokens"])

    def _append_locked(
        self,
        run_id: UUID,
        event_type: RunEventType,
        payload: Mapping[str, Any],
        *,
        payload_json: str,
        **kwargs: Any,
    ) -> RunEvent:
        terminal_state, current_sequence = self._run_state_locked(run_id)
        if terminal_state is not None:
            raise RunAlreadyTerminalError(f"run is terminal: {run_id}")
        event = RunEvent.create(
            run_id,
            current_sequence + 1,
            event_type,
            payload,
            **kwargs,
        )
        self._insert_event_locked(event, payload_json)
        return event

    def _insert_event_locked(self, event: RunEvent, payload_json: str) -> None:
        self._connection.execute(
            "INSERT INTO run_events(event_id, run_id, sequence, type, timestamp, payload_json, terminal_state, error_code, checkpoint_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(event.event_id), str(event.run_id), event.sequence, event.type.value,
                event.timestamp.isoformat(), payload_json,
                event.terminal_state.value if event.terminal_state else None,
                event.error_code, event.checkpoint_id,
            ),
        )
        if event.terminal_state is not None:
            self._connection.execute(
                "UPDATE runs SET terminal_state = ? WHERE run_id = ?",
                (event.terminal_state.value, str(event.run_id)),
            )

    def _run_state_locked(self, run_id: UUID) -> tuple[str | None, int]:
        row = self._connection.execute(
            "SELECT terminal_state, "
            "(SELECT COALESCE(MAX(sequence), 0) FROM run_events WHERE run_id = ?) AS sequence "
            "FROM runs WHERE run_id = ?",
            (str(run_id), str(run_id)),
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return row["terminal_state"], int(row["sequence"])

    def _require_run(self, run_id: UUID) -> None:
        row = self._connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")

    def _row_to_event(self, row: sqlite3.Row) -> RunEvent:
        return RunEvent.from_dict({
            "event_id": row["event_id"], "run_id": row["run_id"], "sequence": row["sequence"],
            "type": row["type"], "timestamp": row["timestamp"], "payload": json.loads(row["payload_json"]),
            "terminal_state": row["terminal_state"], "error_code": row["error_code"], "checkpoint_id": row["checkpoint_id"],
        })

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RunStoreClosedError("run store is closed")

    @staticmethod
    def _encode_json(value: Mapping[str, Any], description: str) -> str:
        try:
            return json.dumps(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RunStoreError(f"{description} is not JSON-serializable") from error

    def _initialize(self) -> None:
        with self._write_transaction():
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    terminal_state TEXT
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    terminal_state TEXT,
                    error_code TEXT,
                    checkpoint_id TEXT,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    sequence INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                    ttft_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER
                );
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(run_metrics)").fetchall()
            }
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                if name not in columns:
                    self._connection.execute(f"ALTER TABLE run_metrics ADD COLUMN {name} INTEGER")
