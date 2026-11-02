import threading
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.events import RunEventType, RunTerminalState
from src.streaming_agent_gateway.run_store import (
    InMemoryRunStore,
    RunAlreadyTerminalError,
    RunSequenceError,
)


class RunStoreTests(unittest.TestCase):
    def test_append_assigns_sequences_and_replay_uses_cursor(self) -> None:
        store = InMemoryRunStore()
        started = store.create_run(payload="ignored")
        store.append(started.run_id, RunEventType.DELTA, {"text": "one"})
        store.append(started.run_id, RunEventType.DELTA, {"text": "two"})

        replayed = store.replay(started.run_id, after_sequence=1)

        self.assertEqual([event.sequence for event in replayed], [2, 3])
        self.assertEqual(store.active_run_count(), 1)

    def test_terminal_run_cannot_be_appended_to_twice(self) -> None:
        store = InMemoryRunStore()
        started = store.create_run()
        store.append(
            started.run_id,
            RunEventType.CANCELLED,
            terminal_state=RunTerminalState.CANCELLED,
        )

        self.assertEqual(store.get(started.run_id).terminal_state, RunTerminalState.CANCELLED)
        self.assertEqual(store.active_run_count(), 0)
        with self.assertRaises(RunAlreadyTerminalError):
            store.append(started.run_id, RunEventType.DELTA, {"text": "late"})

    def test_list_runs_returns_newest_first_with_terminal_state(self) -> None:
        store = InMemoryRunStore()
        first = store.create_run()
        second = store.create_run()
        store.append(
            first.run_id,
            RunEventType.COMPLETED,
            terminal_state=RunTerminalState.COMPLETED,
        )

        snapshots = store.list_runs()

        self.assertEqual([snapshot.run_id for snapshot in snapshots], [second.run_id, first.run_id])
        self.assertIsNone(snapshots[0].terminal_state)
        self.assertEqual(snapshots[1].terminal_state, RunTerminalState.COMPLETED)

    def test_append_event_rejects_sequence_gap(self) -> None:
        store = InMemoryRunStore()
        started = store.create_run()
        invalid = store.replay(started.run_id)[0].__class__.create(
            started.run_id, 3, RunEventType.DELTA, {"text": "gap"}
        )

        with self.assertRaises(RunSequenceError):
            store.append_event(invalid)

    def test_concurrent_appends_remain_contiguous(self) -> None:
        store = InMemoryRunStore()
        started = store.create_run()
        errors: list[Exception] = []

        def append_delta() -> None:
            try:
                store.append(started.run_id, RunEventType.DELTA, {"text": "x"})
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=append_delta) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(
            [event.sequence for event in store.replay(started.run_id)],
            list(range(1, 22)),
        )


if __name__ == "__main__":
    unittest.main()
