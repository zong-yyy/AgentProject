import asyncio
import os
import tempfile
import threading
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.agent_loop import AgentLoop
from src.streaming_agent_gateway.events import RunEvent, RunEventType, RunTerminalState
from src.streaming_agent_gateway.run_store import (
    RunSequenceError,
    RunStoreClosedError,
    RunStoreError,
)
from src.streaming_agent_gateway.sqlite_store import SQLiteRunStore
from src.streaming_agent_gateway.streaming_model import ModelDelta, ModelRequest, StreamingModel
from src.streaming_agent_gateway.streaming_model import ModelUsage


class SQLiteStoreTests(unittest.TestCase):
    def test_events_and_terminal_state_survive_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.sqlite3")
            run_id = uuid4()
            first = SQLiteRunStore(path)
            first.create_run(run_id, prompt="hello")
            first.append(run_id, RunEventType.DELTA, {"text": "hi"})
            first.append(
                run_id,
                RunEventType.COMPLETED,
                terminal_state=RunTerminalState.COMPLETED,
            )
            first.close()

            second = SQLiteRunStore(path)
            snapshot = second.get(run_id)
            self.assertEqual([event.sequence for event in snapshot.events], [1, 2, 3])
            self.assertEqual(snapshot.terminal_state, RunTerminalState.COMPLETED)
            self.assertEqual(second.active_run_count(), 0)
            second.close()

    def test_list_runs_returns_newest_first(self) -> None:
        store = SQLiteRunStore()
        first = store.create_run().run_id
        second = store.create_run().run_id

        snapshots = store.list_runs()

        self.assertEqual([snapshot.run_id for snapshot in snapshots], [second, first])
        store.close()

    def test_checkpoint_and_ttft_survive_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.sqlite3")
            run_id = uuid4()
            first = SQLiteRunStore(path)
            first.create_run(run_id)
            first.append(run_id, RunEventType.DELTA, {"text": "hello"})
            checkpoint = first.save_checkpoint(run_id, 2, {"text": "hello", "delta_count": 1})
            first.record_ttft(run_id, 12.5)
            first.close()

            second = SQLiteRunStore(path)
            restored = second.latest_checkpoint(run_id)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.checkpoint_id, checkpoint.checkpoint_id)
            self.assertEqual(restored.state["text"], "hello")
            self.assertEqual(second.ttft_ms(run_id), 12.5)
            second.close()

    def test_token_usage_survives_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.sqlite3")
            run_id = uuid4()
            first = SQLiteRunStore(path)
            first.create_run(run_id)
            first.record_usage(run_id, ModelUsage(3, 5, 8))
            first.close()

            second = SQLiteRunStore(path)
            usage = second.usage(run_id)
            self.assertEqual(usage, ModelUsage(3, 5, 8))
            second.close()

    def test_checkpoint_requires_an_existing_event_sequence(self) -> None:
        store = SQLiteRunStore()
        run_id = store.create_run().run_id

        with self.assertRaises(RunSequenceError):
            store.save_checkpoint(run_id, 2, {"text": "future"})

        self.assertIsNone(store.latest_checkpoint(run_id))
        store.close()

    def test_invalid_json_raises_store_error_without_appending(self) -> None:
        store = SQLiteRunStore()
        run_id = store.create_run().run_id

        with self.assertRaisesRegex(RunStoreError, "event payload"):
            store.append(run_id, RunEventType.DELTA, {"value": object()})

        self.assertEqual(len(store.get(run_id).events), 1)
        store.close()

    def test_append_event_validates_sequence_and_persists_event(self) -> None:
        store = SQLiteRunStore()
        run_id = store.create_run().run_id
        invalid = RunEvent.create(run_id, 3, RunEventType.DELTA)

        with self.assertRaises(RunSequenceError):
            store.append_event(invalid)

        event = RunEvent.create(run_id, 2, RunEventType.DELTA, {"text": "ok"})
        store.append_event(event)
        self.assertEqual(store.get(run_id).events[-1], event)
        store.close()

    def test_multiple_connections_allocate_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.sqlite3")
            first = SQLiteRunStore(path)
            run_id = first.create_run().run_id
            second = SQLiteRunStore(path)
            errors: list[Exception] = []

            def append_many(store: SQLiteRunStore) -> None:
                try:
                    for _ in range(10):
                        store.append(run_id, RunEventType.DELTA, {"text": "x"})
                except Exception as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=append_many, args=(first,)),
                threading.Thread(target=append_many, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(
                [event.sequence for event in first.replay(run_id)],
                list(range(1, 22)),
            )
            first.close()
            second.close()

    def test_operations_after_close_raise_store_closed_error(self) -> None:
        store = SQLiteRunStore()
        run_id = store.create_run().run_id
        store.close()
        store.close()

        with self.assertRaises(RunStoreClosedError):
            store.get(run_id)


class CheckpointAndTTFTTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttft_starts_at_first_non_empty_text_delta(self) -> None:
        class ProviderModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta(metadata={"provider_event": "role"})
                await asyncio.sleep(0.01)
                yield ModelDelta(usage=ModelUsage(1, 1, 2))
                await asyncio.sleep(0.01)
                yield ModelDelta("visible text")

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(os.path.join(directory, "runs.sqlite3"))
            request = ModelRequest(uuid4(), "hello")

            await AgentLoop(ProviderModel(), store).run(request)

            ttft = store.ttft_ms(request.run_id)
            self.assertIsNotNone(ttft)
            assert ttft is not None
            self.assertGreaterEqual(ttft, 15.0)
            self.assertEqual(store.usage(request.run_id), ModelUsage(1, 1, 2))
            store.close()

    async def test_agent_loop_writes_checkpoint_and_ttft(self) -> None:
        class FiniteModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta("one")
                await asyncio.sleep(0)
                yield ModelDelta("two")
                yield ModelDelta(usage=ModelUsage(3, 5, 8))

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(os.path.join(directory, "runs.sqlite3"))
            request = ModelRequest(uuid4(), "hello")
            snapshot = await AgentLoop(
                FiniteModel(), store, checkpoint_interval=1
            ).run(request)

            checkpoint = store.latest_checkpoint(request.run_id)
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(checkpoint.state["text"], "onetwo")
            self.assertEqual(checkpoint.sequence, 3)
            self.assertIsNotNone(store.ttft_ms(request.run_id))
            self.assertEqual(store.usage(request.run_id), ModelUsage(3, 5, 8))
            self.assertEqual(snapshot.terminal_state, RunTerminalState.COMPLETED)
            store.close()


if __name__ == "__main__":
    unittest.main()
