import asyncio
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.agent_loop import AgentLoop, RunNotActiveError
from src.streaming_agent_gateway.events import RunEventType, RunTerminalState
from src.streaming_agent_gateway.run_store import InMemoryRunStore
from src.streaming_agent_gateway.streaming_model import (
    ModelCancelled,
    ModelDelta,
    ModelRequest,
    ModelStreamError,
    StreamingModel,
)


class FakeModel(StreamingModel):
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.first_delta = asyncio.Event()
        self.cancel_seen = asyncio.Event()

    async def _stream(self, request, cancel_event):
        yield ModelDelta("first")
        self.first_delta.set()
        while not cancel_event.is_set():
            await asyncio.sleep(0)
        self.cancel_seen.set()
        raise ModelCancelled()

    def stream(self, request, cancel_event):
        if self.failure is not None:
            async def fail():
                raise self.failure
                yield ModelDelta()
            return fail()
        return self._stream(request, cancel_event)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_deltas_are_persisted_and_completed_once(self) -> None:
        store = InMemoryRunStore()

        class FiniteModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta("one")
                yield ModelDelta("two")

        request = ModelRequest(uuid4(), "hello")
        snapshot = await AgentLoop(FiniteModel(), store).run(request)

        self.assertEqual(snapshot.terminal_state, RunTerminalState.COMPLETED)
        self.assertEqual([event.type for event in snapshot.events], [
            RunEventType.STARTED,
            RunEventType.DELTA,
            RunEventType.DELTA,
            RunEventType.COMPLETED,
        ])
        self.assertEqual(store.active_run_count(), 0)

    async def test_cancel_creates_cancelled_terminal_event(self) -> None:
        model = FakeModel()
        store = InMemoryRunStore()
        loop = AgentLoop(model, store)
        request = ModelRequest(uuid4(), "hello")
        task = await loop.start(request)
        await model.first_delta.wait()

        await loop.cancel(request.run_id)
        snapshot = await task

        self.assertTrue(model.cancel_seen.is_set())
        self.assertEqual(snapshot.terminal_state, RunTerminalState.CANCELLED)
        self.assertEqual(sum(event.is_terminal for event in snapshot.events), 1)
        self.assertEqual(loop.active_run_count(), 0)

    async def test_model_error_becomes_failed_terminal_event(self) -> None:
        store = InMemoryRunStore()
        loop = AgentLoop(FakeModel(ModelStreamError("quota", "limit reached")), store)
        request = ModelRequest(uuid4(), "hello")

        snapshot = await loop.run(request)

        terminal = snapshot.events[-1]
        self.assertEqual(terminal.type, RunEventType.FAILED)
        self.assertEqual(terminal.error_code, "quota")
        self.assertEqual(snapshot.terminal_state, RunTerminalState.FAILED)

    async def test_cancel_unknown_run_is_rejected(self) -> None:
        with self.assertRaises(RunNotActiveError):
            await AgentLoop(FakeModel(), InMemoryRunStore()).cancel(uuid4())


if __name__ == "__main__":
    unittest.main()
