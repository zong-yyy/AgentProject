import asyncio
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.agent_loop import AgentLoop
from src.streaming_agent_gateway.events import RunEventType, RunTerminalState
from src.streaming_agent_gateway.gateway import RunGateway
from src.streaming_agent_gateway.model_adapter import ProviderStreamingModel
from src.streaming_agent_gateway.run_store import InMemoryRunStore, RunNotFoundError
from src.streaming_agent_gateway.sse import SSEDecoder, SSEMessage, decode_run_event
from src.streaming_agent_gateway.streaming_model import ModelRequest


class FailureProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_failure_becomes_a_single_auditable_failed_run(self) -> None:
        async def provider(request):
            raise ConnectionError("network disconnected")
            yield "unreachable"

        store = InMemoryRunStore()
        model = ProviderStreamingModel(provider)
        snapshot = await AgentLoop(model, store).run(ModelRequest(uuid4(), "hello"))

        terminal_events = [event for event in snapshot.events if event.is_terminal]
        self.assertEqual(len(terminal_events), 1)
        self.assertEqual(terminal_events[0].type, RunEventType.FAILED)
        self.assertEqual(terminal_events[0].error_code, "temporary_network_error")
        self.assertEqual(snapshot.terminal_state, RunTerminalState.FAILED)
        self.assertEqual(store.active_run_count(), 0)

    async def test_task_cancellation_is_recorded_before_task_finishes(self) -> None:
        class StuckModel:
            def stream(self, request, cancel_event):
                async def never_finishes():
                    while True:
                        await asyncio.sleep(0.001)
                        yield "never"
                return never_finishes()

        store = InMemoryRunStore()
        run_id = uuid4()
        task = await AgentLoop(StuckModel(), store).start(ModelRequest(run_id, "hello"))
        await asyncio.sleep(0.002)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        snapshot = store.get(run_id)
        self.assertEqual(snapshot.terminal_state, RunTerminalState.CANCELLED)
        self.assertEqual(snapshot.events[-1].type, RunEventType.CANCELLED)
        self.assertEqual(store.active_run_count(), 0)

    async def test_invalid_sse_json_is_rejected_at_decode_boundary(self) -> None:
        message = SSEMessage(event="run.delta", data="not-json", event_id="2")

        with self.assertRaises(ValueError):
            decode_run_event(message)

    def test_truncated_sse_frame_is_rejected(self) -> None:
        decoder = SSEDecoder()
        decoder.feed("id: 1\nevent: run.delta\ndata: {\"broken\":")

        with self.assertRaises(ValueError):
            decoder.finish()

    async def test_disconnect_does_not_cancel_underlying_run(self) -> None:
        class SlowModel:
            def __init__(self):
                self.cancel_seen = False

            async def stream(self, request, cancel_event):
                yield type("Delta", (), {"text": "first", "metadata": {}})()
                await asyncio.sleep(0.02)
                if cancel_event.is_set():
                    self.cancel_seen = True
                yield type("Delta", (), {"text": "second", "metadata": {}})()

        model = SlowModel()
        store = InMemoryRunStore()
        loop = AgentLoop(model, store)
        gateway = RunGateway(loop, store)
        run_id = uuid4()
        await gateway.submit(ModelRequest(run_id, "hello"))

        subscription = gateway.subscribe(run_id, poll_interval=0.001)
        await anext(subscription)
        await subscription.aclose()
        await asyncio.sleep(0.05)

        self.assertFalse(model.cancel_seen)
        self.assertEqual(store.get(run_id).terminal_state, RunTerminalState.COMPLETED)

    def test_missing_run_is_a_distinct_store_error(self) -> None:
        with self.assertRaises(RunNotFoundError):
            InMemoryRunStore().get(uuid4())


if __name__ == "__main__":
    unittest.main()
