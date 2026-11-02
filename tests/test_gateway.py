import asyncio
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.agent_loop import AgentLoop
from src.streaming_agent_gateway.events import RunEventType, RunTerminalState
from src.streaming_agent_gateway.gateway import RunGateway
from src.streaming_agent_gateway.run_store import InMemoryRunStore
from src.streaming_agent_gateway.streaming_model import ModelDelta, ModelRequest, StreamingModel


class DelayedModel(StreamingModel):
    async def stream(self, request, cancel_event):
        yield ModelDelta("first")
        await asyncio.sleep(0.02)
        if not cancel_event.is_set():
            yield ModelDelta("second")


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_delivers_terminal_event_appended_during_status_check(self) -> None:
        class TerminalDuringGetStore(InMemoryRunStore):
            terminal_appended = False

            def get(self, run_id):
                if not self.terminal_appended:
                    self.terminal_appended = True
                    self.append(
                        run_id,
                        RunEventType.COMPLETED,
                        terminal_state=RunTerminalState.COMPLETED,
                    )
                return super().get(run_id)

        store = TerminalDuringGetStore()
        loop = AgentLoop(DelayedModel(), store)
        gateway = RunGateway(loop, store)
        started = store.create_run()

        events = [event async for event in gateway.subscribe(started.run_id)]

        self.assertEqual(
            [event.type for event in events],
            [RunEventType.STARTED, RunEventType.COMPLETED],
        )

    async def test_subscribe_replays_then_follows_to_terminal(self) -> None:
        store = InMemoryRunStore()
        loop = AgentLoop(DelayedModel(), store)
        gateway = RunGateway(loop, store)
        run_id = uuid4()
        await gateway.submit(ModelRequest(run_id, "hello"))

        events = [event async for event in gateway.subscribe(run_id, poll_interval=0.001)]

        self.assertEqual(
            [event.type for event in events],
            [RunEventType.STARTED, RunEventType.DELTA, RunEventType.DELTA, RunEventType.COMPLETED],
        )
        self.assertEqual(events[-1].terminal_state, RunTerminalState.COMPLETED)

    async def test_reconnect_cursor_does_not_rerun_model(self) -> None:
        store = InMemoryRunStore()
        loop = AgentLoop(DelayedModel(), store)
        gateway = RunGateway(loop, store)
        run_id = uuid4()
        await gateway.submit(ModelRequest(run_id, "hello"))

        first = await anext(gateway.subscribe(run_id, poll_interval=0.001))
        self.assertEqual(first.type, RunEventType.STARTED)
        await asyncio.sleep(0.05)
        replayed = [event async for event in gateway.subscribe(run_id, after_sequence=1)]

        self.assertEqual([event.sequence for event in replayed], [2, 3, 4])
        self.assertEqual(len(store.get(run_id).events), 4)

    async def test_cancel_endpoint_stops_run_and_publishes_terminal_event(self) -> None:
        store = InMemoryRunStore()
        loop = AgentLoop(DelayedModel(), store)
        gateway = RunGateway(loop, store)
        run_id = uuid4()
        await gateway.submit(ModelRequest(run_id, "hello"))

        await asyncio.sleep(0.001)
        await gateway.cancel(run_id)
        events = [event async for event in gateway.subscribe(run_id)]

        self.assertEqual(events[-1].type, RunEventType.CANCELLED)
        self.assertEqual(gateway.active_run_count(), 0)


if __name__ == "__main__":
    unittest.main()
