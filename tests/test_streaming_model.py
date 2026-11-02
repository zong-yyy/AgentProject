import asyncio
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.streaming_model import (
    ModelCancelled,
    ModelDelta,
    ModelRequest,
    StreamingModel,
    raise_if_cancelled,
)


class FakeStreamingModel(StreamingModel):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def _stream(
        self,
        request: ModelRequest,
        cancel_event: asyncio.Event,
    ):
        self.started.set()
        for text in ("first", " second", " final"):
            raise_if_cancelled(cancel_event)
            yield ModelDelta(text=text)
            await asyncio.sleep(0)
        self.stopped.set()

    def stream(self, request, cancel_event):
        return self._stream(request, cancel_event)


class StreamingModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_delta_is_available_without_sdk_types(self) -> None:
        model = FakeStreamingModel()
        request = ModelRequest(uuid4(), "hello", metadata={"source": "test"})
        cancel_event = asyncio.Event()

        stream = model.stream(request, cancel_event)
        first_delta = await anext(stream)

        self.assertEqual(first_delta.text, "first")
        self.assertTrue(model.started.is_set())
        await stream.aclose()

    async def test_adapter_can_observe_cancellation(self) -> None:
        model = FakeStreamingModel()
        cancel_event = asyncio.Event()
        stream = model.stream(ModelRequest(uuid4(), "hello"), cancel_event)

        self.assertEqual((await anext(stream)).text, "first")
        cancel_event.set()

        with self.assertRaises(ModelCancelled):
            await anext(stream)
        await stream.aclose()

    async def test_collect_preserves_all_deltas(self) -> None:
        model = FakeStreamingModel()
        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual([delta.text for delta in deltas], ["first", " second", " final"])
        self.assertTrue(model.stopped.is_set())


if __name__ == "__main__":
    unittest.main()
