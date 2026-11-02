import asyncio
import os
import unittest
from collections.abc import AsyncIterator
from unittest.mock import patch
from uuid import uuid4

from src.streaming_agent_gateway.model_adapter import (
    DemoStreamingModel,
    ProviderStreamingModel,
    create_configured_model,
)
from src.streaming_agent_gateway.streaming_model import (
    ModelCancelled,
    ModelDelta,
    ModelRequest,
    ModelStreamError,
)


class FakeProviderStream:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for chunk in self.chunks:
            yield chunk
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        self.closed = True


class FakeProviderResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.metadata = {"source": "response"}


class ModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_configured_model_uses_demo_without_provider_credentials(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            model = create_configured_model()

        self.assertIsInstance(model, DemoStreamingModel)

    def test_configured_model_delegates_provider_creation_with_credentials(self) -> None:
        configured_model = object()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(
                "src.streaming_agent_gateway.openai_model.create_openai_model",
                return_value=configured_model,
            ) as create_model,
        ):
            model = create_configured_model()

        self.assertIs(model, configured_model)
        create_model.assert_called_once_with()

    async def test_normalizes_provider_chunk_shapes(self) -> None:
        provider_stream = FakeProviderStream(
            ["first", {"text": " second", "metadata": {"source": "fake"}}, ModelDelta(" final")]
        )
        model = ProviderStreamingModel(lambda request: provider_stream)

        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual([delta.text for delta in deltas], ["first", " second", " final"])
        self.assertEqual(deltas[1].metadata["source"], "fake")
        self.assertTrue(provider_stream.closed)

    async def test_normalizes_complete_provider_response(self) -> None:
        async def provider(request):
            return FakeProviderResponse("complete response")

        model = ProviderStreamingModel(provider)

        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual([delta.text for delta in deltas], ["complete response"])
        self.assertEqual(deltas[0].metadata["source"], "response")

    async def test_normalizes_immediate_mapping_response(self) -> None:
        model = ProviderStreamingModel(
            lambda request: {
                "output_text": "complete response",
                "metadata": {"source": "mapping"},
            }
        )

        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual(deltas, [ModelDelta("complete response", {"source": "mapping"})])

    async def test_provider_failure_becomes_model_stream_error(self) -> None:
        async def failing_provider(request):
            yield "before failure"
            raise TimeoutError("provider timed out")

        model = ProviderStreamingModel(failing_provider)
        stream = model.stream(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual((await anext(stream)).text, "before failure")
        with self.assertRaises(ModelStreamError) as context:
            await anext(stream)
        self.assertEqual(context.exception.code, "timeout")

    async def test_cancellation_closes_provider_stream(self) -> None:
        provider_stream = FakeProviderStream(["first", "second"])
        model = ProviderStreamingModel(lambda request: provider_stream)
        cancel_event = asyncio.Event()
        stream = model.stream(ModelRequest(uuid4(), "hello"), cancel_event)

        self.assertEqual((await anext(stream)).text, "first")
        cancel_event.set()
        with self.assertRaises(ModelCancelled):
            await anext(stream)
        await stream.aclose()

        self.assertTrue(provider_stream.closed)

    async def test_invalid_provider_chunk_has_stable_code(self) -> None:
        provider_stream = FakeProviderStream([{"not_text": True}])
        model = ProviderStreamingModel(lambda request: provider_stream)

        with self.assertRaises(ModelStreamError) as context:
            await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual(context.exception.code, "invalid_provider_chunk")

    async def test_normalizes_usage_only_final_chunk(self) -> None:
        provider_stream = FakeProviderStream([
            "answer",
            {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}},
        ])
        model = ProviderStreamingModel(lambda request: provider_stream)

        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual(deltas[0].text, "answer")
        self.assertEqual(deltas[1].text, "")
        self.assertEqual(deltas[1].usage.total_tokens, 8)


if __name__ == "__main__":
    unittest.main()
