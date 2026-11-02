import asyncio
import unittest
from uuid import uuid4

from src.streaming_agent_gateway.agent_loop import AgentLoop
from src.streaming_agent_gateway.events import RunEventType, RunTerminalState
from src.streaming_agent_gateway.failure_policy import RetryPolicy, classify_provider_error, OutputValidationError
from src.streaming_agent_gateway.model_adapter import ProviderStreamingModel
from src.streaming_agent_gateway.run_store import InMemoryRunStore
from src.streaming_agent_gateway.streaming_model import ModelRequest, ModelDelta, StreamingModel


class FailurePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_failure_with_bounded_attempts(self) -> None:
        attempts = 0

        async def provider(request):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TimeoutError("temporary timeout")
            yield "ok"

        model = ProviderStreamingModel(
            provider,
            RetryPolicy(max_attempts=3, base_delay=0, jitter=0),
        )
        deltas = await model.collect(ModelRequest(uuid4(), "hello"), asyncio.Event())

        self.assertEqual(attempts, 3)
        self.assertEqual([delta.text for delta in deltas], ["ok"])

    async def test_does_not_retry_after_partial_output(self) -> None:
        attempts = 0

        async def provider(request):
            nonlocal attempts
            attempts += 1
            yield "partial"
            raise ConnectionError("connection lost")

        model = ProviderStreamingModel(
            provider,
            RetryPolicy(max_attempts=3, base_delay=0, jitter=0),
        )
        stream = model.stream(ModelRequest(uuid4(), "hello"), asyncio.Event())
        self.assertEqual((await anext(stream)).text, "partial")
        with self.assertRaises(Exception) as context:
            await anext(stream)
        self.assertEqual(getattr(context.exception, "code", None), "temporary_network_error")
        self.assertEqual(attempts, 1)

    async def test_refusal_becomes_failed_terminal_state(self) -> None:
        class RefusingModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta(metadata={"refusal": "unsafe request"})

        store = InMemoryRunStore()
        snapshot = await AgentLoop(RefusingModel(), store).run(ModelRequest(uuid4(), "hello"))

        self.assertEqual(snapshot.terminal_state, RunTerminalState.FAILED)
        self.assertEqual(snapshot.events[-1].error_code, "content_refused")

    async def test_schema_validation_failure_is_structured(self) -> None:
        class JsonModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta('{"broken": true}')

        def validate(value: str) -> None:
            raise OutputValidationError("required field is missing")

        snapshot = await AgentLoop(
            JsonModel(), InMemoryRunStore(), output_validator=validate
        ).run(ModelRequest(uuid4(), "hello"))

        self.assertEqual(snapshot.terminal_state, RunTerminalState.FAILED)
        self.assertEqual(snapshot.events[-1].error_code, "invalid_schema")

    async def test_schema_failure_gets_one_correction_retry(self) -> None:
        class InvalidModel(StreamingModel):
            async def stream(self, request, cancel_event):
                yield ModelDelta("invalid" if "correction_attempt" not in request.metadata else "valid")

        def validate(value: str) -> None:
            if value != "valid":
                raise OutputValidationError("invalid structured output")

        snapshot = await AgentLoop(
            InvalidModel(),
            InMemoryRunStore(),
            output_validator=validate,
            correction_model=InvalidModel(),
        ).run(ModelRequest(uuid4(), "hello"))

        self.assertEqual(snapshot.terminal_state, RunTerminalState.COMPLETED)
        self.assertEqual(
            [event.type for event in snapshot.events].count(RunEventType.RETRY), 1
        )


class ErrorClassificationTests(unittest.TestCase):
    def test_retry_after_and_rate_limit_are_classified(self) -> None:
        error = type("RateLimitError", (), {"status_code": 429, "headers": {"Retry-After": "2.5"}})()

        info = classify_provider_error(error)

        self.assertEqual(info.code, "rate_limited")
        self.assertEqual(info.retry_after, 2.5)


if __name__ == "__main__":
    unittest.main()
