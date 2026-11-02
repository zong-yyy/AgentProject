"""Provider adapter that keeps model SDK details outside the gateway contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable, Mapping
import inspect
import os
from typing import Any

from .streaming_model import (
    ModelCancelled,
    ModelDelta,
    ModelRequest,
    ModelStreamError,
    ModelUsage,
    StreamingModel,
    raise_if_cancelled,
)
from .failure_policy import RetryPolicy, classify_provider_error


ProviderFactory = Callable[[ModelRequest], Any]
ProviderStreamFactory = ProviderFactory


class DemoStreamingModel(StreamingModel):
    """Small local model used when no provider credentials are configured."""

    async def stream(self, request: ModelRequest, cancel_event: asyncio.Event):
        for text in ("Echo: ", request.prompt):
            if cancel_event.is_set():
                return
            yield ModelDelta(text=text)
            await asyncio.sleep(0)


def create_configured_model() -> StreamingModel:
    """Create the configured provider without exposing credentials upstream."""

    if not os.getenv("OPENAI_API_KEY"):
        return DemoStreamingModel()

    from .openai_model import create_openai_model

    return create_openai_model()


class ProviderStreamingModel(StreamingModel):
    """Adapt an arbitrary provider stream or response to ``StreamingModel``."""

    def __init__(
        self,
        provider: ProviderFactory,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=1)

    async def _stream(self, request: ModelRequest, cancel_event: asyncio.Event):
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            provider_stream = None
            emitted_delta = False
            try:
                raise_if_cancelled(cancel_event)
                provider_output = self._provider(request)
                if inspect.isawaitable(provider_output):
                    provider_output = await provider_output
                raise_if_cancelled(cancel_event)

                if isinstance(provider_output, AsyncIterable):
                    provider_stream = provider_output
                    async for raw_chunk in provider_stream:
                        raise_if_cancelled(cancel_event)
                        emitted_delta = True
                        yield self._normalize_output(raw_chunk)
                else:
                    emitted_delta = True
                    yield self._normalize_output(provider_output)
                return
            except ModelStreamError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as error:
                info = classify_provider_error(error)
                if (
                    not emitted_delta
                    and info.retryable
                    and attempt < self._retry_policy.max_attempts
                ):
                    await asyncio.sleep(self._retry_policy.delay(attempt, info.retry_after))
                    continue
                raise ModelStreamError(info.code, info.message) from error
            finally:
                if provider_stream is not None:
                    close = getattr(provider_stream, "aclose", None)
                    if close is not None:
                        await close()

    def stream(self, request: ModelRequest, cancel_event: asyncio.Event):
        return self._stream(request, cancel_event)

    @staticmethod
    def _normalize_output(raw_output: Any) -> ModelDelta:
        if isinstance(raw_output, ModelDelta):
            return raw_output
        if isinstance(raw_output, str):
            return ModelDelta(text=raw_output)
        if isinstance(raw_output, Mapping):
            text = raw_output.get("output_text", raw_output.get("text"))
            usage = ProviderStreamingModel._normalize_usage(raw_output.get("usage"))
            metadata = raw_output.get("metadata", {})
            refusal = raw_output.get("refusal")
            finish_reason = raw_output.get("finish_reason")
            if refusal is not None or finish_reason is not None:
                metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
                if refusal is not None:
                    metadata["refusal"] = refusal
                if finish_reason is not None:
                    metadata["finish_reason"] = finish_reason
            if isinstance(text, str):
                return ModelDelta(
                    text=text,
                    metadata=metadata if isinstance(metadata, Mapping) else {},
                    usage=usage,
                )
            if usage is not None:
                return ModelDelta(usage=usage, metadata=metadata if isinstance(metadata, Mapping) else {})
        text = getattr(raw_output, "output_text", None)
        if not isinstance(text, str):
            text = getattr(raw_output, "text", None)
        usage = ProviderStreamingModel._normalize_usage(getattr(raw_output, "usage", None))
        metadata = getattr(raw_output, "metadata", {})
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        for field in ("refusal", "finish_reason"):
            value = getattr(raw_output, field, None)
            if value is not None:
                metadata[field] = value
        if isinstance(text, str):
            return ModelDelta(
                text=text,
                metadata=metadata if isinstance(metadata, Mapping) else {},
                usage=usage,
            )
        if usage is not None:
            return ModelDelta(usage=usage)
        raise ModelStreamError(
            "invalid_provider_chunk",
            "provider output must contain a text string",
        )

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> ModelUsage | None:
        if raw_usage is None:
            return None
        if isinstance(raw_usage, ModelUsage):
            return raw_usage
        if isinstance(raw_usage, Mapping):
            get_value = raw_usage.get
        else:
            get_value = lambda key, default=None: getattr(raw_usage, key, default)

        input_tokens = get_value("input_tokens", get_value("prompt_tokens"))
        output_tokens = get_value("output_tokens", get_value("completion_tokens"))
        total_tokens = get_value("total_tokens")
        values = (input_tokens, output_tokens, total_tokens)
        if not any(isinstance(value, int) and value >= 0 for value in values):
            raise ModelStreamError(
                "invalid_provider_usage",
                "provider usage must contain non-negative integer token counts",
            )
        return ModelUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )
