"""OpenAI Responses API implementation of the streaming model contract."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

from .streaming_model import ModelDelta, ModelRequest, ModelUsage, StreamingModel, raise_if_cancelled


class OpenAIStreamingModel(StreamingModel):
    """Stream text and usage events from the OpenAI Responses API."""

    def __init__(self, client: AsyncOpenAI, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    async def stream(
        self,
        request: ModelRequest,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[ModelDelta]:
        raise_if_cancelled(cancel_event)
        response_stream = await self._client.responses.create(
            model=request.model or self._default_model,
            input=request.prompt,
            stream=True,
        )
        async with response_stream:
            async for event in response_stream:
                raise_if_cancelled(cancel_event)
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield ModelDelta(text=delta)
                elif event_type == "response.completed":
                    usage = getattr(getattr(event, "response", None), "usage", None)
                    if usage is not None:
                        yield ModelDelta(
                            usage=ModelUsage(
                                input_tokens=getattr(usage, "input_tokens", None),
                                output_tokens=getattr(usage, "output_tokens", None),
                                total_tokens=getattr(usage, "total_tokens", None),
                            )
                        )


def create_openai_model() -> OpenAIStreamingModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    base_url = os.getenv("OPENAI_BASE_URL")
    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    return OpenAIStreamingModel(
        AsyncOpenAI(**client_options),
        os.getenv("OPENAI_MODEL", "gpt-5.6-sol"),
    )