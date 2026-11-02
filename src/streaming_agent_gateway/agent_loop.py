"""Run orchestration from a StreamingModel into the Run Store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable
from uuid import UUID

from .events import RunEventType, RunTerminalState
from .failure_policy import OutputValidationError
from .run_store import CheckpointStore, RunSnapshot, RunStore, TTFTStore, UsageStore
from .streaming_model import (
    ModelCancelled,
    ModelRequest,
    ModelStreamError,
    StreamingModel,
)


class RunNotActiveError(Exception):
    """The requested run is not currently owned by this Agent Loop."""


@dataclass(slots=True)
class _ActiveRun:
    cancel_event: asyncio.Event
    task: asyncio.Task[RunSnapshot] | None = None


class AgentLoop:
    """Execute one model stream and persist its complete event history."""

    def __init__(
        self,
        model: StreamingModel,
        store: RunStore,
        checkpoint_interval: int = 0,
        output_validator: Callable[[str], Any] | None = None,
        correction_model: StreamingModel | None = None,
    ) -> None:
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must not be negative")
        self._model = model
        self._store = store
        self._checkpoint_interval = checkpoint_interval
        self._output_validator = output_validator
        self._correction_model = correction_model
        self._active: dict[UUID, _ActiveRun] = {}
        self._lock = asyncio.Lock()

    async def start(self, request: ModelRequest) -> asyncio.Task[RunSnapshot]:
        """Start a background run and return its task."""

        async with self._lock:
            if request.run_id in self._active:
                raise RunNotActiveError(f"run already active: {request.run_id}")
            self._store.create_run(
                request.run_id,
                prompt=request.prompt,
                model=request.model,
                metadata=dict(request.metadata),
            )
            active = _ActiveRun(cancel_event=asyncio.Event())
            task = asyncio.create_task(self._execute(request, active))
            active.task = task
            self._active[request.run_id] = active
            return task

    async def run(self, request: ModelRequest) -> RunSnapshot:
        """Start a run and wait until it reaches its single terminal state."""

        task = await self.start(request)
        return await task

    async def cancel(self, run_id: UUID) -> None:
        """Request cooperative cancellation without force-killing model code."""

        async with self._lock:
            active = self._active.get(run_id)
            if active is None:
                raise RunNotActiveError(f"run is not active: {run_id}")
            active.cancel_event.set()

    async def _execute(
        self,
        request: ModelRequest,
        active: _ActiveRun,
    ) -> RunSnapshot:
        started_at = time.perf_counter()
        delta_count = 0
        accumulated_text = ""
        ttft_recorded = False
        try:
            async for delta in self._model.stream(request, active.cancel_event):
                if active.cancel_event.is_set():
                    raise ModelCancelled()
                delta_count += 1
                delta_text = getattr(delta, "text", "")
                delta_metadata = getattr(delta, "metadata", {})
                delta_usage = getattr(delta, "usage", None)
                if isinstance(delta_metadata, dict):
                    if delta_metadata.get("refusal") is not None:
                        raise ModelStreamError("content_refused", str(delta_metadata["refusal"]))
                    if delta_metadata.get("finish_reason") in {"length", "max_tokens", "truncated"}:
                        raise ModelStreamError("output_truncated", "model output was truncated")
                accumulated_text += delta_text
                if delta_text or delta_metadata:
                    self._store.append(
                        request.run_id,
                        RunEventType.DELTA,
                        {"text": delta_text, "metadata": dict(delta_metadata)},
                    )
                if delta_usage is not None and isinstance(self._store, UsageStore):
                    self._store.record_usage(request.run_id, delta_usage)
                if delta_text and not ttft_recorded:
                    if isinstance(self._store, TTFTStore):
                        self._store.record_ttft(
                            request.run_id,
                            (time.perf_counter() - started_at) * 1000,
                        )
                    ttft_recorded = True
                if (
                    self._checkpoint_interval
                    and delta_count % self._checkpoint_interval == 0
                ):
                    if isinstance(self._store, CheckpointStore):
                        last_sequence = self._store.get(request.run_id).events[-1].sequence
                        self._store.save_checkpoint(
                            request.run_id,
                            last_sequence,
                            {"text": accumulated_text, "delta_count": delta_count},
                        )
            if active.cancel_event.is_set():
                raise ModelCancelled()
            if self._output_validator is not None:
                try:
                    self._output_validator(accumulated_text)
                except OutputValidationError:
                    raise
                except Exception as error:
                    raise OutputValidationError(str(error)) from error
            self._store.append(
                request.run_id,
                RunEventType.COMPLETED,
                terminal_state=RunTerminalState.COMPLETED,
            )
        except ModelCancelled as error:
            self._append_cancelled(request.run_id, str(error))
        except ModelStreamError as error:
            self._append_failed(request.run_id, error.code, error.message)
        except OutputValidationError as error:
            if self._correction_model is None:
                self._append_failed(request.run_id, error.code, str(error))
            else:
                try:
                    self._store.append(
                        request.run_id,
                        RunEventType.RETRY,
                        {"reason": error.code, "message": str(error), "attempt": 1},
                    )
                    corrected_text = await self._consume_correction(request, active)
                    self._output_validator(corrected_text)
                    self._store.append(
                        request.run_id,
                        RunEventType.COMPLETED,
                        terminal_state=RunTerminalState.COMPLETED,
                    )
                except OutputValidationError as correction_error:
                    self._append_failed(
                        request.run_id, correction_error.code, str(correction_error)
                    )
                except ModelStreamError as correction_error:
                    self._append_failed(
                        request.run_id, correction_error.code, correction_error.message
                    )
                except Exception as correction_error:
                    self._append_failed(
                        request.run_id, "correction_failed", str(correction_error)
                    )
        except asyncio.CancelledError:
            self._append_cancelled(request.run_id, "agent task cancelled")
            raise
        except Exception as error:
            self._append_failed(request.run_id, "agent_error", str(error))
        finally:
            async with self._lock:
                self._active.pop(request.run_id, None)
        return self._store.get(request.run_id)

    async def _consume_correction(
        self,
        request: ModelRequest,
        active: _ActiveRun,
    ) -> str:
        corrected_request = ModelRequest(
            run_id=request.run_id,
            prompt=(
                f"{request.prompt}\n\nReturn a corrected structured answer. "
                "The previous answer failed validation."
            ),
            model=request.model,
            metadata={**request.metadata, "correction_attempt": 1},
        )
        text_parts: list[str] = []
        async for delta in self._correction_model.stream(
            corrected_request, active.cancel_event
        ):
            if active.cancel_event.is_set():
                raise ModelCancelled()
            delta_text = getattr(delta, "text", "")
            delta_metadata = getattr(delta, "metadata", {})
            if isinstance(delta_metadata, dict) and delta_metadata.get("refusal"):
                raise ModelStreamError("content_refused", str(delta_metadata["refusal"]))
            text_parts.append(delta_text)
            if delta_text or delta_metadata:
                self._store.append(
                    request.run_id,
                    RunEventType.DELTA,
                    {"text": delta_text, "metadata": dict(delta_metadata)},
                )
        return "".join(text_parts)

    def _append_cancelled(self, run_id: UUID, message: str) -> None:
        self._store.append(
            run_id,
            RunEventType.CANCELLED,
            {"message": message},
            terminal_state=RunTerminalState.CANCELLED,
        )

    def _append_failed(self, run_id: UUID, code: str, message: str) -> None:
        self._store.append(
            run_id,
            RunEventType.FAILED,
            {"message": message},
            terminal_state=RunTerminalState.FAILED,
            error_code=code,
        )

    def active_run_count(self) -> int:
        """Return the number of tasks currently owned by this loop."""

        return len(self._active)

    def active_run_ids(self) -> tuple[UUID, ...]:
        """Return active IDs for diagnostics and the future gateway endpoint."""

        return tuple(self._active)
