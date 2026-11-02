"""FastAPI HTTP and SSE boundary for the streaming gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent_loop import AgentLoop, RunNotActiveError
from .events import RunEvent
from .gateway import RunGateway
from .model_adapter import create_configured_model
from .run_store import InMemoryRunStore, RunNotFoundError, RunStoreError
from .sse import SSEEncoder
from .sqlite_store import SQLiteRunStore
from .streaming_model import ModelRequest


class RunRequestBody(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RunResponse(BaseModel):
    run_id: UUID


class ActiveRunsResponse(BaseModel):
    count: int
    run_ids: list[UUID]


def create_app(
    gateway: RunGateway | None = None,
    db_path: str | None = None,
) -> FastAPI:
    """Build the standalone gateway application."""

    if gateway is None:
        configured_path = db_path or os.getenv("STREAMING_GATEWAY_DB")
        store = SQLiteRunStore(configured_path) if configured_path else InMemoryRunStore()
        gateway = RunGateway(AgentLoop(create_configured_model(), store), store)

    app = FastAPI(title="Streaming Agent Gateway", version="0.1.0")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def client() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.post("/runs", response_model=RunResponse, status_code=202)
    async def submit_run(body: RunRequestBody) -> RunResponse:
        run_id = uuid4()
        await gateway.submit(
            ModelRequest(
                run_id=run_id,
                prompt=body.prompt,
                model=body.model,
                metadata=body.metadata,
            )
        )
        return RunResponse(run_id=run_id)

    @app.get("/runs")
    async def run_history() -> dict[str, object]:
        runs = []
        for snapshot in gateway.list_runs():
            started = snapshot.events[0]
            runs.append(
                {
                    "run_id": str(snapshot.run_id),
                    "prompt": started.payload.get("prompt", ""),
                    "started_at": started.timestamp.isoformat(),
                    "terminal_state": snapshot.terminal_state,
                    "last_sequence": snapshot.events[-1].sequence,
                }
            )
        return {"runs": runs}

    @app.post("/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: UUID) -> dict[str, str]:
        try:
            await gateway.cancel(run_id)
        except RunNotActiveError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "cancellation_requested"}

    @app.get("/runs/active", response_model=ActiveRunsResponse)
    async def active_runs() -> ActiveRunsResponse:
        run_ids = list(gateway.active_run_ids())
        return ActiveRunsResponse(count=len(run_ids), run_ids=run_ids)

    @app.get("/runs/{run_id}")
    async def run_snapshot(run_id: UUID) -> dict[str, object]:
        try:
            snapshot = gateway.snapshot(run_id)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        usage = gateway.usage(run_id)
        return {
            "run_id": str(snapshot.run_id),
            "terminal_state": snapshot.terminal_state,
            "usage": (
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage is not None
                else None
            ),
            "events": [event.to_dict() for event in snapshot.events],
        }

    @app.get("/runs/{run_id}/events")
    async def subscribe_run(
        run_id: UUID,
        after_sequence: int | None = Query(default=None, ge=0),
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        cursor = after_sequence
        if cursor is None and last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from error
            if cursor < 0:
                raise HTTPException(status_code=400, detail="invalid Last-Event-ID")
        if cursor is None:
            cursor = 0

        try:
            gateway.snapshot(run_id)
        except RunNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def event_stream() -> AsyncIterator[str]:
            async for event in gateway.subscribe(run_id, after_sequence=cursor):
                yield SSEEncoder.encode_event(event)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
