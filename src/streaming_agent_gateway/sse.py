"""Small dependency-free Server-Sent Events codec."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import json
from typing import Any, Iterable

from .events import RunEvent


@dataclass(frozen=True, slots=True)
class SSEMessage:
    """One decoded SSE message. Comments and heartbeat frames are omitted."""

    event: str | None
    data: str
    event_id: str | None = None
    retry: int | None = None


class SSEEncoder:
    """Encode run events using the SSE wire format."""

    @staticmethod
    def encode_event(event: RunEvent) -> str:
        """Encode one event; its sequence is the reconnect cursor."""

        data = json.dumps(
            event.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            f"id: {event.sequence}\n"
            f"event: {event.type.value}\n"
            f"data: {data}\n\n"
        )

    @classmethod
    def encode_many(cls, events: Iterable[RunEvent]) -> str:
        return "".join(cls.encode_event(event) for event in events)

    @staticmethod
    def encode_comment(comment: str = "heartbeat") -> str:
        """Encode a comment frame suitable for keeping a connection alive."""

        return f": {comment}\n\n"


class SSEDecoder:
    """Incrementally decode SSE text received from a possibly split stream."""

    def __init__(self) -> None:
        self._buffer = ""
        self._fields: dict[str, list[str] | str] = {}
        self._byte_decoder = codecs.getincrementaldecoder("utf-8")()

    def feed(self, chunk: str | bytes) -> list[SSEMessage]:
        """Consume a chunk and return every complete, non-comment message."""

        if isinstance(chunk, bytes):
            chunk = self._byte_decoder.decode(chunk, final=False)
        self._buffer += chunk
        messages: list[SSEMessage] = []
        while True:
            line_end = self._find_line_end()
            if line_end is None:
                break
            line, consumed = line_end
            self._buffer = self._buffer[consumed:]
            if line == "":
                message = self._dispatch()
                if message is not None:
                    messages.append(message)
            else:
                self._parse_line(line)
        return messages

    def finish(self) -> list[SSEMessage]:
        """Flush a final complete event, rejecting a truncated line."""

        self._buffer += self._byte_decoder.decode(b"", final=True)
        if self._buffer:
            raise ValueError("incomplete SSE line at end of stream")
        message = self._dispatch()
        return [message] if message is not None else []

    def _find_line_end(self) -> tuple[str, int] | None:
        for index, character in enumerate(self._buffer):
            if character == "\n":
                line = self._buffer[:index]
                if line.endswith("\r"):
                    line = line[:-1]
                return line, index + 1
            if character == "\r":
                if index + 1 == len(self._buffer):
                    return None
                consumed = index + 2 if self._buffer[index + 1] == "\n" else index + 1
                return self._buffer[:index], consumed
        return None

    def _parse_line(self, line: str) -> None:
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            data = self._fields.setdefault("data", [])
            assert isinstance(data, list)
            data.append(value)
        elif field in {"event", "id", "retry"}:
            self._fields[field] = value

    def _dispatch(self) -> SSEMessage | None:
        fields = self._fields
        self._fields = {}
        data = fields.get("data", [])
        if not isinstance(data, list) or not data:
            return None
        retry_value = fields.get("retry")
        retry = None
        if isinstance(retry_value, str) and retry_value.isdigit():
            retry = int(retry_value)
        event = fields.get("event")
        event_id = fields.get("id")
        return SSEMessage(
            event=event if isinstance(event, str) else None,
            data="\n".join(data),
            event_id=event_id if isinstance(event_id, str) else None,
            retry=retry,
        )


def decode_run_event(message: SSEMessage) -> RunEvent:
    """Decode the JSON data from an SSE message into a RunEvent."""

    value: Any = json.loads(message.data)
    if not isinstance(value, dict):
        raise ValueError("run event data must be a JSON object")
    return RunEvent.from_dict(value)
