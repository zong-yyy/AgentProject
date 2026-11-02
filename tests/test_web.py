import unittest
import os
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient

from src.streaming_agent_gateway.events import RunEventType
from src.streaming_agent_gateway.sse import SSEDecoder, decode_run_event
from src.streaming_agent_gateway.web import create_app


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.client = TestClient(create_app())

    def test_submit_snapshot_and_sse_stream(self) -> None:
        response = self.client.post("/runs", json={"prompt": "hello"})
        self.assertEqual(response.status_code, 202)
        run_id = UUID(response.json()["run_id"])

        snapshot = self.client.get(f"/runs/{run_id}")
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.json()["terminal_state"], "completed")

        stream = self.client.get(f"/runs/{run_id}/events")
        self.assertEqual(stream.status_code, 200)
        self.assertEqual(stream.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: run.delta", stream.text)
        messages = SSEDecoder().feed(stream.text)
        events = [decode_run_event(message) for message in messages]
        output = "".join(
            event.payload["text"]
            for event in events
            if event.type is RunEventType.DELTA
        )
        self.assertEqual(output, "Echo: hello")

    def test_last_event_id_replays_only_later_events(self) -> None:
        run_id = UUID(self.client.post("/runs", json={"prompt": "hello"}).json()["run_id"])

        stream = self.client.get(
            f"/runs/{run_id}/events",
            headers={"Last-Event-ID": "2"},
        )

        self.assertNotIn("id: 1\n", stream.text)
        self.assertNotIn("id: 2\n", stream.text)
        self.assertIn("id: 3\n", stream.text)

    def test_active_run_endpoint_returns_count(self) -> None:
        response = self.client.get("/runs/active")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 0, "run_ids": []})

    def test_run_history_reports_prompt_and_terminal_state(self) -> None:
        created = self.client.post("/runs", json={"prompt": "history prompt"}).json()

        response = self.client.get("/runs")

        self.assertEqual(response.status_code, 200)
        item = response.json()["runs"][0]
        self.assertEqual(item["run_id"], created["run_id"])
        self.assertEqual(item["prompt"], "history prompt")
        self.assertEqual(item["terminal_state"], "completed")

    def test_unknown_run_and_invalid_cursor_are_rejected(self) -> None:
        run_id = UUID(int=0)
        self.assertEqual(self.client.get(f"/runs/{run_id}/events").status_code, 404)

        created = UUID(self.client.post("/runs", json={"prompt": "hello"}).json()["run_id"])
        invalid = self.client.get(
            f"/runs/{created}/events",
            headers={"Last-Event-ID": "not-a-number"},
        )
        self.assertEqual(invalid.status_code, 400)

        negative = self.client.get(
            f"/runs/{created}/events",
            headers={"Last-Event-ID": "-1"},
        )
        self.assertEqual(negative.status_code, 400)

    def test_client_page_is_available(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Streaming Agent Gateway", response.text)
        self.assertIn('connectedSource.addEventListener(eventType, handleEvent)', response.text)
        self.assertNotIn("source.onmessage", response.text)
        self.assertIn("localStorage.setItem(currentRunKey, runId)", response.text)
        self.assertIn("if (snapshot.terminal_state)", response.text)
        self.assertIn("localStorage.removeItem(currentRunKey)", response.text)
        self.assertIn("applySnapshot(snapshot)", response.text)
        self.assertIn('button id="unsubscribe" disabled>取消订阅</button>', response.text)
        self.assertIn('$("unsubscribe").onclick = unsubscribe', response.text)
        self.assertIn("connect(lastSequence)", response.text)
        self.assertIn("overflow-wrap: anywhere", response.text)
        self.assertIn('id="history-list"', response.text)
        self.assertIn('fetch("/runs")', response.text)
        self.assertIn("item.onclick = () => openRun(run.run_id)", response.text)
        self.assertIn('$("prompt").value = started?.payload?.prompt || ""', response.text)


if __name__ == "__main__":
    unittest.main()
