import unittest
from uuid import uuid4

from src.streaming_agent_gateway.events import RunEvent, RunEventType
from src.streaming_agent_gateway.sse import (
    SSEDecoder,
    SSEEncoder,
    decode_run_event,
)


class SSECodecTests(unittest.TestCase):
    def test_single_frame_survives_every_split_position(self) -> None:
        frame = "id: 7\nevent: run.delta\ndata: hello\n\n"

        for split_at in range(len(frame) + 1):
            with self.subTest(split_at=split_at):
                decoder = SSEDecoder()
                messages = decoder.feed(frame[:split_at])
                messages.extend(decoder.feed(frame[split_at:]))

                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].event_id, "7")
                self.assertEqual(messages[0].event, "run.delta")
                self.assertEqual(messages[0].data, "hello")

    def test_multiple_frames_in_one_chunk_preserve_all_events(self) -> None:
        decoder = SSEDecoder()
        chunk = "id: 1\nevent: first\ndata: one\n\nid: 2\nevent: second\ndata: two\n\n"

        messages = decoder.feed(chunk)

        self.assertEqual([message.event_id for message in messages], ["1", "2"])
        self.assertEqual([message.event for message in messages], ["first", "second"])
        self.assertEqual([message.data for message in messages], ["one", "two"])

    def test_unicode_survives_utf8_character_split_across_byte_chunks(self) -> None:
        frame = "event: text\ndata: 中文内容\n\n".encode("utf-8")
        character_start = frame.index("中".encode("utf-8"))
        split_at = character_start + 1
        decoder = SSEDecoder()

        messages = decoder.feed(frame[:split_at])
        messages.extend(decoder.feed(frame[split_at:]))

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].data, "中文内容")

    def test_run_event_round_trip_across_split_chunks(self) -> None:
        event = RunEvent.create(uuid4(), 7, RunEventType.DELTA, {"text": "hello"})
        encoded = SSEEncoder.encode_event(event)
        decoder = SSEDecoder()
        messages = []
        for chunk in (encoded[:4], encoded[4:19], encoded[19:]):
            messages.extend(decoder.feed(chunk))

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].event_id, "7")
        self.assertEqual(messages[0].event, "run.delta")
        self.assertEqual(decode_run_event(messages[0]), event)

    def test_decoder_handles_comments_crlf_and_multiple_data_lines(self) -> None:
        decoder = SSEDecoder()

        messages = decoder.feed(
            b": heartbeat\r\n\r\nid: 12\r\nevent: text\r\ndata: first\r\ndata: second\r\n\r\n"
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].data, "first\nsecond")
        self.assertEqual(messages[0].event_id, "12")

    def test_encoder_can_build_replayable_stream(self) -> None:
        run_id = uuid4()
        events = [
            RunEvent.create(run_id, 1, RunEventType.STARTED),
            RunEvent.create(run_id, 2, RunEventType.DELTA, {"text": "hi"}),
        ]

        stream = SSEEncoder.encode_many(events)

        self.assertEqual(stream.count("\n\n"), 2)
        self.assertIn("id: 1\n", stream)
        self.assertIn('"sequence":2', stream)

    def test_finish_rejects_truncated_line(self) -> None:
        decoder = SSEDecoder()
        decoder.feed("data: incomplete")

        with self.assertRaises(ValueError):
            decoder.finish()


if __name__ == "__main__":
    unittest.main()
