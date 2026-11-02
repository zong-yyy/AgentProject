import unittest
from uuid import uuid4

from src.streaming_agent_gateway.events import (
    RunEvent,
    RunEventType,
    RunTerminalState,
    validate_event_sequence,
)


class RunEventTests(unittest.TestCase):
    def test_round_trip_preserves_event(self) -> None:
        run_id = uuid4()
        event = RunEvent.create(
            run_id,
            1,
            RunEventType.DELTA,
            {"text": "hello"},
        )

        restored = RunEvent.from_dict(event.to_dict())

        self.assertEqual(restored, event)

    def test_sequence_has_one_terminal_event(self) -> None:
        run_id = uuid4()
        events = [
            RunEvent.create(run_id, 1, RunEventType.STARTED),
            RunEvent.create(run_id, 2, RunEventType.DELTA, {"text": "done"}),
            RunEvent.create(
                run_id,
                3,
                RunEventType.COMPLETED,
                terminal_state=RunTerminalState.COMPLETED,
            ),
        ]

        validate_event_sequence(events)

        with self.assertRaises(ValueError):
            validate_event_sequence(events + [RunEvent.create(run_id, 4, RunEventType.DELTA)])

    def test_failed_event_requires_error_code(self) -> None:
        with self.assertRaises(ValueError):
            RunEvent.create(
                uuid4(),
                1,
                RunEventType.FAILED,
                terminal_state=RunTerminalState.FAILED,
            )


if __name__ == "__main__":
    unittest.main()
