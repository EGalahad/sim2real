from __future__ import annotations

import pytest


pytest.importorskip("general_motion_retargeting")

from sim2real.teleop.pico_retarget_pub import (
    _hand_grip_message_from_data,
    _pico_controller_state_from_data,
)


def test_hand_grip_message_uses_independent_analog_triggers() -> None:
    message = _hand_grip_message_from_data(
        {
            "LeftController": {"index_trig": 0.25},
            "RightController": {"index_trig": 0.75},
            "timestamp": 123,
        }
    )

    assert message.timestamp_ns == 123
    assert message.left_grip == pytest.approx(0.25)
    assert message.right_grip == pytest.approx(0.75)


def test_hand_grip_message_clamps_and_defaults_malformed_triggers() -> None:
    message = _hand_grip_message_from_data(
        {
            "LeftController": {"index_trig": -2.0},
            "RightController": {"index_trig": "bad"},
            "timestamp": 456,
        }
    )

    assert message.left_grip == 0.0
    assert message.right_grip == 0.0

    high_message = _hand_grip_message_from_data(
        {
            "LeftController": {"index_trig": 2.0},
            "RightController": {"index_trig": float("nan")},
            "timestamp": 789,
        }
    )
    assert high_message.left_grip == 1.0
    assert high_message.right_grip == 0.0


def test_hand_grip_message_defaults_missing_controller_data() -> None:
    message = _hand_grip_message_from_data(None)

    assert message.timestamp_ns == 0
    assert message.left_grip == 0.0
    assert message.right_grip == 0.0


def test_existing_pico_button_mapping_is_unchanged() -> None:
    state = _pico_controller_state_from_data(
        {
            "LeftController": {"key_one": True, "key_two": False},
            "RightController": {"key_one": False, "key_two": True},
            "timestamp": 321,
        }
    )

    assert state.timestamp_ns == 321
    assert not state.A
    assert state.B
    assert state.X
    assert not state.Y
