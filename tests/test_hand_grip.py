from __future__ import annotations

import struct

import numpy as np
import pytest

from sim2real.utils.common import HandGripMessage
from sim2real.utils.dh116s import (
    DH116SHandDriver,
    HandGripConsumer,
    JOINT_RAD_RANGES,
    TRIGGER_CLOSE_RATIOS,
    grip_to_joint_targets,
)


def test_hand_grip_message_round_trip() -> None:
    message = HandGripMessage(timestamp_ns=123, left_grip=0.25, right_grip=0.75)

    encoded = message.to_bytes()
    decoded = HandGripMessage.from_bytes(encoded)

    assert len(encoded) == struct.calcsize("<Qff")
    assert decoded.timestamp_ns == 123
    assert decoded.left_grip == pytest.approx(0.25)
    assert decoded.right_grip == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"timestamp_ns": -1}, "uint64"),
        ({"timestamp_ns": 2**64}, "uint64"),
        ({"left_grip": -0.1}, "within"),
        ({"right_grip": 1.1}, "within"),
        ({"left_grip": np.nan}, "finite"),
        ({"right_grip": np.inf}, "finite"),
    ],
)
def test_hand_grip_message_rejects_invalid_fields(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        HandGripMessage(**kwargs)


def test_hand_grip_message_rejects_invalid_wire_data() -> None:
    with pytest.raises(ValueError, match="invalid size"):
        HandGripMessage.from_bytes(b"short")
    with pytest.raises(ValueError, match="finite"):
        HandGripMessage.from_bytes(struct.pack("<Qff", 0, np.nan, 0.0))


@pytest.mark.parametrize("grip", [0.0, 0.5, 1.0])
def test_grip_mapping_preserves_conservative_closure(grip: float) -> None:
    targets = grip_to_joint_targets(grip)

    assert targets.shape == (6,)
    assert targets[0] == pytest.approx(JOINT_RAD_RANGES[0][1])
    for idx in range(1, 6):
        expected = JOINT_RAD_RANGES[idx][1] * TRIGGER_CLOSE_RATIOS[idx] * grip
        assert targets[idx] == pytest.approx(expected)
        assert targets[idx] <= JOINT_RAD_RANGES[idx][1] * 0.4 + 1e-6


class FakeLHP:
    def __init__(self) -> None:
        self._lib = None
        self._handle = None
        self.hand_direction = None
        self.move_no_home = None
        self.target_angles: dict[int, float] = {}
        self.angular_velocities: dict[int, float] = {}
        self.max_currents: dict[int, int] = {}
        self.move_count = 0

    def set_hand_direction(self, value: int) -> None:
        self.hand_direction = value

    def set_move_no_home(self, value: int) -> None:
        self.move_no_home = value

    def set_target_angle(self, motor_id: int, value: float) -> None:
        self.target_angles[motor_id] = value

    def set_angular_velocity(self, motor_id: int, value: float) -> None:
        self.angular_velocities[motor_id] = value

    def set_max_current(self, motor_id: int, value: int) -> None:
        self.max_currents[motor_id] = value

    def move_motors(self, mode: int) -> None:
        assert mode == 0
        self.move_count += 1

    def get_now_angle(self, motor_id: int) -> float:
        return self.target_angles.get(motor_id, 0.0)


class FakeController:
    instances: list["FakeController"] = []

    def __init__(self, canfd_node_id: int) -> None:
        self.canfd_node_id = canfd_node_id
        self.lhp = FakeLHP()
        self.connect_kwargs = None
        self.enabled = False
        self.home_wait_time = None
        self.disconnected = False
        self.__class__.instances.append(self)

    def connect(self, **kwargs) -> bool:
        self.connect_kwargs = kwargs
        return True

    def get_alarm(self) -> bool:
        return False

    def enable_motors(self, enabled: bool) -> None:
        self.enabled = enabled

    def home(self, wait_time: float) -> None:
        self.home_wait_time = wait_time

    def disconnect(self) -> None:
        self.disconnected = True


def test_dual_driver_uses_expected_can_mapping_and_auto_homes() -> None:
    FakeController.instances.clear()
    driver = DH116SHandDriver(controller_cls=FakeController, hand_dir="double")
    try:
        left, right = FakeController.instances
        assert (left.canfd_node_id, left.connect_kwargs["device_index"]) == (1, 0)
        assert (right.canfd_node_id, right.connect_kwargs["device_index"]) == (1, 1)
        assert left.enabled and right.enabled
        assert left.home_wait_time == pytest.approx(2.0)
        assert right.home_wait_time == pytest.approx(2.0)
        assert left.lhp.hand_direction == 1
        assert right.lhp.hand_direction == 0
        assert left.lhp.move_no_home == 1
        assert right.lhp.move_no_home == 1

        driver.apply_grips(0.0, 1.0)
        assert left.lhp.move_count == 1
        assert right.lhp.move_count == 1
        assert left.lhp.target_angles[2] == pytest.approx(0.0)
        full_joint_2_deg = np.rad2deg(JOINT_RAD_RANGES[1][1])
        assert right.lhp.target_angles[2] == pytest.approx(full_joint_2_deg * 0.4)
    finally:
        driver.close()

    assert all(controller.disconnected for controller in FakeController.instances)


def test_dual_driver_accepts_host_specific_can_device_indices() -> None:
    FakeController.instances.clear()
    driver = DH116SHandDriver(
        controller_cls=FakeController,
        hand_dir="double",
        left_device_index=2,
        right_device_index=1,
    )
    try:
        left, right = FakeController.instances
        assert left.connect_kwargs["device_index"] == 2
        assert right.connect_kwargs["device_index"] == 1
    finally:
        driver.close()


class RecordingDriver:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float]] = []

    def apply_grips(self, left_grip: float, right_grip: float) -> None:
        self.commands.append((left_grip, right_grip))


def test_consumer_decodes_and_applies_independent_hands() -> None:
    driver = RecordingDriver()
    consumer = HandGripConsumer(driver)

    decoded = consumer.consume(
        HandGripMessage(timestamp_ns=42, left_grip=0.2, right_grip=0.8).to_bytes()
    )

    assert decoded.timestamp_ns == 42
    assert driver.commands == [(pytest.approx(0.2), pytest.approx(0.8))]
    assert consumer.last_message is decoded


def test_consumer_does_not_apply_malformed_message() -> None:
    driver = RecordingDriver()
    consumer = HandGripConsumer(driver)

    with pytest.raises(ValueError, match="invalid size"):
        consumer.consume(b"invalid")

    assert driver.commands == []


def test_consumer_stale_state_holds_last_command_until_recovery() -> None:
    driver = RecordingDriver()
    consumer = HandGripConsumer(driver)
    first = HandGripMessage(timestamp_ns=1, left_grip=0.3, right_grip=0.7)

    consumer.consume(first.to_bytes(), received_monotonic=10.0)
    assert not consumer.is_stale(now_monotonic=10.9, timeout_s=1.0)
    assert consumer.is_stale(now_monotonic=11.0, timeout_s=1.0)
    assert driver.commands == [(pytest.approx(0.3), pytest.approx(0.7))]

    second = HandGripMessage(timestamp_ns=2, left_grip=0.4, right_grip=0.6)
    consumer.consume(second.to_bytes(), received_monotonic=12.0)
    assert not consumer.is_stale(now_monotonic=12.0, timeout_s=1.0)
    assert len(driver.commands) == 2
