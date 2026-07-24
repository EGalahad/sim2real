"""DH116S hand control helpers for normalized PICO grip commands."""

from __future__ import annotations

from ctypes import POINTER, byref, c_float, c_int, c_void_p
import os
from pathlib import Path
import sys
import threading
import time
from typing import Literal

import numpy as np

from sim2real.utils.common import HandGripMessage


HandSide = Literal["left", "right", "double"]

DH116S_NUM_MOTORS = 6
DH116S_JOINT_NAMES = (
    "thumb_abd",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
)
JOINT_RAD_RANGES = (
    (0.0, 1.588),
    (0.0, 1.030),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.291),
)
TRIGGER_CLOSE_RATIOS = np.asarray(
    [1.0, 0.4, 0.4, 0.4, 0.4, 0.4],
    dtype=np.float32,
)
TRIGGER_CLOSE_Q = np.asarray(
    [upper * ratio for (_, upper), ratio in zip(JOINT_RAD_RANGES, TRIGGER_CLOSE_RATIOS)],
    dtype=np.float32,
)

DEFAULT_SDK_DIR = str(
    Path(__file__).resolve().parents[2] / "third_party" / "dh116s_sdk" / "python"
)
DEFAULT_LEFT_NODE_ID = 1
DEFAULT_RIGHT_NODE_ID = 1
DEFAULT_LEFT_DEVICE_INDEX = 0
DEFAULT_RIGHT_DEVICE_INDEX = 1
DEFAULT_MAX_CURRENT = 800
DEFAULT_ANGULAR_VELOCITY = 200.0
DEFAULT_HOME_WAIT_TIME = 2.0


def grip_to_joint_targets(grip: float) -> np.ndarray:
    """Map normalized grip to the conservative DH116S six-motor grasp pose."""
    grip_value = float(grip)
    if not np.isfinite(grip_value) or not 0.0 <= grip_value <= 1.0:
        raise ValueError("grip must be finite and within [0, 1]")

    targets = (TRIGGER_CLOSE_Q * grip_value).astype(np.float32)
    # Keep thumb abduction at the grasp alignment used by the reference driver.
    targets[0] = JOINT_RAD_RANGES[0][1]
    return targets


class DH116SHandDriver:
    """Thin dual-hand adapter around the LHandPro Python SDK."""

    def __init__(
        self,
        *,
        hand_dir: HandSide = "double",
        sdk_dir: str = DEFAULT_SDK_DIR,
        left_node_id: int = DEFAULT_LEFT_NODE_ID,
        right_node_id: int = DEFAULT_RIGHT_NODE_ID,
        left_device_index: int = DEFAULT_LEFT_DEVICE_INDEX,
        right_device_index: int = DEFAULT_RIGHT_DEVICE_INDEX,
        max_current: int = DEFAULT_MAX_CURRENT,
        angular_velocity: float = DEFAULT_ANGULAR_VELOCITY,
        home_wait_time: float = DEFAULT_HOME_WAIT_TIME,
        controller_cls=None,
    ) -> None:
        if hand_dir not in {"left", "right", "double"}:
            raise ValueError("hand_dir must be 'left', 'right', or 'double'")
        if not 0 <= int(max_current) <= 1000:
            raise ValueError("max_current must be within [0, 1000]")
        if float(angular_velocity) <= 0:
            raise ValueError("angular_velocity must be > 0")
        if float(home_wait_time) < 0:
            raise ValueError("home_wait_time must be >= 0")
        if int(left_device_index) < 0 or int(right_device_index) < 0:
            raise ValueError("CAN device indices must be >= 0")

        self.hand_dir = hand_dir
        self.sdk_dir = os.path.expanduser(sdk_dir)
        self.max_current = int(max_current)
        self.angular_velocity = float(angular_velocity)
        self.home_wait_time = float(home_wait_time)
        self._sdk_lock = threading.Lock()
        self._controllers: dict[str, object] = {}
        self._angle_limits_deg: dict[str, list[tuple[float, float]] | None] = {}

        if controller_cls is None:
            controller_cls = self._load_controller_class()

        side_configs = {
            "left": (int(left_node_id), int(left_device_index)),
            "right": (int(right_node_id), int(right_device_index)),
        }
        requested_sides = ("left", "right") if hand_dir == "double" else (hand_dir,)
        try:
            for side in requested_sides:
                node_id, device_index = side_configs[side]
                self._connect_side(controller_cls, side, node_id, device_index)
        except Exception:
            self.close()
            raise

    def _load_controller_class(self):
        if not os.path.isdir(self.sdk_dir):
            raise FileNotFoundError(f"DH116S SDK directory not found: {self.sdk_dir}")
        if self.sdk_dir not in sys.path:
            sys.path.insert(1, self.sdk_dir)
        try:
            from lhandprolib_python_sdk.controller import LHandProController
        except ImportError as exc:
            raise ImportError(f"Failed to import LHandPro SDK from {self.sdk_dir}: {exc}") from exc
        return LHandProController

    def _connect_side(self, controller_cls, side: str, node_id: int, device_index: int) -> None:
        print(
            f"[DH116S] connecting {side}: device_index={device_index} node_id={node_id}",
            flush=True,
        )
        controller = controller_cls(canfd_node_id=node_id)
        connected = controller.connect(
            enable_motors=True,
            home_motors=False,
            home_wait_time=self.home_wait_time,
            device_index=device_index,
            auto_select=False,
        )
        if not connected:
            try:
                controller.disconnect()
            except Exception:
                pass
            raise RuntimeError(
                f"DH116S {side} CANFD connection failed: "
                f"device_index={device_index} node_id={node_id}"
            )

        try:
            limits = self._query_sdk_angle_limits(controller.lhp)
            if controller.lhp is not None:
                controller.lhp.set_hand_direction(1 if side == "left" else 0)
                controller.lhp.set_move_no_home(0)
            get_alarm = getattr(controller, "get_alarm", None)
            if callable(get_alarm) and get_alarm():
                print(f"[DH116S] clearing existing {side} alarm", flush=True)
                controller.clear_alarm()
                time.sleep(0.5)
            controller.enable_motors(True)
            controller.home(wait_time=self.home_wait_time)
            if controller.lhp is not None:
                # The vendor synchronous home() can report completion while its
                # internal home gate still rejects the first move with Error 11.
                # The hand has physically homed above; release that stale gate.
                controller.lhp.set_move_no_home(1)
        except Exception:
            controller.disconnect()
            raise

        self._controllers[side] = controller
        self._angle_limits_deg[side] = limits
        print(f"[DH116S] ready: {side}", flush=True)

    @staticmethod
    def _query_sdk_angle_limits(lhp) -> list[tuple[float, float]] | None:
        if lhp is None:
            return None
        lib = getattr(lhp, "_lib", None)
        handle = getattr(lhp, "_handle", None)
        if lib is None or handle is None or not hasattr(lib, "lhandprolib_get_limit_target_angle"):
            return None

        lib.lhandprolib_get_limit_target_angle.restype = c_int
        lib.lhandprolib_get_limit_target_angle.argtypes = [
            c_void_p,
            c_int,
            POINTER(c_float),
            POINTER(c_float),
        ]
        limits: list[tuple[float, float]] = []
        try:
            for motor_id in range(1, DH116S_NUM_MOTORS + 1):
                angle_a = c_float()
                angle_b = c_float()
                result = lib.lhandprolib_get_limit_target_angle(
                    handle,
                    motor_id,
                    byref(angle_a),
                    byref(angle_b),
                )
                if result != 0:
                    raise RuntimeError(
                        f"get_limit_target_angle failed for motor {motor_id}: {result}"
                    )
                limits.append(tuple(sorted((float(angle_a.value), float(angle_b.value)))))
        except Exception as exc:
            print(f"[DH116S] SDK angle-limit query failed, using URDF limits: {exc}")
            return None
        return limits

    @staticmethod
    def _urdf_rad_to_sdk_angle_deg(
        joint_idx: int,
        joint_rad: float,
        limits: list[tuple[float, float]] | None,
    ) -> float:
        urdf_lower, urdf_upper = JOINT_RAD_RANGES[joint_idx]
        ratio = float(
            np.clip(
                (joint_rad - urdf_lower) / (urdf_upper - urdf_lower),
                0.0,
                1.0,
            )
        )
        if limits is None:
            sdk_lower = np.rad2deg(urdf_lower)
            sdk_upper = np.rad2deg(urdf_upper)
        else:
            sdk_lower, sdk_upper = limits[joint_idx]
        return float(sdk_lower + ratio * (sdk_upper - sdk_lower))

    @staticmethod
    def _sdk_angle_deg_to_urdf_rad(
        joint_idx: int,
        angle_deg: float,
        limits: list[tuple[float, float]] | None,
    ) -> float:
        urdf_lower, urdf_upper = JOINT_RAD_RANGES[joint_idx]
        if limits is None:
            return float(np.clip(np.deg2rad(angle_deg), urdf_lower, urdf_upper))
        sdk_lower, sdk_upper = limits[joint_idx]
        if np.isclose(sdk_upper, sdk_lower):
            return urdf_lower
        ratio = float(np.clip((angle_deg - sdk_lower) / (sdk_upper - sdk_lower), 0.0, 1.0))
        return float(urdf_lower + ratio * (urdf_upper - urdf_lower))

    def apply_grips(self, left_grip: float, right_grip: float) -> None:
        targets = {
            "left": grip_to_joint_targets(left_grip),
            "right": grip_to_joint_targets(right_grip),
        }
        for side, controller in self._controllers.items():
            self._send_targets(
                controller,
                targets[side],
                self._angle_limits_deg[side],
            )

    def _send_targets(
        self,
        controller,
        targets: np.ndarray,
        limits: list[tuple[float, float]] | None,
    ) -> None:
        if controller.lhp is None:
            raise RuntimeError("DH116S controller has no active LHandPro handle")
        with self._sdk_lock:
            for joint_idx, target in enumerate(np.asarray(targets, dtype=np.float32)):
                angle_deg = self._urdf_rad_to_sdk_angle_deg(joint_idx, float(target), limits)
                motor_id = joint_idx + 1
                controller.lhp.set_target_angle(motor_id, angle_deg)
                controller.lhp.set_angular_velocity(motor_id, self.angular_velocity)
                controller.lhp.set_max_current(motor_id, self.max_current)
            controller.lhp.move_motors(0)

    def read_feedback(self) -> dict[str, np.ndarray]:
        feedback: dict[str, np.ndarray] = {}
        for side, controller in self._controllers.items():
            if controller.lhp is None:
                continue
            values = np.zeros(DH116S_NUM_MOTORS, dtype=np.float32)
            with self._sdk_lock:
                for joint_idx in range(DH116S_NUM_MOTORS):
                    angle_deg = controller.lhp.get_now_angle(joint_idx + 1)
                    values[joint_idx] = self._sdk_angle_deg_to_urdf_rad(
                        joint_idx,
                        angle_deg,
                        self._angle_limits_deg[side],
                    )
            feedback[side] = values
        return feedback

    def close(self) -> None:
        for controller in self._controllers.values():
            try:
                controller.disconnect()
            except Exception:
                pass
        self._controllers.clear()
        self._angle_limits_deg.clear()


class DryRunDH116SHandDriver:
    """No-hardware driver used to validate transport and mapping."""

    def __init__(self, hand_dir: HandSide = "double") -> None:
        self.hand_dir = hand_dir
        self.last_grips: tuple[float, float] | None = None
        self.last_targets: dict[str, np.ndarray] = {}

    def apply_grips(self, left_grip: float, right_grip: float) -> None:
        self.last_grips = (float(left_grip), float(right_grip))
        sides = ("left", "right") if self.hand_dir == "double" else (self.hand_dir,)
        values = {"left": left_grip, "right": right_grip}
        self.last_targets = {side: grip_to_joint_targets(values[side]) for side in sides}

    def read_feedback(self) -> dict[str, np.ndarray]:
        return {side: target.copy() for side, target in self.last_targets.items()}

    def close(self) -> None:
        pass


class HandGripConsumer:
    """Decode hand-grip messages and apply them to a DH116S-compatible driver."""

    def __init__(self, driver) -> None:
        self.driver = driver
        self.last_message: HandGripMessage | None = None
        self.last_receive_monotonic: float | None = None

    def consume(
        self,
        data: bytes,
        *,
        received_monotonic: float | None = None,
    ) -> HandGripMessage:
        message = HandGripMessage.from_bytes(data)
        self.driver.apply_grips(message.left_grip, message.right_grip)
        self.last_message = message
        self.last_receive_monotonic = (
            time.monotonic() if received_monotonic is None else float(received_monotonic)
        )
        return message

    def is_stale(self, *, now_monotonic: float, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        return (
            self.last_receive_monotonic is not None
            and now_monotonic - self.last_receive_monotonic >= timeout_s
        )
