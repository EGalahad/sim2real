from __future__ import annotations

import json
import threading
import time
from bisect import bisect_right
from typing import Any, Iterable

import numpy as np
import zmq

from loguru import logger
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.utils.motion import MotionData


def _ensure_np(value: Any, ndim: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"Expected ndim={ndim}, got shape={arr.shape}")
    return arr


def _normalize_quat_batch(quat_wxyz: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    return quat_wxyz / denom


def _quat_slerp_batch(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_batch(q0_wxyz.astype(np.float32, copy=False))
    q1 = _normalize_quat_batch(q1_wxyz.astype(np.float32, copy=False))

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)

    linear_mask = np.abs(dot) > 0.9995
    alpha_arr = np.full_like(dot, float(alpha), dtype=np.float32)

    lerp = _normalize_quat_batch((1.0 - alpha_arr) * q0 + alpha_arr * q1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha_arr
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / np.clip(sin_theta_0, 1e-8, None)
    s1 = sin_theta / np.clip(sin_theta_0, 1e-8, None)
    slerp = _normalize_quat_batch(s0 * q0 + s1 * q1)
    return np.where(linear_mask, lerp, slerp).astype(np.float32)


def _frame_with_zero_velocities(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "joint_pos": frame["joint_pos"].astype(np.float32, copy=False),
        "body_pos_w": frame["body_pos_w"].astype(np.float32, copy=False),
        "body_quat_w": frame["body_quat_w"].astype(np.float32, copy=False),
        "joint_vel": np.zeros_like(frame["joint_pos"], dtype=np.float32),
        "body_lin_vel_w": np.zeros_like(frame["body_pos_w"], dtype=np.float32),
        "body_ang_vel_w": np.zeros_like(frame["body_quat_w"][..., :3], dtype=np.float32),
    }


class RealtimeMotionBuffer:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        future_steps: Iterable[int],
        motion_zmq_connect: str | None = None,
        motion_zmq_topic: str = "",
        motion_zmq_hwm: int = 1,
        dt_s: float = 0.02,
        tolerance_s: float = 0.04,
    ):
        self.robot_cfg = robot_cfg
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.joint_names: list[str] = list(self.robot_cfg.joint_names)
        self.body_names: list[str] = list(self.robot_cfg.body_names)
        self.future_steps = np.asarray(list(future_steps), dtype=int)
        if self.future_steps.ndim != 1:
            raise ValueError(f"future_steps must be 1D, got {self.future_steps.shape}")
        self.dt_s = float(dt_s)
        self.tolerance_s = float(tolerance_s)
        self.min_future_step = int(np.min(self.future_steps)) if self.future_steps.size else 0
        self.max_future_step = int(np.max(self.future_steps)) if self.future_steps.size else 0
        self.delay_s = float(self.max_future_step * self.dt_s + self.tolerance_s)

        self._lock = threading.Lock()
        self._timestamps_ns: list[int] = []
        self._frames: list[dict[str, np.ndarray]] = []
        self._zmq_context = zmq.Context.instance()
        self._motion_zmq_connect = motion_zmq_connect
        self._motion_zmq_topic = motion_zmq_topic
        self._motion_zmq_hwm = int(motion_zmq_hwm)
        self._motion_stream_socket: zmq.Socket | None = None
        self._motion_stream_thread: threading.Thread | None = None
        self._motion_stream_stop = threading.Event()
        if self._motion_zmq_connect:
            self._start_motion_stream()

    def _start_motion_stream(self) -> None:
        if self._motion_stream_thread is not None:
            return

        sock = self._zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, self._motion_zmq_hwm)
        if self._motion_zmq_topic:
            sock.setsockopt_string(zmq.SUBSCRIBE, self._motion_zmq_topic)
        else:
            sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self._motion_zmq_connect)
        self._motion_stream_socket = sock

        def _stream_loop() -> None:
            while not self._motion_stream_stop.is_set():
                try:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                except Exception as exc:
                    logger.warning(f"Motion subscriber error: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    self.__append_payload(raw, topic=self._motion_zmq_topic)
                except Exception as exc:
                    logger.warning(f"Failed to decode motion payload: {exc}")

        self._motion_stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._motion_stream_thread.start()

    def __append_payload(
        self,
        payload: dict[str, Any] | str | bytes,
        recv_time_ns: int | None = None,
        topic: str = "",
    ) -> None:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = payload.strip()
            if topic:
                prefix = f"{topic} "
                if payload.startswith(prefix):
                    payload = payload[len(prefix) :]
                elif payload.startswith(topic):
                    payload = payload[len(topic) :].lstrip()
            elif not payload.startswith("{") and " " in payload:
                payload = payload.split(" ", 1)[1].lstrip()
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        # The live publisher is expected to emit joint/body arrays in the canonical
        # G1 order. We keep the order fixed here instead of inferring it from payloads.
        payload_joint_names = payload.get("joint_names")
        if payload_joint_names is not None:
            payload_joint_names = [str(name) for name in payload_joint_names]
            if payload_joint_names != self.joint_names:
                logger.warning(
                    "Live motion payload joint_names do not match RobotCfg canonical order"
                )
        payload_body_names = payload.get("body_names")
        if payload_body_names is not None:
            payload_body_names = [str(name) for name in payload_body_names]
            if payload_body_names != self.body_names:
                logger.warning(
                    "Live motion payload body_names do not match RobotCfg canonical order"
                )

        timestamp_ns = (
            payload.get("smplx_t_ns")
            or recv_time_ns
            or time.time_ns()
        )
        timestamp_ns = int(timestamp_ns)

        joint_pos = payload.get("joint_pos", payload.get("dof_pos", payload.get("qpos", None)))
        if joint_pos is None:
            raise ValueError("Payload missing joint_pos/dof_pos/qpos")
        joint_pos = _ensure_np(joint_pos, 1)
        if joint_pos.shape[0] >= 7 + len(self.joint_names) and payload.get("joint_pos") is None:
            joint_pos = joint_pos[7 : 7 + len(self.joint_names)]
        if joint_pos.shape[0] != len(self.joint_names):
            raise ValueError(
                f"Expected {len(self.joint_names)} joint positions, got {joint_pos.shape[0]}"
            )

        body_pos_w = payload.get("body_pos_w", None)
        body_quat_w = payload.get("body_quat_w", None)

        if body_pos_w is None or body_quat_w is None:
            raise ValueError("Payload missing body_pos_w/body_quat_w")

        body_pos_w = _ensure_np(body_pos_w, 2)
        body_quat_w = _ensure_np(body_quat_w, 2)

        if body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w[..., 3], got {body_pos_w.shape}")
        if body_quat_w.shape[-1] != 4:
            raise ValueError(f"Expected body_quat_w[..., 4], got {body_quat_w.shape}")
        if body_pos_w.shape[-2] != len(self.body_names):
            raise ValueError(
                f"Expected {len(self.body_names)} body positions, got {body_pos_w.shape[-2]}"
            )
        if body_quat_w.shape[-2] != len(self.body_names):
            raise ValueError(
                f"Expected {len(self.body_names)} body quaternions, got {body_quat_w.shape[-2]}"
            )

        frame = {
            "joint_pos": joint_pos.astype(np.float32, copy=True),
            "body_pos_w": body_pos_w.astype(np.float32, copy=True),
            "body_quat_w": body_quat_w.astype(np.float32, copy=True),
        }

        with self._lock:
            insert_idx = bisect_right(self._timestamps_ns, timestamp_ns)
            self._timestamps_ns.insert(insert_idx, timestamp_ns)
            self._frames.insert(insert_idx, frame)

    @property
    def latest_timestamp_ns(self) -> int | None:
        with self._lock:
            return self._timestamps_ns[-1] if self._timestamps_ns else None

    def _sample_frame_locked(self, timestamp_ns: int) -> dict[str, np.ndarray]:
        if not self._timestamps_ns:
            return self._empty_frame()

        if timestamp_ns <= self._timestamps_ns[0]:
            return _frame_with_zero_velocities(self._frames[0])
        if timestamp_ns >= self._timestamps_ns[-1]:
            return _frame_with_zero_velocities(self._frames[-1])

        right = bisect_right(self._timestamps_ns, timestamp_ns)
        left = right - 1
        t0 = self._timestamps_ns[left]
        t1 = self._timestamps_ns[right]
        if t1 == t0:
            return self._frames[left]

        frame0 = self._frames[left]
        frame1 = self._frames[right]
        alpha = float((timestamp_ns - t0) / (t1 - t0))

        out = {
            "joint_pos": (frame0["joint_pos"] * (1.0 - alpha) + frame1["joint_pos"] * alpha).astype(np.float32),
            "body_pos_w": (frame0["body_pos_w"] * (1.0 - alpha) + frame1["body_pos_w"] * alpha).astype(np.float32),
            "body_quat_w": _quat_slerp_batch(frame0["body_quat_w"], frame1["body_quat_w"], alpha),
        }

        out["joint_vel"] = np.zeros_like(frame0["joint_pos"], dtype=np.float32)
        out["body_lin_vel_w"] = np.zeros_like(frame0["body_pos_w"], dtype=np.float32)
        out["body_ang_vel_w"] = np.zeros_like(frame0["body_quat_w"][..., :3], dtype=np.float32)

        return out

    def _empty_frame(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.zeros((len(self.joint_names),), dtype=np.float32),
            "joint_vel": np.zeros((len(self.joint_names),), dtype=np.float32),
            "body_pos_w": np.zeros((len(self.body_names), 3), dtype=np.float32),
            "body_lin_vel_w": np.zeros((len(self.body_names), 3), dtype=np.float32),
            "body_quat_w": np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(self.body_names), 1)
            ),
            "body_ang_vel_w": np.zeros((len(self.body_names), 3), dtype=np.float32),
        }

    def cleanup(self, cutoff_ns: int) -> None:
        with self._lock:
            # Keep one frame before the cutoff so interpolation still has a left endpoint.
            while len(self._timestamps_ns) > 1 and self._timestamps_ns[1] < cutoff_ns:
                self._timestamps_ns.pop(0)
                self._frames.pop(0)

    def get_obs(self) -> MotionData:
        current_time_ns = time.time_ns()
        future_steps = self.future_steps
        retain_cutoff_ns = int(
            current_time_ns - (self.delay_s + abs(self.min_future_step) * self.dt_s) * 1e9
        )
        self.cleanup(retain_cutoff_ns)

        target_base_ns = int(current_time_ns - self.delay_s * 1e9)
        target_times_ns = target_base_ns + np.asarray(future_steps, dtype=np.int64) * int(self.dt_s * 1e9)

        with self._lock:
            frames = [self._sample_frame_locked(int(ts)) for ts in target_times_ns]

        motion_data = MotionData(
            motion_id=np.zeros((1, len(frames)), dtype=np.int64),
            step=future_steps.reshape(1, -1),
            timestamps_ns=target_times_ns.reshape(1, -1),
            joint_pos=np.stack([frame["joint_pos"] for frame in frames], axis=0)[None, ...],
            joint_vel=np.stack([frame["joint_vel"] for frame in frames], axis=0)[None, ...],
            body_pos_w=np.stack([frame["body_pos_w"] for frame in frames], axis=0)[None, ...],
            body_lin_vel_w=np.stack([frame["body_lin_vel_w"] for frame in frames], axis=0)[None, ...],
            body_quat_w=np.stack([frame["body_quat_w"] for frame in frames], axis=0)[None, ...],
            body_ang_vel_w=np.stack([frame["body_ang_vel_w"] for frame in frames], axis=0)[None, ...],
        )
        return motion_data
