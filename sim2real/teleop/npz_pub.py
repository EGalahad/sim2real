#!/usr/bin/env python3
"""
Replay an any4hdmi/npz motion as a canonical ZMQ motion stream.

This is intentionally compatible with tracking.py --motion-backend zmq and the
normal stream published by pico_retarget_pub.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time

import mujoco
import numpy as np
import tyro
import zmq

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import (
    BODY_NAMES_KEY,
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_NAMES_KEY,
    JOINT_POS_KEY,
    PUBLISH_T_NS_KEY,
    SEQ_KEY,
)
from sim2real.rl_policy.utils.motion import MotionDataset, motion_dataset_first_motion
from sim2real.utils.math import yaw_quat
from sim2real.utils.profiling import ScopedTimer


SEND_TIMER_NAME = "npz_pub.send_payload"
SAMPLE_TIMER_NAME = "npz_pub.sample_motion"


def _array_payload(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float32).tolist()


class KeyboardControls:
    def __init__(self) -> None:
        self._pending_keys: list[str] = []
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        def on_press(keycode: str) -> None:
            with self._lock:
                if keycode in self._pressed:
                    return
                self._pressed.add(keycode)
                self._pending_keys.append(keycode)

        def on_release(keycode: str) -> None:
            with self._lock:
                self._pressed.discard(keycode)

        from sshkeyboard import listen_keyboard

        listen_keyboard(on_press=on_press, on_release=on_release)

    def pop_keys(self) -> list[str]:
        with self._lock:
            keys = list(self._pending_keys)
            self._pending_keys.clear()
        return keys

    def close(self) -> None:
        try:
            from sshkeyboard import stop_listening

            stop_listening()
        except Exception:
            pass


class NpzMotionPublisher:
    def __init__(self, args: "PublisherArgs") -> None:
        self.args = args
        self.robot_cfg = get_robot_cfg(args.robot)
        self.publish_hz = float(args.publish_hz)
        if self.publish_hz <= 0.0:
            raise ValueError("publish_hz must be > 0")

        self.period_s = 1.0 / self.publish_hz
        dataset = MotionDataset.create_from_path(
            args.motion_path,
            self.robot_cfg,
            target_fps=int(round(self.publish_hz)),
            mjcf_path=args.mjcf_path,
        )
        self.motion_dataset = motion_dataset_first_motion(dataset)
        self.motion_length = int(self.motion_dataset.num_steps)
        if self.motion_length <= 0:
            raise ValueError(f"Motion has no frames: {args.motion_path}")

        self.motion_ids = np.array([0], dtype=np.int64)
        self.frame = int(args.start_frame)
        self.frame = min(max(self.frame, 0), self.motion_length - 1)
        self.seq = 0
        self.source = str(args.initial_source)
        self.paused = True
        self.motion_body_indices = self._resolve_motion_body_indices()
        self.root_body_index = tuple(self.robot_cfg.body_names).index("pelvis")

        self.model = mujoco.MjModel.from_xml_path(str(self.robot_cfg.resolve_mjcf_path()))
        self.joint_qpos_indices = self._resolve_joint_qpos_indices()
        self.body_ids = self._resolve_body_ids()
        self.default_qpos = np.asarray(self.robot_cfg.default_qpos, dtype=np.float32).copy()
        (
            self.default_joint_pos,
            self.default_body_pos_w,
            self.default_body_quat_w,
        ) = self._pose_arrays_from_qpos(self.default_qpos)
        self.aligned_default_qpos = self.default_qpos.copy()
        self.aligned_default_joint_pos = self.default_joint_pos.copy()
        self.aligned_default_body_pos_w = self.default_body_pos_w.copy()
        self.aligned_default_body_quat_w = self.default_body_quat_w.copy()
        self.latest_root_pos_w = self.default_body_pos_w[self.root_body_index].copy()
        self.latest_root_quat_w = self.default_body_quat_w[self.root_body_index].copy()

        self.keyboard = KeyboardControls() if args.keyboard else None

    def _resolve_motion_body_indices(self) -> list[int]:
        motion_body_names = list(self.motion_dataset.body_names)
        missing = [name for name in self.robot_cfg.body_names if name not in motion_body_names]
        if missing:
            raise ValueError(f"Motion dataset missing robot bodies: {missing}")
        return [motion_body_names.index(name) for name in self.robot_cfg.body_names]

    def _resolve_joint_qpos_indices(self) -> list[int]:
        indices: list[int] = []
        for joint_name in self.robot_cfg.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Failed to resolve joint name in MJCF: {joint_name}")
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return indices

    def _resolve_body_ids(self) -> list[int]:
        body_ids: list[int] = []
        for body_name in self.robot_cfg.body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Failed to resolve body name in MJCF: {body_name}")
            body_ids.append(int(body_id))
        return body_ids

    def _pose_arrays_from_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.model)
        data.qpos[:] = np.asarray(qpos, dtype=np.float32).reshape(-1)
        mujoco.mj_forward(self.model, data)
        joint_pos = np.asarray(data.qpos[self.joint_qpos_indices], dtype=np.float32)
        body_pos_w = np.asarray(data.xpos[self.body_ids], dtype=np.float32)
        body_quat_w = np.asarray(data.xquat[self.body_ids], dtype=np.float32)
        return joint_pos, body_pos_w, body_quat_w

    def process_controls(self) -> None:
        if self.keyboard is None:
            return
        for key in self.keyboard.pop_keys():
            if key == "x":
                if self.source == "default":
                    self.source = "motion"
                    print("[npz-publish] source=motion, press space to play/pause")
                else:
                    self.source = "default"
                    self.paused = True
                    self._capture_aligned_default_pose()
                    print("[npz-publish] source=default, motion paused")
            elif key == "space":
                if self.source != "motion":
                    print("[npz-publish] source=default; press x before playing motion")
                    continue
                self.paused = not self.paused
                print(f"[npz-publish] motion paused={self.paused}")

    def _next_frame(self) -> int:
        frame = self.frame
        if self.paused:
            return frame
        if self.frame < self.motion_length - 1:
            self.frame += 1
        elif self.args.loop:
            self.frame = 0
        elif self.args.hold_last:
            self.paused = True
            print("[npz-publish] reached end of motion; holding last frame")
        else:
            raise StopIteration
        return frame

    def _base_payload(self, source: str) -> dict[str, object]:
        return {
            "source": source,
            "motion_path": str(self.args.motion_path),
            "paused": bool(self.paused or source == "default"),
            PUBLISH_T_NS_KEY: int(time.time_ns()),
            SEQ_KEY: int(self.seq),
        }

    def _capture_aligned_default_pose(self) -> None:
        aligned_qpos = self.default_qpos.copy()
        aligned_qpos[self.robot_cfg.root_pos_slice.start : self.robot_cfg.root_pos_slice.stop] = (
            self.default_qpos[self.robot_cfg.root_pos_slice]
        )
        aligned_qpos[0:2] = np.asarray(self.latest_root_pos_w[:2], dtype=np.float32)
        aligned_qpos[self.robot_cfg.root_quat_slice] = yaw_quat(
            np.asarray(self.latest_root_quat_w, dtype=np.float32)
        ).astype(np.float32, copy=False)
        self.aligned_default_qpos = aligned_qpos
        (
            self.aligned_default_joint_pos,
            self.aligned_default_body_pos_w,
            self.aligned_default_body_quat_w,
        ) = self._pose_arrays_from_qpos(self.aligned_default_qpos)

    def _default_payload(self) -> dict[str, object]:
        payload = self._base_payload("default")
        payload.update(
            {
                "frame": -1,
                JOINT_NAMES_KEY: list(self.robot_cfg.joint_names),
                BODY_NAMES_KEY: list(self.robot_cfg.body_names),
                JOINT_POS_KEY: _array_payload(self.aligned_default_joint_pos),
                BODY_POS_W_KEY: _array_payload(self.aligned_default_body_pos_w),
                BODY_QUAT_W_KEY: _array_payload(self.aligned_default_body_quat_w),
                "qpos": _array_payload(self.aligned_default_qpos),
            }
        )
        if self.args.pub_vel:
            payload.update(
                {
                    "joint_vel": _array_payload(np.zeros_like(self.aligned_default_joint_pos)),
                    "body_lin_vel_w": _array_payload(np.zeros_like(self.aligned_default_body_pos_w)),
                    "body_ang_vel_w": _array_payload(
                        np.zeros((len(self.robot_cfg.body_names), 3), dtype=np.float32)
                    ),
                }
            )
        self.seq += 1
        return payload

    def sample_payload(self) -> dict[str, object]:
        with ScopedTimer(SAMPLE_TIMER_NAME):
            self.process_controls()
            if self.source == "default":
                return self._default_payload()

            frame = self._next_frame()
            motion = self.motion_dataset.get_slice(
                self.motion_ids,
                np.array([frame], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )

            payload = self._base_payload("npz")
            body_pos_w = motion.body_pos_w[0, 0, self.motion_body_indices]
            body_quat_w = motion.body_quat_w[0, 0, self.motion_body_indices]
            self.latest_root_pos_w = np.asarray(body_pos_w[self.root_body_index], dtype=np.float32).copy()
            self.latest_root_quat_w = np.asarray(body_quat_w[self.root_body_index], dtype=np.float32).copy()
            payload.update(
                {
                    "frame": int(frame),
                    JOINT_NAMES_KEY: list(self.motion_dataset.joint_names),
                    BODY_NAMES_KEY: list(self.robot_cfg.body_names),
                    JOINT_POS_KEY: _array_payload(motion.joint_pos[0, 0]),
                    BODY_POS_W_KEY: _array_payload(body_pos_w),
                    BODY_QUAT_W_KEY: _array_payload(body_quat_w),
                }
            )
            if self.args.pub_vel:
                payload.update(
                    {
                        "joint_vel": _array_payload(motion.joint_vel[0, 0]),
                        "body_lin_vel_w": _array_payload(
                            motion.body_lin_vel_w[0, 0, self.motion_body_indices]
                        ),
                        "body_ang_vel_w": _array_payload(
                            motion.body_ang_vel_w[0, 0, self.motion_body_indices]
                        ),
                    }
                )
            self.seq += 1
            return payload

    def close(self) -> None:
        if self.keyboard is not None:
            self.keyboard.close()


@dataclass
class PublisherArgs:
    """Publish an npz motion as canonical motion JSON over ZMQ."""

    motion_path: str
    robot: str = "g1"
    bind: str = "tcp://*:28701"
    publish_hz: float = 50.0
    hwm: int = 1
    startup_sleep_s: float = 0.5
    start_frame: int = 0
    loop: bool = False
    hold_last: bool = True
    pub_vel: bool = False
    mjcf_path: str | None = None
    initial_source: str = "default"
    keyboard: bool = True


def run_publish(args: PublisherArgs) -> None:
    worker = NpzMotionPublisher(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, int(args.hwm))
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)

    print(
        f"[npz-publish] bind={args.bind} publish_hz={args.publish_hz} "
        f"motion_path={args.motion_path} frames={worker.motion_length} "
        f"loop={args.loop} hold_last={args.hold_last} "
        f"pub_vel={args.pub_vel} "
        f"initial_source={args.initial_source}"
    )
    if args.keyboard:
        print("[npz-publish] keys: x toggles default/motion, space plays/pauses motion")
    if args.startup_sleep_s > 0:
        time.sleep(float(args.startup_sleep_s))

    try:
        next_tick = time.perf_counter()
        while True:
            payload = worker.sample_payload()
            with ScopedTimer(SEND_TIMER_NAME):
                sock.send_string(
                    json.dumps(payload, separators=(",", ":")),
                    flags=zmq.NOBLOCK,
                )
            next_tick += worker.period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
    except StopIteration:
        print("[npz-publish] reached end of motion.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting npz publisher.")
    finally:
        worker.close()
        sock.close(0)


if __name__ == "__main__":
    run_publish(tyro.cli(PublisherArgs))
