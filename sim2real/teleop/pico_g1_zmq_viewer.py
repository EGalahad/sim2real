#!/usr/bin/env python3
"""
ZMQ viewer for retargeted G1 poses.

Subscribe to a live motion stream published by pico_g1_zmq_publisher.py and
render the robot in a MuJoCo viewer window.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from loop_rate_limiters import RateLimiter

from sim2real.utils.robot_defs import (
    G1_BODY_NAMES,
    G1_JOINT_NAMES,
    G1_MJCF_PATH,
    G1_QPOS_SIZE,
    JOINT_POS_SLICE,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
    validate_name_order,
)


DEFAULT_G1_QPOS = np.concatenate(
    [
        np.array([0.0, 0.0, 0.8], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    ]
)


def _payload_to_qpos(payload: dict[str, object]) -> Optional[np.ndarray]:
    joint_names = payload.get("joint_names")
    body_names = payload.get("body_names")
    if joint_names is not None:
        validate_name_order(G1_JOINT_NAMES, joint_names, label="joint_names")
    if body_names is not None:
        validate_name_order(G1_BODY_NAMES, body_names, label="body_names")

    qpos = payload.get("qpos")
    if qpos is not None:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape[0] >= G1_QPOS_SIZE:
            return q[:G1_QPOS_SIZE]

    root_pos = payload.get("root_pos")
    root_quat = payload.get("root_quat")
    joint_pos = payload.get("joint_pos")
    if root_pos is None or root_quat is None or joint_pos is None:
        joint_pos = payload.get("dof_pos")
    if root_pos is None or root_quat is None or joint_pos is None:
        return None

    joint_pos_arr = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    if joint_pos_arr.shape[0] < len(G1_JOINT_NAMES):
        return None

    q = np.zeros(G1_QPOS_SIZE, dtype=np.float32)
    q[ROOT_POS_SLICE] = np.asarray(root_pos, dtype=np.float32).reshape(-1)[:3]
    q[ROOT_QUAT_SLICE] = np.asarray(root_quat, dtype=np.float32).reshape(-1)[:4]
    q[JOINT_POS_SLICE] = joint_pos_arr[: len(G1_JOINT_NAMES)]
    return q


def _parse_payload(raw: str, topic: str) -> Optional[np.ndarray]:
    if topic:
        prefix = f"{topic} "
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
        elif topic != "":
            return None

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return None
    return _payload_to_qpos(payload)


class NativeG1Viewer:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(G1_MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        self.track_body_id = self._resolve_track_body_id()
        if self.track_body_id is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.track_body_id
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10

    def _resolve_track_body_id(self) -> Optional[int]:
        for body_name in ("torso_link", "pelvis"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return int(body_id)
        return None

    def is_running(self) -> bool:
        return bool(self.viewer.is_running())

    def render(self, qpos: np.ndarray) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[: min(self.model.nq, qpos_arr.shape[0])] = qpos_arr[: self.model.nq]
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def close(self) -> None:
        self.viewer.close()


def run_viewer(args: argparse.Namespace) -> None:
    rate = RateLimiter(frequency=float(args.viewer_hz), warn=False)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(args.hwm))
    if hasattr(zmq, "CONFLATE"):
        sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    viewer = NativeG1Viewer()

    latest_qpos = DEFAULT_G1_QPOS.copy()
    last_recv_log = 0.0
    print(f"[viewer] connect={args.connect} topic={args.topic} viewer_hz={args.viewer_hz}")

    try:
        while viewer.is_running():
            try:
                while True:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                    qpos = _parse_payload(raw, args.topic)
                    if qpos is not None:
                        latest_qpos = qpos
                        last_recv_log = time.monotonic()
            except zmq.Again:
                pass
            except json.JSONDecodeError as exc:
                print(f"[viewer] bad JSON payload: {exc}")

            if time.monotonic() - last_recv_log > 2.0:
                last_recv_log = time.monotonic()

            viewer.render(latest_qpos)
            rate.sleep()
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting viewer.")
    finally:
        viewer.close()
        sock.close(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subscribe to G1 motion over ZMQ and visualize it")
    parser.add_argument("--connect", type=str, default="tcp://127.0.0.1:28701")
    parser.add_argument("--topic", type=str, default="g1")
    parser.add_argument("--viewer_hz", type=float, default=30.0)
    parser.add_argument("--hwm", type=int, default=1)
    parser.set_defaults(func=run_viewer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
