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

from sim2real.config.robots import RobotCfg, get_robot_cfg
from sim2real.config.robots.base import validate_name_order
from sim2real.teleop.mujoco_viewer_utils import temp_mjcf_with_floor

GROUND_RGB = (0.6, 0.7, 0.6)


def _payload_to_qpos(payload: dict[str, object], robot_cfg: RobotCfg) -> Optional[np.ndarray]:
    joint_names = payload.get("joint_names")
    body_names = payload.get("body_names")
    if joint_names is not None:
        validate_name_order(robot_cfg.joint_names, joint_names, label="joint_names")
    if body_names is not None:
        validate_name_order(robot_cfg.body_names, body_names, label="body_names")

    qpos = payload.get("qpos")
    if qpos is not None:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape[0] >= robot_cfg.qpos_size:
            return q[: robot_cfg.qpos_size]

    root_pos = payload.get("root_pos")
    root_quat = payload.get("root_quat")
    joint_pos = payload.get("joint_pos")
    if root_pos is None or root_quat is None or joint_pos is None:
        joint_pos = payload.get("dof_pos")
    if root_pos is None or root_quat is None or joint_pos is None:
        return None

    joint_pos_arr = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    if joint_pos_arr.shape[0] < len(robot_cfg.joint_names):
        return None

    q = np.zeros(robot_cfg.qpos_size, dtype=np.float32)
    q[robot_cfg.root_pos_slice] = np.asarray(root_pos, dtype=np.float32).reshape(-1)[:3]
    q[robot_cfg.root_quat_slice] = np.asarray(root_quat, dtype=np.float32).reshape(-1)[:4]
    q[robot_cfg.joint_pos_slice] = joint_pos_arr[: len(robot_cfg.joint_names)]
    return q


def _parse_payload(raw: str, topic: str, robot_cfg: RobotCfg) -> Optional[np.ndarray]:
    if topic:
        prefix = f"{topic} "
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
        elif topic != "":
            return None

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return None
    return _payload_to_qpos(payload, robot_cfg)


class NativeG1Viewer:
    def __init__(self, robot_cfg: RobotCfg) -> None:
        self.robot_cfg = robot_cfg
        if self.robot_cfg.mjcf_path is None:
            raise ValueError(f"Robot '{self.robot_cfg.name}' does not define mjcf_path")
        with temp_mjcf_with_floor(
            self.robot_cfg.mjcf_path,
            ground_rgb=GROUND_RGB,
        ) as viewer_mjcf_path:
            self.model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.track_body_id = self._resolve_track_body_id()
        if self.track_body_id is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.track_body_id
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10

    def _resolve_track_body_id(self) -> Optional[int]:
        for body_name in self.robot_cfg.elastic_band_attach_body_names + self.robot_cfg.viewer_track_body_names:
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
    robot_cfg = get_robot_cfg(args.robot)
    rate = RateLimiter(frequency=float(args.viewer_hz), warn=False)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(args.hwm))
    if hasattr(zmq, "CONFLATE"):
        sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    viewer = NativeG1Viewer(robot_cfg)

    latest_qpos = np.asarray(robot_cfg.default_qpos, dtype=np.float32).copy()
    last_recv_log = 0.0
    print(f"[viewer] connect={args.connect} topic={args.topic} viewer_hz={args.viewer_hz}")

    try:
        while viewer.is_running():
            try:
                while True:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                    qpos = _parse_payload(raw, args.topic, robot_cfg)
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
    parser.add_argument("--robot", type=str, default="g1")
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
