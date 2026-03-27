#!/usr/bin/env python3
"""
Live PICO/XRobot -> G1 ZMQ publisher.

This script reads XR body data from XRobotStreamer, retargets to Unitree G1 with
GMR, forward-kinematics the resulting MuJoCo qpos, and publishes a canonical
motion payload over ZMQ.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import mujoco
import numpy as np
import zmq
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer
from loop_rate_limiters import RateLimiter

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import (
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_POS_KEY,
    PUBLISH_T_NS_KEY,
    SEQ_KEY,
    SMPLX_T_NS_KEY,
)


def _body_pose_dict_from_streamer(streamer: XRobotStreamer) -> tuple[dict[str, list[np.ndarray]], int]:
    body_poses, _body_velocities, _body_accelerations, _imu_timestamps, body_timestamp = (
        streamer.get_raw_body_data()
    )
    if body_poses is None:
        raise RuntimeError("No XR body data available")

    body_pose_dict: dict[str, list[np.ndarray]] = {}
    for i, body_name in enumerate(streamer.body_joint_names):
        pose = np.asarray(body_poses[i], dtype=np.float32).reshape(-1)
        pos = pose[:3].astype(np.float32, copy=False)
        quat = np.asarray([pose[6], pose[3], pose[4], pose[5]], dtype=np.float32)
        body_pose_dict[body_name] = [pos, quat]

    # Keep the same coordinate transform that the streamer uses for its live path.
    body_pose_dict = streamer.coordinate_transform_unity_data(body_pose_dict).copy()
    return body_pose_dict, int(body_timestamp)


class LiveRetargetPublisher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot_cfg = get_robot_cfg(args.robot)
        self.publish_hz = float(args.publish_hz)
        if self.publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")

        self.rate = RateLimiter(frequency=self.publish_hz, warn=False)
        self.streamer = XRobotStreamer()
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=float(args.actual_human_height),
            verbose=bool(args.verbose),
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.robot_cfg.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_indices = self._resolve_joint_qpos_indices()
        self.body_ids = self._resolve_body_ids()

        expected_qpos_size = self.robot_cfg.qpos_size
        if self.model.nq != expected_qpos_size:
            print(
                "[publish] warning: G1 MJCF qpos size mismatch "
                f"(model.nq={self.model.nq}, expected={expected_qpos_size})"
            )

        self.latest_qpos = np.asarray(self.robot_cfg.default_qpos, dtype=np.float32).copy()
        self.last_stream_wait_log_monotonic = 0.0
        self.fixed_min_link_height_offset: Optional[float] = None
        self.min_link_height_offset_samples: list[float] = []

    def _resolve_joint_qpos_indices(self) -> list[int]:
        joint_qpos_indices: list[int] = []
        for joint_name in self.robot_cfg.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Failed to resolve joint name in MJCF: {joint_name}")
            joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
        return joint_qpos_indices

    def _resolve_body_ids(self) -> list[int]:
        body_ids: list[int] = []
        for body_name in self.robot_cfg.body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Failed to resolve body name in MJCF: {body_name}")
            body_ids.append(int(body_id))
        return body_ids

    def _get_current_min_body_z(self) -> Optional[float]:
        body_z = self.retarget.configuration.data.xpos[1:, 2]
        if body_z.size == 0:
            return None
        min_body_z = float(np.min(body_z))
        if not np.isfinite(min_body_z):
            return None
        return min_body_z

    def _apply_min_link_height_offset(self, qpos: np.ndarray) -> np.ndarray:
        strategy = self.args.min_link_height_align_strategy
        if strategy == "none":
            return qpos

        qpos_adj = np.asarray(qpos, dtype=np.float32).copy()
        min_body_z = self._get_current_min_body_z()
        if min_body_z is None:
            return qpos_adj

        offset = float(self.args.min_link_height) - min_body_z
        if strategy == "per_frame":
            qpos_adj[2] += offset
            return qpos_adj

        self.min_link_height_offset_samples.append(offset)
        if (
            self.fixed_min_link_height_offset is None
            and len(self.min_link_height_offset_samples)
            >= int(self.args.min_link_height_bootstrap_frames)
        ):
            self.fixed_min_link_height_offset = float(np.mean(self.min_link_height_offset_samples))
            self.min_link_height_offset_samples.clear()
            print(
                "[Info] fixed min-link-height offset calibrated: "
                f"{self.fixed_min_link_height_offset:.6f} m"
            )

        applied_offset = (
            self.fixed_min_link_height_offset
            if self.fixed_min_link_height_offset is not None
            else float(np.mean(self.min_link_height_offset_samples))
        )
        qpos_adj[2] += applied_offset
        return qpos_adj


    def _extract_joint_pos(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[qpos_index] for qpos_index in self.joint_qpos_indices],
            dtype=np.float32,
        )

    def _extract_body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        body_pos_w = np.asarray([self.data.xpos[body_id] for body_id in self.body_ids], dtype=np.float32)
        body_quat_w = np.asarray([self.data.xquat[body_id] for body_id in self.body_ids], dtype=np.float32)
        return body_pos_w, body_quat_w

    def sample_and_retarget(self) -> Optional[dict[str, object]]:
        try:
            smplx_data, source_smplx_t_ns = _body_pose_dict_from_streamer(self.streamer)
        except RuntimeError:
            now = time.monotonic()
            if now - self.last_stream_wait_log_monotonic > 2.0:
                print("[Info] Waiting for XR body data from PICO...")
                self.last_stream_wait_log_monotonic = now
            return None

        try:
            qpos: np.ndarray = self.retarget.retarget(smplx_data, offset_to_ground=False)
        except Exception as exc:
            print(f"[Warning] retarget failed: {exc}")
            return None

        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        qpos = self._apply_min_link_height_offset(qpos)
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)

        joint_pos = self._extract_joint_pos()
        body_pos_w, body_quat_w = self._extract_body_poses()
        publish_t_ns = int(time.time_ns())

        self.latest_qpos = qpos.copy()
        return {
            # SEQ_KEY: None,
            PUBLISH_T_NS_KEY: publish_t_ns,
            SMPLX_T_NS_KEY: int(source_smplx_t_ns),
            "qpos": qpos.tolist(),
            JOINT_POS_KEY: joint_pos.tolist(),
            BODY_POS_W_KEY: body_pos_w.tolist(),
            BODY_QUAT_W_KEY: body_quat_w.tolist(),
        }


def run_publish(args: argparse.Namespace) -> None:
    worker = LiveRetargetPublisher(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, int(args.hwm))
    if hasattr(zmq, "CONFLATE"):
        sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)

    print(
        f"[publish] bind={args.bind} topic={args.topic} publish_hz={args.publish_hz} "
        f"mjcf={worker.robot_cfg.mjcf_path}"
    )
    if args.startup_sleep_s > 0:
        time.sleep(float(args.startup_sleep_s))

    seq = 0
    try:
        while True:
            payload = worker.sample_and_retarget()
            if payload is not None:
                payload[SEQ_KEY] = seq
                sock.send_string(
                    f"{args.topic} {json.dumps(payload, separators=(',', ':'))}",
                    flags=zmq.NOBLOCK,
                )
                seq += 1
            worker.rate.sleep()
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting publisher.")
    finally:
        sock.close(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget live XR frames and publish canonical G1 motion over ZMQ"
    )
    parser.add_argument("--robot", type=str, default="g1")
    parser.add_argument("--bind", type=str, default="tcp://*:28701")
    parser.add_argument("--topic", type=str, default="g1")
    parser.add_argument("--publish_hz", type=float, default=30.0)
    parser.add_argument("--hwm", type=int, default=1)
    parser.add_argument("--startup_sleep_s", type=float, default=0.5)
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--min_link_height", type=float, default=0.0)
    parser.add_argument(
        "--min_link_height_align_strategy",
        type=str,
        default="boostrap",
        choices=("none", "per_frame", "bootstrap"),
        help="Optional z-offset compensation strategy.",
    )
    parser.add_argument(
        "--min_link_height_bootstrap_frames",
        type=int,
        default=30,
        help="Frames used to calibrate the bootstrap z-offset.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(func=run_publish)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
