#!/usr/bin/env python3
"""Replay an official SONIC SMPL PKL and paired G1 motion over ZMQ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
import tyro
import zmq

from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.smpl_stream import (
    axis_angle_to_quat_wxyz,
    json_safe_payload,
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
)
from sim2real.utils.math import quat_rotate_inverse_numpy


G1_MUJOCO_TO_ISAACLAB_DOF = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
     17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def _resample(values: np.ndarray, source_fps: float, target_fps: float, length: int) -> np.ndarray:
    source_times = np.arange(values.shape[0], dtype=np.float64) / source_fps
    target_times = np.arange(length, dtype=np.float64) / target_fps
    return np.stack(
        [np.interp(target_times, source_times, values[:, i]) for i in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def load_official_walk(
    smpl_path: Path,
    robot_path: Path,
) -> dict[str, np.ndarray | float]:
    smpl = joblib.load(smpl_path)
    robot_container = joblib.load(robot_path)
    robot = next(iter(robot_container.values()))

    pose_aa = np.asarray(smpl["pose_aa"], dtype=np.float32)
    smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
    fps = float(smpl["fps"])
    if pose_aa.shape != (smpl_joints.shape[0], 72) or smpl_joints.shape[1:] != (24, 3):
        raise ValueError(
            f"Unexpected SMPL arrays: pose_aa={pose_aa.shape}, smpl_joints={smpl_joints.shape}"
        )

    root_quat_y_up = axis_angle_to_quat_wxyz(pose_aa[:, :3])
    root_quat_w = remove_smpl_base_rot(smpl_root_ytoz_up(root_quat_y_up))
    joint_root_quat = np.broadcast_to(root_quat_w[:, None, :], (pose_aa.shape[0], 24, 4))
    smpl_joint_pos_root = quat_rotate_inverse_numpy(joint_root_quat, smpl_joints)

    robot_dof_mujoco = _resample(
        np.asarray(robot["dof"], dtype=np.float32),
        float(robot["fps"]),
        fps,
        pose_aa.shape[0],
    )
    robot_dof_isaaclab = robot_dof_mujoco[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    joint_vel = np.gradient(robot_dof_isaaclab, 1.0 / fps, axis=0).astype(np.float32)
    return {
        "fps": fps,
        "smpl_body_pose_aa": pose_aa[:, 3:66].reshape(-1, 21, 3),
        "smpl_joint_pos_root": smpl_joint_pos_root.astype(np.float32),
        "smpl_root_quat_w": root_quat_w.astype(np.float32),
        "joint_pos": robot_dof_isaaclab,
        "joint_vel": joint_vel,
    }


@dataclass
class Args:
    smpl_path: Path
    robot_path: Path
    bind: str = "tcp://127.0.0.1:28702"
    future_frames: int = 4
    max_frames: int | None = None
    loop: bool = False


def main() -> None:
    args = tyro.cli(Args)
    motion = load_official_walk(args.smpl_path, args.robot_path)
    fps = float(motion["fps"])
    period_s = 1.0 / fps
    num_frames = int(np.asarray(motion["joint_pos"]).shape[0])
    if args.max_frames is not None:
        num_frames = min(num_frames, int(args.max_frames))
    if num_frames <= 0 or args.future_frames <= 0:
        raise ValueError("max_frames and future_frames must be positive")

    joint_names = list(get_robot_cfg("g1").joint_names)
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(args.bind)
    time.sleep(0.2)
    print(f"[smpl-pkl-pub] {num_frames} frames at {fps:g} Hz -> {args.bind}")

    frame = 0
    deadline = time.monotonic()
    try:
        while True:
            indices = np.minimum(
                np.arange(frame, frame + args.future_frames, dtype=np.int64),
                num_frames - 1,
            )
            now_ns = time.time_ns()
            fields = {
                key: np.asarray(value)[indices]
                for key, value in motion.items()
                if key != "fps"
            }
            fields["frame_index"] = indices
            fields["publish_t_ns"] = np.asarray(
                [now_ns + (args.future_frames - 1) * int(round(period_s * 1e9))],
                dtype=np.int64,
            )
            fields["motion_first_frame"] = np.asarray([frame == 0], dtype=np.bool_)
            socket.send_json(
                {
                    "topic": "pose",
                    "version": 3,
                    "joint_names": joint_names,
                    **json_safe_payload(fields),
                }
            )

            frame += 1
            if frame >= num_frames:
                if not args.loop:
                    break
                frame = 0
            deadline += period_s
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        socket.close(linger=0)


if __name__ == "__main__":
    main()
