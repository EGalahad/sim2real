#!/usr/bin/env python3
"""
Subscribe to live G1 motion from ZMQ and save an any4hdmi qpos motion clip.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import tyro
import zmq

from sim2real.config.robots.base import resolve_mjcf_joint_names, resolve_mjcf_root_body_name
from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.motion_legacy import (
    estimate_fps_from_timestamps_ns,
    motion_to_qpos,
)
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset


@dataclass
class RecordArgs:
    """Record retargeted G1 motion from pico_retarget_pub.py."""

    robot: str = "g1"
    connect: str = "tcp://127.0.0.1:28701"
    output_dir: Path | None = None
    fps: int = 0
    default_fps: int = 30
    hwm: int = 1024


def run_record(args: RecordArgs) -> None:
    robot_cfg = get_robot_cfg(args.robot)
    mjcf_path = robot_cfg.resolve_mjcf_path()
    mjcf_root_body_name = resolve_mjcf_root_body_name(mjcf_path)
    mjcf_joint_names = resolve_mjcf_joint_names(mjcf_path)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / f"g1_motion_{timestamp}"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(args.hwm))
    sock.connect(args.connect)
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    frames: list[dict[str, object]] = []
    invalid_frames = 0
    start_monotonic = time.monotonic()

    print(f"[record] connect={args.connect}")
    print(f"[record] output_dir={output_dir}")
    print("Recording G1 motion from ZMQ. Press Ctrl-C to stop.")

    try:
        while True:
            raw = sock.recv_string()
            receive_t_ns = time.time_ns()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                invalid_frames += 1
                print(f"[record] bad JSON payload: {exc}")
                continue
            if not isinstance(payload, dict):
                continue

            try:
                qpos_raw = payload.get("qpos")
                if qpos_raw is not None:
                    qpos = np.asarray(qpos_raw, dtype=np.float32).reshape(-1)
                    if qpos.shape[0] != robot_cfg.qpos_size:
                        raise ValueError(
                            f"qpos length mismatch: expected {robot_cfg.qpos_size}, got {qpos.shape[0]}"
                        )
                else:
                    joint_pos_raw = payload.get("joint_pos", payload.get("dof_pos"))
                    body_pos_w_raw = payload.get("body_pos_w")
                    body_quat_w_raw = payload.get("body_quat_w")
                    if joint_pos_raw is None or body_pos_w_raw is None or body_quat_w_raw is None:
                        qpos = None
                    else:
                        joint_pos = np.asarray(joint_pos_raw, dtype=np.float32).reshape(-1)
                        body_pos_w = np.asarray(body_pos_w_raw, dtype=np.float32)
                        body_quat_w = np.asarray(body_quat_w_raw, dtype=np.float32)
                        if joint_pos.shape[0] != len(robot_cfg.joint_names):
                            raise ValueError(
                                f"joint_pos length mismatch: expected {len(robot_cfg.joint_names)}, got {joint_pos.shape[0]}"
                            )
                        if body_pos_w.shape != (len(robot_cfg.body_names), 3):
                            raise ValueError(
                                f"body_pos_w shape mismatch: expected {(len(robot_cfg.body_names), 3)}, got {body_pos_w.shape}"
                            )
                        if body_quat_w.shape != (len(robot_cfg.body_names), 4):
                            raise ValueError(
                                f"body_quat_w shape mismatch: expected {(len(robot_cfg.body_names), 4)}, got {body_quat_w.shape}"
                            )
                        qpos = motion_to_qpos(
                            body_pos_w,
                            body_quat_w,
                            joint_pos,
                            robot_cfg,
                            mjcf_root_body_name,
                            mjcf_joint_names,
                        )

                if qpos is None:
                    frame = None
                else:
                    if qpos.shape[0] != robot_cfg.qpos_size:
                        raise ValueError(
                            f"qpos length mismatch after conversion: expected {robot_cfg.qpos_size}, got {qpos.shape[0]}"
                        )
                    frame = {
                        "qpos": qpos.copy(),
                        "publish_t_ns": int(payload["publish_t_ns"]) if payload.get("publish_t_ns") is not None else None,
                        "seq": int(payload["seq"]) if payload.get("seq") is not None else None,
                        "smplx_t_ns": int(payload["smplx_t_ns"]) if payload.get("smplx_t_ns") is not None else None,
                    }
            except Exception as exc:
                invalid_frames += 1
                print(f"[record] invalid payload skipped: {exc}")
                continue
            if frame is None:
                invalid_frames += 1
                print("[record] incomplete payload skipped")
                continue

            if frame.get("publish_t_ns") is None:
                frame["publish_t_ns"] = receive_t_ns
            frames.append(frame)

            if len(frames) % 50 == 0:
                elapsed = max(1e-6, time.monotonic() - start_monotonic)
                print(
                    f"[record] frames={len(frames)} invalid={invalid_frames} "
                    f"recv_fps={len(frames) / elapsed:.2f}"
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, saving motion...")
    finally:
        sock.close(0)

    if not frames:
        print("[record] no valid frames received; nothing written")
        return

    fps = int(args.fps)
    if fps <= 0:
        fps = estimate_fps_from_timestamps_ns(
            [
                int(cast(Any, value)) if value is not None else None
                for value in (frame.get("publish_t_ns") for frame in frames)
            ],
            default_fps=int(args.default_fps),
        )

    qpos = np.stack(
        [np.asarray(frame["qpos"], dtype=np.float32) for frame in frames],
        axis=0,
    )
    motion_path, manifest_path = write_single_motion_qpos_dataset(
        output_dir,
        robot_cfg=robot_cfg,
        qpos=qpos,
        fps=float(fps),
        dataset_name=output_dir.name,
        source={
            "recorder": "sim2real.teleop.record_motion",
            "robot": robot_cfg.name,
            "connect": args.connect,
            "fps": int(fps),
            "frame_count": int(len(frames)),
            "qpos_source": "payload.qpos or fallback motion_to_qpos",
        },
    )

    print(f"[record] saved {len(frames)} frames to {motion_path}")
    print(f"[record] wrote manifest to {manifest_path}")
    print(f"[record] root motion alias: {output_dir / 'motion.npz'}")
    print(f"[record] invalid frames: {invalid_frames}")
    print(f"[record] fps: {fps}")


def main() -> None:
    run_record(tyro.cli(RecordArgs))


if __name__ == "__main__":
    main()
