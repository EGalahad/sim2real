#!/usr/bin/env python3
"""
Record live XRobot frames until Ctrl-C and save them as a self-describing npz.

The primary payload is a body pose sequence with:
  - body_joint_names: ordered joint names
  - body_pos:         [T, J, 3] positions in meters
  - body_rot_wxyz:    [T, J, 4] quaternions in wxyz order

For convenience/debugging, the raw per-frame tuples returned by
XRobotStreamer.get_current_frame() are also stored as an object array.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

RateLimiter = None
XRobotStreamer = None


def _load_runtime_dependencies() -> None:
    global RateLimiter, XRobotStreamer

    from loop_rate_limiters import RateLimiter as _RateLimiter

    try:
        from general_motion_retargeting import XRobotStreamer as _XRobotStreamer
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'general_motion_retargeting'. Install GMR and its "
            "XRoboToolkit dependencies in the active uv environment."
        ) from exc

    if _XRobotStreamer is None:
        raise ImportError(
            "general_motion_retargeting imported, but XRobotStreamer is unavailable."
        )

    RateLimiter = _RateLimiter
    XRobotStreamer = _XRobotStreamer


def _frame_to_arrays(
    body_pose_dict: dict[str, list[Any]], joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    body_pos = np.zeros((len(joint_names), 3), dtype=np.float32)
    body_rot = np.zeros((len(joint_names), 4), dtype=np.float32)

    for joint_idx, joint_name in enumerate(joint_names):
        if joint_name not in body_pose_dict:
            raise KeyError(f"Missing joint '{joint_name}' in XRobot frame")
        pos, rot = body_pose_dict[joint_name]
        body_pos[joint_idx] = np.asarray(pos, dtype=np.float32).reshape(3)
        body_rot[joint_idx] = np.asarray(rot, dtype=np.float32).reshape(4)

    return body_pos, body_rot


def _empty_recording(joint_names: list[str]) -> dict[str, np.ndarray]:
    return {
        "body_pos": np.empty((0, len(joint_names), 3), dtype=np.float32),
        "body_rot_wxyz": np.empty((0, len(joint_names), 4), dtype=np.float32),
        "capture_time_ns": np.empty((0,), dtype=np.int64),
        "frame_valid": np.empty((0,), dtype=bool),
        "frames": np.empty((0,), dtype=object),
    }


def _save_recording(
    output_path: Path,
    joint_names: list[str],
    sample_fps: int,
    actual_human_height: float,
    frames: list[dict[str, Any]],
    body_pos_list: list[np.ndarray],
    body_rot_list: list[np.ndarray],
    capture_time_list: list[int],
) -> None:
    if frames:
        body_pos = np.stack(body_pos_list, axis=0).astype(np.float32, copy=False)
        body_rot = np.stack(body_rot_list, axis=0).astype(np.float32, copy=False)
        capture_time_ns = np.asarray(capture_time_list, dtype=np.int64)
        frame_valid = np.ones((len(frames),), dtype=bool)
        raw_frames = np.asarray(frames, dtype=object)
    else:
        empty = _empty_recording(joint_names)
        body_pos = empty["body_pos"]
        body_rot = empty["body_rot_wxyz"]
        capture_time_ns = empty["capture_time_ns"]
        frame_valid = empty["frame_valid"]
        raw_frames = empty["frames"]

    metadata = {
        "schema": "xrobot_body_pose_v1",
        "source": "XRobotStreamer.get_current_frame",
        "rotation_order": "wxyz",
        "position_unit": "m",
        "sample_fps": int(sample_fps),
        "actual_human_height": float(actual_human_height),
        "frame_count": int(len(frames)),
        "joint_names": joint_names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata)),
        body_joint_names=np.asarray(joint_names, dtype=np.str_),
        body_pos=body_pos,
        body_rot_wxyz=body_rot,
        capture_time_ns=capture_time_ns,
        frame_valid=frame_valid,
        frames=raw_frames,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record live XRobot body poses to npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output npz file. Defaults to a timestamped file in the current directory.",
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=30,
        help="Sampling frequency used while polling XRobotStreamer.",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.6,
        help="Stored in metadata for downstream benchmark consistency.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime_dependencies()

    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path.cwd() / f"xrobot_smplx_{timestamp}.npz"

    streamer = XRobotStreamer()
    rate = RateLimiter(frequency=int(args.sample_fps), warn=False)
    joint_names = list(streamer.body_joint_names)

    frames: list[dict[str, Any]] = []
    body_pos_list: list[np.ndarray] = []
    body_rot_list: list[np.ndarray] = []
    capture_time_list: list[int] = []
    skipped_frames = 0
    last_wait_log = 0.0

    print("Recording live XRobot frames. Press Ctrl-C to stop.")
    print(f"  output: {output_path}")
    print(f"  sample_fps: {int(args.sample_fps)}")
    print(f"  joints: {len(joint_names)}")

    try:
        while True:
            frame = streamer.get_current_frame()
            body_pose_dict = frame[0]
            if body_pose_dict is None:
                skipped_frames += 1
                now = time.monotonic()
                if now - last_wait_log > 2.0:
                    print("[Info] Waiting for XR body data from PICO/XRobot...")
                    last_wait_log = now
                rate.sleep()
                continue

            try:
                body_pos, body_rot = _frame_to_arrays(body_pose_dict, joint_names)
            except Exception as exc:
                skipped_frames += 1
                print(f"[Warning] Skipping malformed frame: {exc}")
                rate.sleep()
                continue

            capture_time_ns = time.time_ns()
            frames.append(
                {
                    "capture_time_ns": capture_time_ns,
                    "body_pose_dict": body_pose_dict,
                    "left_hand_data": frame[1],
                    "right_hand_data": frame[2],
                    "controller_data": frame[3],
                    "headset_pose": frame[4],
                }
            )
            body_pos_list.append(body_pos)
            body_rot_list.append(body_rot)
            capture_time_list.append(capture_time_ns)

            if len(frames) % 30 == 0:
                print(f"[Info] recorded {len(frames)} frames")

            rate.sleep()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, saving recording...")
    finally:
        _save_recording(
            output_path=output_path,
            joint_names=joint_names,
            sample_fps=int(args.sample_fps),
            actual_human_height=float(args.actual_human_height),
            frames=frames,
            body_pos_list=body_pos_list,
            body_rot_list=body_rot_list,
            capture_time_list=capture_time_list,
        )

    print(f"[Done] saved {len(frames)} valid frames to: {output_path}")
    print(f"[Done] skipped frames: {skipped_frames}")


if __name__ == "__main__":
    main()
