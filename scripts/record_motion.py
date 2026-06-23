#!/usr/bin/env python3
"""
Subscribe to live G1 motion from ZMQ and save a 50 Hz any4hdmi qpos motion clip.
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

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import (
    PICO_RECV_TIME_NS_KEY,
    PUBLISH_T_NS_KEY,
    SEQ_KEY,
    SMPLX_T_NS_KEY,
    resolve_mjcf_joint_names,
    resolve_mjcf_root_body_name,
)
from sim2real.teleop.motion_legacy import motion_to_qpos


RECORD_T_NS_KEY = "record_t_ns"


@dataclass
class RecordArgs:
    """Record retargeted G1 motion and resample it to the requested output fps."""

    robot: str = "g1"
    connect: str = "tcp://127.0.0.1:28701"
    output_dir: Path | None = None
    fps: int = 50
    hwm: int = 1024


def _normalize_quat_batch(quat_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return quat_wxyz / denom


def _quat_slerp_batch(
    q0_wxyz: np.ndarray,
    q1_wxyz: np.ndarray,
    alpha: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    q0 = _normalize_quat_batch(np.asarray(q0_wxyz, dtype=np.float32), eps=eps)
    q1 = _normalize_quat_batch(np.asarray(q1_wxyz, dtype=np.float32), eps=eps)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.where(flip_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    alpha_arr = np.asarray(alpha, dtype=np.float32)
    while alpha_arr.ndim < dot.ndim:
        alpha_arr = np.expand_dims(alpha_arr, axis=-1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha_arr

    safe_denom = np.where(sin_theta_0 > eps, sin_theta_0, 1.0)
    s0 = np.sin(theta_0 - theta) / safe_denom
    s1 = np.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha_arr) * q0 + alpha_arr * q1
    out = np.where(dot > 0.9995, nlerp_out, slerp_out)
    return _normalize_quat_batch(out, eps=eps).astype(np.float32, copy=False)


def _frame_timestamp_ns(
    payload: dict[str, object],
    *,
    receive_t_ns: int,
) -> int:
    for key in (PICO_RECV_TIME_NS_KEY, PUBLISH_T_NS_KEY):
        value = payload.get(key)
        if value is not None:
            return int(cast(Any, value))
    return int(receive_t_ns)


def _dedupe_sorted_samples(
    timestamps_ns: np.ndarray,
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(timestamps_ns, kind="stable")
    sorted_times = timestamps_ns[order]
    sorted_qpos = qpos[order]

    unique_times: list[int] = []
    unique_qpos: list[np.ndarray] = []
    duplicate_count = 0
    for timestamp_ns, qpos_frame in zip(sorted_times, sorted_qpos, strict=True):
        timestamp = int(timestamp_ns)
        if unique_times and timestamp == unique_times[-1]:
            unique_qpos[-1] = qpos_frame
            duplicate_count += 1
            continue
        unique_times.append(timestamp)
        unique_qpos.append(qpos_frame)

    return (
        np.asarray(unique_times, dtype=np.int64),
        np.stack(unique_qpos, axis=0).astype(np.float32, copy=False),
        duplicate_count,
    )


def resample_qpos_to_fps(
    *,
    timestamps_ns: np.ndarray,
    qpos: np.ndarray,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if timestamps_ns.ndim != 1:
        raise ValueError(f"timestamps_ns must be 1D, got {timestamps_ns.shape}")
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D, got {qpos.shape}")
    if timestamps_ns.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"timestamp/qpos length mismatch: {timestamps_ns.shape[0]} vs {qpos.shape[0]}"
        )
    if timestamps_ns.shape[0] == 0:
        raise ValueError("cannot resample empty motion")

    source_times_ns, source_qpos, duplicate_count = _dedupe_sorted_samples(timestamps_ns, qpos)
    if source_times_ns.shape[0] == 1:
        return source_qpos.copy(), source_times_ns.copy(), duplicate_count

    start_ns = int(source_times_ns[0])
    end_ns = int(source_times_ns[-1])
    tick_ns = int(round(1e9 / float(target_fps)))
    if tick_ns <= 0:
        raise ValueError(f"invalid target_fps={target_fps}")

    target_count = max(1, int((end_ns - start_ns) // tick_ns) + 1)
    target_times_ns = start_ns + np.arange(target_count, dtype=np.int64) * tick_ns

    right = np.searchsorted(source_times_ns, target_times_ns, side="right")
    right = np.clip(right, 1, source_times_ns.shape[0] - 1)
    left = right - 1
    t0 = source_times_ns[left]
    t1 = source_times_ns[right]
    alpha = np.divide(
        target_times_ns - t0,
        t1 - t0,
        out=np.zeros(target_times_ns.shape, dtype=np.float32),
        where=t1 > t0,
    ).astype(np.float32, copy=False)

    qpos_left = source_qpos[left]
    qpos_right = source_qpos[right]
    resampled_qpos = qpos_left + alpha[:, None] * (qpos_right - qpos_left)
    if resampled_qpos.shape[1] >= 7:
        resampled_qpos[:, 3:7] = _quat_slerp_batch(
            qpos_left[:, 3:7],
            qpos_right[:, 3:7],
            alpha,
        )
    return resampled_qpos.astype(np.float32, copy=False), target_times_ns, duplicate_count


def _write_recording_times(
    output_dir: Path,
    *,
    raw_timestamps_ns: np.ndarray,
    resampled_timestamps_ns: np.ndarray,
    target_fps: int,
) -> Path:
    timestamps_path = output_dir.expanduser().resolve() / "recording_times.npz"
    np.savez_compressed(
        timestamps_path,
        raw_timestamps_ns=raw_timestamps_ns.astype(np.int64, copy=False),
        resampled_timestamps_ns=resampled_timestamps_ns.astype(np.int64, copy=False),
        target_fps=np.asarray(target_fps, dtype=np.int32),
    )
    return timestamps_path


def run_record(args: RecordArgs) -> None:
    from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset

    robot_cfg = get_robot_cfg(args.robot)
    mjcf_path = robot_cfg.resolve_mjcf_path()
    mjcf_root_body_name = resolve_mjcf_root_body_name(mjcf_path)
    mjcf_joint_names = resolve_mjcf_joint_names(mjcf_path)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / f"g1_motion_{timestamp}"

    target_fps = int(args.fps)
    if target_fps <= 0:
        raise ValueError(f"fps must be a positive output resampling rate, got {target_fps}")

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
    print(f"[record] target_fps={target_fps}")
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
                    frame_t_ns = _frame_timestamp_ns(payload, receive_t_ns=receive_t_ns)
                    frame = {
                        "qpos": qpos.copy(),
                        RECORD_T_NS_KEY: int(frame_t_ns),
                        PICO_RECV_TIME_NS_KEY: int(payload[PICO_RECV_TIME_NS_KEY])
                        if payload.get(PICO_RECV_TIME_NS_KEY) is not None
                        else None,
                        PUBLISH_T_NS_KEY: int(payload[PUBLISH_T_NS_KEY])
                        if payload.get(PUBLISH_T_NS_KEY) is not None
                        else None,
                        SEQ_KEY: int(payload[SEQ_KEY]) if payload.get(SEQ_KEY) is not None else None,
                        SMPLX_T_NS_KEY: int(payload[SMPLX_T_NS_KEY])
                        if payload.get(SMPLX_T_NS_KEY) is not None
                        else None,
                    }
            except Exception as exc:
                invalid_frames += 1
                print(f"[record] invalid payload skipped: {exc}")
                continue
            if frame is None:
                invalid_frames += 1
                print("[record] incomplete payload skipped")
                continue

            if frame.get(PICO_RECV_TIME_NS_KEY) is None:
                frame[PICO_RECV_TIME_NS_KEY] = receive_t_ns
            if frame.get(PUBLISH_T_NS_KEY) is None:
                frame[PUBLISH_T_NS_KEY] = receive_t_ns
            frames.append(frame)

            if len(frames) % 50 == 0:
                elapsed = max(1e-6, time.monotonic() - start_monotonic)
                print(
                    f"[record] frames={len(frames)} invalid={invalid_frames} "
                    f"recv_fps={len(frames) / elapsed:.2f}"
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, resampling and saving motion...")
    finally:
        sock.close(0)

    if not frames:
        print("[record] no valid frames received; nothing written")
        return

    raw_timestamps_ns = np.asarray(
        [int(cast(Any, frame[RECORD_T_NS_KEY])) for frame in frames],
        dtype=np.int64,
    )
    raw_qpos = np.stack(
        [np.asarray(frame["qpos"], dtype=np.float32) for frame in frames],
        axis=0,
    )
    qpos, resampled_timestamps_ns, duplicate_timestamp_count = resample_qpos_to_fps(
        timestamps_ns=raw_timestamps_ns,
        qpos=raw_qpos,
        target_fps=target_fps,
    )

    motion_path, manifest_path = write_single_motion_qpos_dataset(
        output_dir,
        robot_cfg=robot_cfg,
        qpos=qpos,
        fps=float(target_fps),
        dataset_name=output_dir.name,
        source={
            "recorder": "scripts/record_motion.py",
            "robot": robot_cfg.name,
            "connect": args.connect,
            "target_fps": int(target_fps),
            "fps": int(target_fps),
            "raw_frame_count": int(len(frames)),
            "resampled_frame_count": int(qpos.shape[0]),
            "duplicate_timestamp_count": int(duplicate_timestamp_count),
            "timestamp_source_order": [
                PICO_RECV_TIME_NS_KEY,
                PUBLISH_T_NS_KEY,
                "receive_t_ns",
            ],
            "raw_start_t_ns": int(raw_timestamps_ns[0]),
            "raw_end_t_ns": int(raw_timestamps_ns[-1]),
            "resampled_start_t_ns": int(resampled_timestamps_ns[0]),
            "resampled_end_t_ns": int(resampled_timestamps_ns[-1]),
            "qpos_source": "payload.qpos or fallback motion_to_qpos",
        },
    )
    timestamps_path = _write_recording_times(
        output_dir,
        raw_timestamps_ns=raw_timestamps_ns,
        resampled_timestamps_ns=resampled_timestamps_ns,
        target_fps=target_fps,
    )

    print(f"[record] saved {qpos.shape[0]} resampled frames to {motion_path}")
    print(f"[record] wrote manifest to {manifest_path}")
    print(f"[record] wrote timestamps to {timestamps_path}")
    print(f"[record] raw frames: {len(frames)}")
    print(f"[record] invalid frames: {invalid_frames}")
    print(f"[record] saved fps: {target_fps}")


def main() -> None:
    run_record(tyro.cli(RecordArgs))


if __name__ == "__main__":
    main()
