from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from sim2real.config.robots.base import (
    BODY_ANG_VEL_W_KEY,
    BODY_LIN_VEL_W_KEY,
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_POS_KEY,
    JOINT_VEL_KEY,
    RobotCfg,
)


LEGACY_MOTION_KEYS = (
    BODY_POS_W_KEY,
    BODY_LIN_VEL_W_KEY,
    BODY_QUAT_W_KEY,
    BODY_ANG_VEL_W_KEY,
    JOINT_POS_KEY,
    JOINT_VEL_KEY,
)


@lru_cache(maxsize=None)
def _resolve_canonical_root_body_index(
    canonical_body_names: tuple[str, ...],
    mjcf_root_body_name: str,
) -> int:
    return canonical_body_names.index(mjcf_root_body_name)


@lru_cache(maxsize=None)
def _resolve_canonical_to_mjcf_joint_indices(
    canonical_joint_names: tuple[str, ...],
    mjcf_joint_names: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(canonical_joint_names.index(name) for name in mjcf_joint_names)


def motion_to_qpos(
    body_pos_w: object,
    body_quat_w: object,
    joint_pos: object,
    robot_cfg: RobotCfg,
    mjcf_root_body_name: str,
    mjcf_joint_names: Sequence[str],
) -> np.ndarray:
    body_pos_arr = np.asarray(body_pos_w, dtype=np.float32)
    body_quat_arr = np.asarray(body_quat_w, dtype=np.float32)
    joint_pos_arr = np.asarray(joint_pos, dtype=np.float32)
    if body_pos_arr.ndim not in (2, 3) or body_pos_arr.shape[-1] != 3:
        raise ValueError(f"{BODY_POS_W_KEY} must have shape (..., num_bodies, 3), got {body_pos_arr.shape}")
    if body_quat_arr.ndim != body_pos_arr.ndim or body_quat_arr.shape[-1] != 4:
        raise ValueError(
            f"{BODY_QUAT_W_KEY} must have shape (..., num_bodies, 4), got {body_quat_arr.shape}"
        )
    if body_pos_arr.shape[:-1] != body_quat_arr.shape[:-1]:
        raise ValueError(
            f"{BODY_POS_W_KEY} and {BODY_QUAT_W_KEY} shape mismatch: "
            f"{body_pos_arr.shape} vs {body_quat_arr.shape}"
        )
    if joint_pos_arr.ndim != body_pos_arr.ndim - 1:
        raise ValueError(
            f"{JOINT_POS_KEY} ndim mismatch: expected {body_pos_arr.ndim - 1}, got {joint_pos_arr.ndim}"
        )
    if joint_pos_arr.shape[:-1] != body_pos_arr.shape[:-2]:
        raise ValueError(
            f"{JOINT_POS_KEY} leading shape mismatch: {joint_pos_arr.shape[:-1]} vs {body_pos_arr.shape[:-2]}"
        )
    if body_pos_arr.shape[-2] != len(robot_cfg.body_names):
        raise ValueError(
            f"{BODY_POS_W_KEY} count mismatch: expected {len(robot_cfg.body_names)}, got {body_pos_arr.shape[-2]}"
        )
    if body_quat_arr.shape[-2] != len(robot_cfg.body_names):
        raise ValueError(
            f"{BODY_QUAT_W_KEY} count mismatch: expected {len(robot_cfg.body_names)}, got {body_quat_arr.shape[-2]}"
        )
    if joint_pos_arr.shape[-1] != len(robot_cfg.joint_names):
        raise ValueError(
            f"{JOINT_POS_KEY} length mismatch: expected {len(robot_cfg.joint_names)}, got {joint_pos_arr.shape[-1]}"
        )

    root_body_index = _resolve_canonical_root_body_index(
        tuple(robot_cfg.body_names),
        str(mjcf_root_body_name),
    )
    canonical_to_mjcf_joint_indices = _resolve_canonical_to_mjcf_joint_indices(
        tuple(robot_cfg.joint_names),
        tuple(str(name) for name in mjcf_joint_names),
    )
    qpos_arr = np.zeros(joint_pos_arr.shape[:-1] + (robot_cfg.qpos_size,), dtype=np.float32)
    qpos_arr[..., robot_cfg.root_pos_slice] = body_pos_arr[..., root_body_index, :]
    qpos_arr[..., robot_cfg.root_quat_slice] = body_quat_arr[..., root_body_index, :]
    qpos_arr[..., robot_cfg.joint_pos_slice] = joint_pos_arr[..., list(canonical_to_mjcf_joint_indices)]
    return qpos_arr


def estimate_fps_from_timestamps_ns(
    timestamps_ns: Sequence[int | None],
    *,
    default_fps: int,
) -> int:
    values = np.asarray([value for value in timestamps_ns if value is not None], dtype=np.int64)
    if values.size < 2:
        return int(default_fps)
    dt_ns = np.diff(values)
    positive = dt_ns[dt_ns > 0]
    if positive.size == 0:
        return int(default_fps)
    fps = int(np.rint(1e9 / float(np.median(positive))))
    return max(1, fps)


def empty_legacy_motion(num_frames: int, num_bodies: int, num_joints: int) -> dict[str, np.ndarray]:
    return {
        BODY_POS_W_KEY: np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        BODY_LIN_VEL_W_KEY: np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        BODY_QUAT_W_KEY: np.zeros((num_frames, num_bodies, 4), dtype=np.float32),
        BODY_ANG_VEL_W_KEY: np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        JOINT_POS_KEY: np.zeros((num_frames, num_joints), dtype=np.float32),
        JOINT_VEL_KEY: np.zeros((num_frames, num_joints), dtype=np.float32),
    }


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] <= 1:
        return out
    out[:-1] = (arr[1:] - arr[:-1]) * np.float32(fps)
    out[-1] = out[-2]
    return out


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float32).copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_angular_velocity_wxyz(quat: np.ndarray, fps: float) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32)
    out = np.zeros(quat_arr.shape[:-1] + (3,), dtype=np.float32)
    if quat_arr.shape[0] <= 1:
        return out

    q1 = quat_arr[:-1]
    q2 = quat_arr[1:]
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True).clip(min=1e-12)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True).clip(min=1e-12)
    q_delta = _quat_mul_wxyz(_quat_conjugate_wxyz(q1), q2)

    negative_w = q_delta[..., 0] < 0.0
    q_delta[negative_w] *= -1.0

    vec = q_delta[..., 1:]
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
    cos_half = np.clip(q_delta[..., 0:1], -1.0, 1.0)
    angle = 2.0 * np.arctan2(sin_half, cos_half)
    axis = np.where(
        sin_half > 1e-12,
        vec / np.clip(sin_half, 1e-12, None),
        np.zeros_like(vec),
    )
    out[:-1] = (axis * angle * fps).astype(np.float32, copy=False)
    out[-1] = out[-2]
    return out


def build_legacy_motion_from_frames(
    frames: Sequence[Mapping[str, object]],
    *,
    fps: int,
    num_bodies: int,
    num_joints: int,
) -> dict[str, np.ndarray]:
    if not frames:
        return empty_legacy_motion(0, num_bodies=num_bodies, num_joints=num_joints)

    body_pos_w = np.stack(
        [np.asarray(frame[BODY_POS_W_KEY], dtype=np.float32) for frame in frames],
        axis=0,
    )
    body_quat_w = np.stack(
        [np.asarray(frame[BODY_QUAT_W_KEY], dtype=np.float32) for frame in frames],
        axis=0,
    )
    joint_pos = np.stack(
        [np.asarray(frame[JOINT_POS_KEY], dtype=np.float32) for frame in frames],
        axis=0,
    )
    if body_pos_w.shape[1] != num_bodies or body_quat_w.shape[1] != num_bodies:
        raise ValueError(
            f"Body count mismatch while building legacy motion: expected {num_bodies}, got {body_pos_w.shape[1]}"
        )
    if joint_pos.shape[1] != num_joints:
        raise ValueError(
            f"Joint count mismatch while building legacy motion: expected {num_joints}, got {joint_pos.shape[1]}"
        )

    return {
        BODY_POS_W_KEY: body_pos_w,
        BODY_LIN_VEL_W_KEY: _finite_difference(body_pos_w, fps=float(fps)),
        BODY_QUAT_W_KEY: body_quat_w,
        BODY_ANG_VEL_W_KEY: quaternion_angular_velocity_wxyz(body_quat_w, fps=float(fps)),
        JOINT_POS_KEY: joint_pos,
        JOINT_VEL_KEY: _finite_difference(joint_pos, fps=float(fps)),
    }


def save_legacy_motion_dataset(
    output_dir: Path,
    *,
    motion: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_path = output_dir / "motion.npz"
    meta_path = output_dir / "meta.json"
    motion_arrays: dict[str, Any] = {key: np.asarray(motion[key]) for key in LEGACY_MOTION_KEYS}
    np.savez_compressed(motion_path, **motion_arrays)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(dict(meta), handle, indent=2)
        handle.write("\n")
    return motion_path, meta_path


def load_legacy_motion_dataset(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(path)
    if path.is_dir():
        motion_path = path / "motion.npz"
        meta_path = path / "meta.json"
    else:
        motion_path = path
        meta_path = path.with_name("meta.json")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(motion_path, allow_pickle=False) as raw:
        motion = {key: np.asarray(raw[key], dtype=np.float32) for key in raw.files}
    return meta, motion
