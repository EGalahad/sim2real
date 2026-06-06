from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from any4hdmi.core.format import MOTIONS_SUBDIR, MOTION_DTYPE, save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from sim2real.config.robots import RobotCfg


def write_single_motion_qpos_dataset(
    output_dir: str | Path,
    *,
    robot_cfg: RobotCfg,
    qpos: np.ndarray,
    fps: float,
    dataset_name: str,
    source: dict[str, Any],
) -> tuple[Path, Path]:
    """Write one qpos clip as an any4hdmi dataset."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    qpos_arr = np.asarray(qpos, dtype=MOTION_DTYPE)
    if qpos_arr.ndim != 2:
        raise ValueError(f"qpos must have shape (frames, qpos_dim), got {qpos_arr.shape}")
    if qpos_arr.shape[0] <= 0:
        raise ValueError("qpos must contain at least one frame")
    if qpos_arr.shape[1] != robot_cfg.qpos_size:
        raise ValueError(
            f"qpos dimension mismatch: expected {robot_cfg.qpos_size}, got {qpos_arr.shape[1]}"
        )

    output_root = Path(output_dir).expanduser().resolve()
    mjcf_path = robot_cfg.resolve_mjcf_path()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    qpos_names = qpos_names_from_model(model)
    if len(qpos_names) != qpos_arr.shape[1]:
        raise ValueError(
            f"MJCF qpos dimension mismatch: expected {qpos_arr.shape[1]}, got {len(qpos_names)} names"
        )

    motion_path = save_motion(output_root / MOTIONS_SUBDIR / "motion.npz", qpos_arr)
    manifest_path = write_manifest(
        output_root,
        dataset_name=dataset_name,
        mjcf=robot_cfg.mjcf_path if robot_cfg.mjcf_path is not None else mjcf_path,
        timestep=1.0 / float(fps),
        qpos_names=qpos_names,
        num_motions=1,
        total_hours=float(qpos_arr.shape[0]) / float(fps) / 3600.0,
        source=source,
    )
    return motion_path, manifest_path
