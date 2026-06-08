from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)


TRACKING_BODY_PATTERNS = (
    "pelvis",
    "torso_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_toe_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_yaw_link",
)
TERMINATION_ROOT_BODY_NAME = "torso_link"
ANCHOR_BODY_NAME = "pelvis"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute motion progress, global root tracking, and local body tracking "
            "from full trajectory NPZ files saved by the integrated MuJoCo evaluator."
        )
    )
    parser.add_argument("paths", nargs="+", help="Full trajectory .npz files or glob patterns.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern).expanduser()
        expanded = (
            sorted(Path(item) for item in glob.glob(str(path), recursive=True))
            if any(char in pattern for char in "*?[]")
            else [path]
        )
        for item in expanded:
            resolved = item.resolve()
            if resolved.is_file() and resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    if not paths:
        raise FileNotFoundError(f"No trajectory files matched: {patterns}")
    return paths


def _scalar(value: np.ndarray | str | bytes | object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _select_policy_frames(data: dict[str, np.ndarray]) -> np.ndarray:
    motion_t = np.asarray(data["motion_t"], dtype=np.int32)
    if motion_t.size == 0:
        raise ValueError("empty motion_t")
    return np.flatnonzero(np.r_[True, motion_t[1:] != motion_t[:-1]])


def _indices_for_patterns(names: list[str], patterns: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for pattern in patterns:
        for idx, name in enumerate(names):
            if idx in indices:
                continue
            if name == pattern or re.fullmatch(pattern, name):
                indices.append(idx)
    if not indices:
        raise ValueError(f"No body names matched patterns: {patterns}")
    return indices


def _quat_angle_magnitude(quat: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    xyz_norm = np.linalg.norm(quat[..., 1:], axis=-1)
    return 2.0 * np.arctan2(xyz_norm, np.maximum(np.abs(quat[..., 0]), eps))


def _local_tracking_state(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_pos = body_pos_w[:, anchor_idx].copy()
    anchor_pos[:, 2] = 0.0
    anchor_yaw = projected_yaw_quat(body_quat_w[:, anchor_idx])
    anchor_yaw_expanded = np.broadcast_to(
        anchor_yaw[:, None, :],
        body_quat_w.shape,
    )
    body_pos_local = quat_rotate_inverse_numpy(
        anchor_yaw_expanded,
        body_pos_w - anchor_pos[:, None, :],
    )
    body_quat_local = quat_mul(
        quat_conjugate(anchor_yaw_expanded),
        body_quat_w,
    )
    return body_pos_local, body_quat_local


def _first_cumulative_failure(error: np.ndarray, *, threshold: float, min_steps: int) -> int | None:
    count = 0
    for idx, value in enumerate(error):
        if float(value) >= threshold:
            count += 1
        else:
            count = 0
        if count >= min_steps:
            return idx
    return None


def _relative_translation(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        quat[0].reshape(1, 4),
        (pos[-1] - pos[0]).reshape(1, 3),
    )[0]


def _relative_translation_series(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        np.broadcast_to(quat[0].reshape(1, 4), quat.shape),
        pos - pos[0].reshape(1, 3),
    )


def _compute_one(path: Path) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    frame_idx = _select_policy_frames(data)
    names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
    tracking_indices = _indices_for_patterns(names, TRACKING_BODY_PATTERNS)
    root_idx = names.index(TERMINATION_ROOT_BODY_NAME)
    anchor_idx = names.index(ANCHOR_BODY_NAME)

    robot_pos = np.asarray(data["robot_body_pos_w"], dtype=np.float32)[frame_idx]
    robot_quat = np.asarray(data["robot_body_quat_w"], dtype=np.float32)[frame_idx]
    motion_pos = np.asarray(data["motion_body_pos_w"], dtype=np.float32)[frame_idx]
    motion_quat = np.asarray(data["motion_body_quat_w"], dtype=np.float32)[frame_idx]
    motion_t = np.asarray(data["motion_t"], dtype=np.int32)[frame_idx]

    robot_pos_local, robot_quat_local = _local_tracking_state(robot_pos, robot_quat, anchor_idx)
    motion_pos_local, motion_quat_local = _local_tracking_state(motion_pos, motion_quat, anchor_idx)

    root_ori_error = _quat_angle_magnitude(
        quat_mul(quat_conjugate(motion_quat[:, root_idx]), robot_quat[:, root_idx])
    )
    body_pos_error_local = np.linalg.norm(
        motion_pos_local[:, tracking_indices] - robot_pos_local[:, tracking_indices],
        axis=-1,
    )
    body_ori_error_local = _quat_angle_magnitude(
        quat_mul(
            quat_conjugate(motion_quat_local[:, tracking_indices]),
            robot_quat_local[:, tracking_indices],
        )
    )

    failures = {
        "root_ori_error": _first_cumulative_failure(root_ori_error, threshold=1.2, min_steps=25),
        "body_pos_error": _first_cumulative_failure(body_pos_error_local.max(axis=1), threshold=0.4, min_steps=5),
        "body_ori_error": _first_cumulative_failure(body_ori_error_local.max(axis=1), threshold=1.2, min_steps=5),
    }
    valid_failures = {name: idx for name, idx in failures.items() if idx is not None}
    if valid_failures:
        termination_reason, termination_idx = min(valid_failures.items(), key=lambda item: item[1])
        terminated = True
    else:
        termination_reason = "motion_end"
        termination_idx = int(len(motion_t) - 1)
        terminated = False

    pre_end = max(1, termination_idx if terminated else termination_idx + 1)
    motion_length = int(np.asarray(data["motion_length"]).reshape(())) if "motion_length" in data else int(motion_t[-1]) + 1
    motion_denominator = max(1, motion_length - 1)
    progress = min(1.0, max(0.0, float(motion_t[termination_idx]) / float(motion_denominator)))
    local_body_tracking_error = float(np.mean(body_pos_error_local[:pre_end]))

    robot_root_pos = np.asarray(data["robot_root_pos_w"], dtype=np.float32)[frame_idx]
    robot_root_quat = np.asarray(data["robot_root_quat_w"], dtype=np.float32)[frame_idx]
    motion_root_pos = np.asarray(data["motion_root_pos_w"], dtype=np.float32)[frame_idx]
    motion_root_quat = np.asarray(data["motion_root_quat_w"], dtype=np.float32)[frame_idx]
    robot_root_rel = _relative_translation_series(robot_root_pos[:pre_end], robot_root_quat[:pre_end])
    motion_root_rel = _relative_translation_series(motion_root_pos[:pre_end], motion_root_quat[:pre_end])
    root_tracking_error = robot_root_rel - motion_root_rel
    global_root_tracking_error = float(np.mean(np.linalg.norm(root_tracking_error, axis=-1)))
    global_root_tracking_error_xy = float(np.mean(np.linalg.norm(root_tracking_error[:, :2], axis=-1)))
    root_final_error = _relative_translation(robot_root_pos, robot_root_quat) - _relative_translation(
        motion_root_pos,
        motion_root_quat,
    )

    return {
        "path": str(path),
        "policy_config": _scalar(data["policy_config"]) if "policy_config" in data else "",
        "motion_path": _scalar(data["motion_path"]) if "motion_path" in data else "",
        "seed": int(np.asarray(data["seed"]).reshape(())) if "seed" in data else -1,
        "frames": int(len(frame_idx)),
        "motion_start": int(motion_t[0]),
        "motion_end": int(motion_t[-1]),
        "motion_length": motion_length,
        "termination_idx": int(termination_idx),
        "termination_motion_t": int(motion_t[termination_idx]),
        "termination_reason": termination_reason,
        "terminated": int(terminated),
        "progress": progress,
        "global_root_tracking_error": global_root_tracking_error,
        "global_root_tracking_error_xy": global_root_tracking_error_xy,
        "local_body_tracking_error": local_body_tracking_error,
        "mpjpe": local_body_tracking_error,
        "root_final_error_norm": float(np.linalg.norm(root_final_error)),
        "root_final_error_xy_norm": float(np.linalg.norm(root_final_error[:2])),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(rows),
        "progress": _mean_std([float(row["progress"]) for row in rows]),
        "global_root_tracking_error": _mean_std([float(row["global_root_tracking_error"]) for row in rows]),
        "global_root_tracking_error_xy": _mean_std([float(row["global_root_tracking_error_xy"]) for row in rows]),
        "local_body_tracking_error": _mean_std([float(row["local_body_tracking_error"]) for row in rows]),
        "mpjpe": _mean_std([float(row["mpjpe"]) for row in rows]),
        "root_final_error_norm": _mean_std([float(row["root_final_error_norm"]) for row in rows]),
        "root_final_error_xy_norm": _mean_std([float(row["root_final_error_xy_norm"]) for row in rows]),
    }


def main() -> None:
    args = _parse_args()
    rows = [_compute_one(path) for path in _expand_paths(args.paths)]
    summary = _summary(rows)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_config"])].append(row)
    per_policy_config = {
        policy_config: _summary(policy_rows)
        for policy_config, policy_rows in grouped.items()
    }
    payload = {
        "summary": summary,
        "per_policy_config": per_policy_config,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
