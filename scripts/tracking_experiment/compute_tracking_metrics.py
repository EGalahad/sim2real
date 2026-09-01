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
    quat_rotate_numpy,
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
ANCHOR_BODY_NAME = "pelvis"
RETURN_ANCHOR_BODY_NAME = "torso_link"
TERMINATION_ANCHOR_BODY_NAME = "pelvis"
RETURN_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
TERMINATION_PELVIS_HEIGHT_THRESHOLD_M = 0.3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute progress, root/body tracking, and normalized tracking return "
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


def _normalized_tracking_return(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    motion_pos: np.ndarray,
    motion_quat: np.ndarray,
    names: list[str],
    motion_t: np.ndarray,
    motion_steps: int,
) -> tuple[float, float, bool, str, int, int]:
    reward_anchor_idx = names.index(RETURN_ANCHOR_BODY_NAME)
    termination_anchor_idx = names.index(TERMINATION_ANCHOR_BODY_NAME)
    body_indices = [names.index(name) for name in RETURN_BODY_NAMES]
    delta_pos_w = robot_pos[:, reward_anchor_idx].copy()
    delta_pos_w[:, 2] = motion_pos[:, reward_anchor_idx, 2]
    delta_yaw_w = projected_yaw_quat(
        quat_mul(
            robot_quat[:, reward_anchor_idx],
            quat_conjugate(motion_quat[:, reward_anchor_idx]),
        )
    )
    delta_yaw_expanded = np.broadcast_to(delta_yaw_w[:, None, :], motion_quat.shape)
    motion_pos_relative_w = delta_pos_w[:, None, :] + quat_rotate_numpy(
        delta_yaw_expanded,
        motion_pos - motion_pos[:, reward_anchor_idx, None, :],
    )
    motion_quat_relative_w = quat_mul(delta_yaw_expanded, motion_quat)

    pos_error_sq = np.mean(
        np.sum(
            np.square(
                motion_pos_relative_w[:, body_indices] - robot_pos[:, body_indices]
            ),
            axis=-1,
        ),
        axis=-1,
    )
    ori_error_sq = np.mean(
        np.square(
            _quat_angle_magnitude(
                quat_mul(
                    quat_conjugate(motion_quat_relative_w[:, body_indices]),
                    robot_quat[:, body_indices],
                )
            )
        ),
        axis=-1,
    )
    reward = np.exp(-pos_error_sq / 0.3**2) + np.exp(-ori_error_sq / 0.4**2)

    anchor_pos_failed = (
        np.abs(
            motion_pos[:, termination_anchor_idx, 2]
            - robot_pos[:, termination_anchor_idx, 2]
        )
        > TERMINATION_PELVIS_HEIGHT_THRESHOLD_M
    )
    gravity_w = np.broadcast_to(
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        (len(motion_t), 3),
    )
    motion_gravity_b = quat_rotate_inverse_numpy(
        motion_quat[:, termination_anchor_idx], gravity_w
    )
    robot_gravity_b = quat_rotate_inverse_numpy(
        robot_quat[:, termination_anchor_idx], gravity_w
    )
    anchor_ori_failed = (
        np.abs(motion_gravity_b[:, 2] - robot_gravity_b[:, 2]) > 0.8
    )
    failed = anchor_pos_failed | anchor_ori_failed
    failure_indices = np.flatnonzero(failed)
    if failure_indices.size:
        end = int(failure_indices[0])
        reason = "anchor_pos_z" if anchor_pos_failed[end] else "anchor_ori"
        termination_motion_t = int(motion_t[end])
        terminated = True
    else:
        if int(motion_t[-1]) < motion_steps:
            raise ValueError(
                "trajectory ended before motion completion without anchor termination: "
                f"motion_t={int(motion_t[-1])}, expected={motion_steps}"
            )
        end = len(motion_t)
        reason = "motion_end"
        termination_motion_t = int(motion_t[-1])
        terminated = False

    return (
        float(np.sum(reward[:end], dtype=np.float64) / motion_steps),
        float(np.sum(reward, dtype=np.float64) / motion_steps),
        terminated,
        reason,
        termination_motion_t,
        end if terminated else len(motion_t) - 1,
    )


def _compute_one(path: Path) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    frame_idx = _select_policy_frames(data)
    names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
    tracking_indices = _indices_for_patterns(names, TRACKING_BODY_PATTERNS)
    anchor_idx = names.index(ANCHOR_BODY_NAME)

    robot_pos = np.asarray(data["robot_body_pos_w"], dtype=np.float32)[frame_idx]
    robot_quat = np.asarray(data["robot_body_quat_w"], dtype=np.float32)[frame_idx]
    motion_pos = np.asarray(data["motion_body_pos_w"], dtype=np.float32)[frame_idx]
    motion_quat = np.asarray(data["motion_body_quat_w"], dtype=np.float32)[frame_idx]
    motion_t = np.asarray(data["motion_t"], dtype=np.int32)[frame_idx]

    robot_pos_local, robot_quat_local = _local_tracking_state(
        robot_pos, robot_quat, anchor_idx
    )
    motion_pos_local, motion_quat_local = _local_tracking_state(
        motion_pos, motion_quat, anchor_idx
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
    wrist_indices = [names.index(name) for name in WRIST_BODY_NAMES]
    wrist_pos_error_local = np.linalg.norm(
        motion_pos_local[:, wrist_indices] - robot_pos_local[:, wrist_indices],
        axis=-1,
    )
    wrist_ori_error_local = _quat_angle_magnitude(
        quat_mul(
            quat_conjugate(motion_quat_local[:, wrist_indices]),
            robot_quat_local[:, wrist_indices],
        )
    )
    motion_length = int(np.asarray(data["motion_length"]).reshape(())) if "motion_length" in data else int(motion_t[-1]) + 1
    motion_denominator = max(1, motion_length - 1)
    (
        normalized_tracking_return,
        mean_tracking_reward,
        return_terminated,
        return_termination_reason,
        return_termination_motion_t,
        termination_idx,
    ) = _normalized_tracking_return(
        robot_pos,
        robot_quat,
        motion_pos,
        motion_quat,
        names,
        motion_t,
        motion_denominator,
    )
    terminated = return_terminated
    termination_reason = return_termination_reason
    termination_motion_t = return_termination_motion_t
    pre_end = max(1, termination_idx if terminated else termination_idx + 1)
    progress = min(1.0, max(0.0, float(termination_motion_t) / float(motion_denominator)))
    local_body_tracking_error = float(np.mean(body_pos_error_local[:pre_end]))
    local_body_orientation_error = float(np.mean(body_ori_error_local[:pre_end]))
    wrist_tracking_error = float(np.mean(wrist_pos_error_local[:pre_end]))
    wrist_orientation_error = float(np.mean(wrist_ori_error_local[:pre_end]))

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
        "termination_motion_t": termination_motion_t,
        "termination_reason": termination_reason,
        "terminated": int(terminated),
        "progress": progress,
        "global_root_tracking_error": global_root_tracking_error,
        "global_root_tracking_error_xy": global_root_tracking_error_xy,
        "local_body_tracking_error": local_body_tracking_error,
        "local_body_orientation_error": local_body_orientation_error,
        "wrist_tracking_error": wrist_tracking_error,
        "wrist_orientation_error": wrist_orientation_error,
        "mpjpe": local_body_tracking_error,
        "normalized_tracking_return": normalized_tracking_return,
        "mean_tracking_reward": mean_tracking_reward,
        "return_terminated": int(return_terminated),
        "return_termination_reason": return_termination_reason,
        "return_termination_motion_t": return_termination_motion_t,
        "root_final_error_norm": float(np.linalg.norm(root_final_error)),
        "root_final_error_xy_norm": float(np.linalg.norm(root_final_error[:2])),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _weighted_mean_std(values: list[float], weights: list[float]) -> dict[str, float]:
    value_arr = np.asarray(values, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    mean = float(np.average(value_arr, weights=weight_arr))
    variance = float(np.average(np.square(value_arr - mean), weights=weight_arr))
    return {"mean": mean, "std": variance**0.5}


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    motion_steps = [max(1, int(row["motion_length"]) - 1) for row in rows]
    return {
        "count": len(rows),
        "progress": _mean_std([float(row["progress"]) for row in rows]),
        "global_root_tracking_error": _mean_std([float(row["global_root_tracking_error"]) for row in rows]),
        "global_root_tracking_error_xy": _mean_std([float(row["global_root_tracking_error_xy"]) for row in rows]),
        "local_body_tracking_error": _mean_std([float(row["local_body_tracking_error"]) for row in rows]),
        "local_body_orientation_error": _mean_std(
            [float(row["local_body_orientation_error"]) for row in rows]
        ),
        "wrist_tracking_error": _mean_std(
            [float(row["wrist_tracking_error"]) for row in rows]
        ),
        "wrist_orientation_error": _mean_std(
            [float(row["wrist_orientation_error"]) for row in rows]
        ),
        "mpjpe": _mean_std([float(row["mpjpe"]) for row in rows]),
        "normalized_tracking_return": _weighted_mean_std(
            [float(row["normalized_tracking_return"]) for row in rows],
            motion_steps,
        ),
        "mean_tracking_reward": _weighted_mean_std(
            [float(row["mean_tracking_reward"]) for row in rows],
            motion_steps,
        ),
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
