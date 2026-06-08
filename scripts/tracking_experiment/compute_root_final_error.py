from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from sim2real.utils.math import quat_rotate_inverse_numpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute root final displacement error from root trajectory files saved by the shared MuJoCo evaluation pipeline."
    )
    parser.add_argument("paths", nargs="+", help="Root trajectory .npz files or glob patterns.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if any(char in pattern for char in "*?[]"):
            expanded = sorted(Path(item) for item in glob.glob(str(path), recursive=True))
        else:
            expanded = [path]
        for item in expanded:
            resolved = item.resolve()
            if resolved.is_file() and resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    if not paths:
        raise FileNotFoundError(f"No root trajectory files matched: {patterns}")
    return paths


def _relative_translation(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        quat[0].reshape(1, 4),
        (pos[-1] - pos[0]).reshape(1, 3),
    )[0]


def _scalar(value: np.ndarray | str | bytes | object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _compute_one(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    robot_pos = np.asarray(data["robot_root_pos_w"], dtype=np.float32)
    robot_quat = np.asarray(data["robot_root_quat_w"], dtype=np.float32)
    motion_pos = np.asarray(data["motion_root_pos_w"], dtype=np.float32)
    motion_quat = np.asarray(data["motion_root_quat_w"], dtype=np.float32)
    if robot_pos.shape[0] < 2 or motion_pos.shape[0] < 2:
        raise ValueError(f"Need at least two trajectory frames in {path}")

    robot_rel = _relative_translation(robot_pos, robot_quat)
    motion_rel = _relative_translation(motion_pos, motion_quat)
    error = robot_rel - motion_rel
    return {
        "path": str(path),
        "policy_config": _scalar(data["policy_config"]) if "policy_config" in data else "",
        "motion_path": _scalar(data["motion_path"]) if "motion_path" in data else "",
        "seed": int(np.asarray(data["seed"]).reshape(())) if "seed" in data else -1,
        "frames": int(robot_pos.shape[0]),
        "robot_start_x": float(robot_pos[0, 0]),
        "robot_start_y": float(robot_pos[0, 1]),
        "robot_start_z": float(robot_pos[0, 2]),
        "robot_end_x": float(robot_pos[-1, 0]),
        "robot_end_y": float(robot_pos[-1, 1]),
        "robot_end_z": float(robot_pos[-1, 2]),
        "motion_start_x": float(motion_pos[0, 0]),
        "motion_start_y": float(motion_pos[0, 1]),
        "motion_start_z": float(motion_pos[0, 2]),
        "motion_end_x": float(motion_pos[-1, 0]),
        "motion_end_y": float(motion_pos[-1, 1]),
        "motion_end_z": float(motion_pos[-1, 2]),
        "robot_rel_x": float(robot_rel[0]),
        "robot_rel_y": float(robot_rel[1]),
        "robot_rel_z": float(robot_rel[2]),
        "motion_rel_x": float(motion_rel[0]),
        "motion_rel_y": float(motion_rel[1]),
        "motion_rel_z": float(motion_rel[2]),
        "root_final_error_x": float(error[0]),
        "root_final_error_y": float(error[1]),
        "root_final_error_z": float(error[2]),
        "root_final_error_norm": float(np.linalg.norm(error)),
        "root_final_error_xy_norm": float(np.linalg.norm(error[:2])),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    values = np.asarray([row["root_final_error_norm"] for row in rows], dtype=np.float64)
    xy_values = np.asarray([row["root_final_error_xy_norm"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "root_final_error_norm_mean": float(values.mean()),
        "root_final_error_norm_std": float(values.std(ddof=0)),
        "root_final_error_xy_norm_mean": float(xy_values.mean()),
        "root_final_error_xy_norm_std": float(xy_values.std(ddof=0)),
    }


def main() -> None:
    args = _parse_args()
    rows = [_compute_one(path) for path in _expand_paths(args.paths)]
    summary = _summary(rows)
    print(json.dumps({"summary": summary, "rows": rows}, indent=2))

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
        output_json.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
