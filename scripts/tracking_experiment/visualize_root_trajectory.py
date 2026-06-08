from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim2real.utils.math import quat_rotate_inverse_numpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize robot and motion root trajectories saved by the shared MuJoCo evaluation pipeline."
    )
    parser.add_argument(
        "trajectory_npz",
        help="NPZ file produced by the shared MuJoCo evaluation pipeline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <trajectory_npz stem>_root_trajectory.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def _relative_positions(pos_w: np.ndarray, quat_w: np.ndarray) -> np.ndarray:
    rel_w = pos_w - pos_w[0]
    start_quat = np.repeat(quat_w[0:1], pos_w.shape[0], axis=0)
    return quat_rotate_inverse_numpy(start_quat, rel_w)


def _axis_equal_xy(ax, *paths: np.ndarray) -> None:
    xy = np.concatenate([path[:, :2] for path in paths], axis=0)
    center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    span = max(span, 0.5)
    pad = span * 0.08
    half = span * 0.5 + pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def _plot_xy(ax, robot: np.ndarray, motion: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    ax.plot(motion[:, 0], motion[:, 1], color="#2f7d32", linewidth=2.0, label="motion")
    ax.plot(robot[:, 0], robot[:, 1], color="#1f5fbf", linewidth=2.0, label="robot")
    ax.scatter(motion[0, 0], motion[0, 1], color="#2f7d32", marker="o", s=32, label="motion start")
    ax.scatter(motion[-1, 0], motion[-1, 1], color="#2f7d32", marker="x", s=48, label="motion end")
    ax.scatter(robot[0, 0], robot[0, 1], color="#1f5fbf", marker="o", s=32, label="robot start")
    ax.scatter(robot[-1, 0], robot[-1, 1], color="#1f5fbf", marker="x", s=48, label="robot end")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    _axis_equal_xy(ax, robot, motion)


def _plot_components(ax, t: np.ndarray, robot: np.ndarray, motion: np.ndarray) -> None:
    colors = {"x": "#1f5fbf", "y": "#8b2bbf", "z": "#c76f00"}
    for dim, name in enumerate(["x", "y", "z"]):
        ax.plot(t, motion[:, dim], color=colors[name], linestyle="--", linewidth=1.4, label=f"motion {name}")
        ax.plot(t, robot[:, dim], color=colors[name], linestyle="-", linewidth=1.4, label=f"robot {name}")
    ax.set_title("World position components")
    ax.set_xlabel("time since motion unpause (s)")
    ax.set_ylabel("position (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(ncols=3, fontsize=8)


def main() -> None:
    args = _parse_args()
    trajectory_path = Path(args.trajectory_npz).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else trajectory_path.with_name(f"{trajectory_path.stem}_root_trajectory.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(trajectory_path, allow_pickle=False)
    robot_w = np.asarray(data["robot_root_pos_w"], dtype=np.float64)
    motion_w = np.asarray(data["motion_root_pos_w"], dtype=np.float64)
    robot_quat_w = np.asarray(data["robot_root_quat_w"], dtype=np.float64)
    motion_quat_w = np.asarray(data["motion_root_quat_w"], dtype=np.float64)
    sim_time = np.asarray(data["sim_time"], dtype=np.float64)
    t = sim_time - sim_time[0] if sim_time.size else np.arange(robot_w.shape[0], dtype=np.float64)

    robot_rel = _relative_positions(robot_w, robot_quat_w)
    motion_rel = _relative_positions(motion_w, motion_quat_w)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root_error = float(np.asarray(data["root_final_error_norm"]).reshape(())) if "root_final_error_norm" in data else None
    xy_error = float(np.asarray(data["root_final_error_xy_norm"]).reshape(())) if "root_final_error_xy_norm" in data else None
    title = args.title or trajectory_path.name
    if root_error is not None and xy_error is not None:
        title = f"{title}\nroot final error={root_error:.3f} m, xy={xy_error:.3f} m"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(title, fontsize=11)
    _plot_xy(axes[0], robot_w, motion_w, "World XY root trajectory", "world x (m)", "world y (m)")
    _plot_xy(axes[1], robot_rel, motion_rel, "Start-frame relative XY", "local x (m)", "local y (m)")
    _plot_components(axes[2], t, robot_w, motion_w)
    axes[0].legend(fontsize=8, loc="best")

    fig.savefig(output_path, dpi=args.dpi)
    print(output_path)


if __name__ == "__main__":
    main()
