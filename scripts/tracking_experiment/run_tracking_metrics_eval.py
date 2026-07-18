from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated MuJoCo evaluator and compute progress, "
            "global root tracking, and local body tracking metrics."
        )
    )
    parser.add_argument("--motions-root", required=True, help="Motion directory or dataset root.")
    parser.add_argument("--motion-list", default=None, help="Optional newline-delimited motion path list.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output dir.")
    parser.add_argument("--num-motions", type=int, default=8, help="Evaluate first N motions.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds to run.")
    parser.add_argument("--initial-pause-s", type=float, default=5.0)
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=None,
        help="Optional simulated-time limit forwarded to integrated_sim2sim.",
    )
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--policy", action="append", default=[], help="Policy as name=path.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--motion-index-offset",
        type=int,
        default=0,
        help="Offset added to motion indices when naming trajectories for sharded motion lists.",
    )
    return parser.parse_args()


def _policy_map(items: list[str]) -> dict[str, str]:
    policies: dict[str, str] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--policy must be name=path, got {item!r}")
        policies[name] = path
    if not policies:
        raise ValueError("At least one --policy name=path is required.")
    return policies


def _motion_paths(motions_root: Path, count: int, motion_list: str | None = None) -> list[Path]:
    if motion_list is not None:
        paths: list[Path] = []
        for line in Path(motion_list).expanduser().read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        if len(paths) < count:
            raise RuntimeError(f"Expected at least {count} motions in {motion_list}, got {len(paths)}")
        return paths[:count]

    root = motions_root.expanduser().resolve()
    scan_root = root / "motions" if (root / "motions").is_dir() else root
    motions = sorted(scan_root.rglob("*.npz"))
    if len(motions) < count:
        raise RuntimeError(f"Expected at least {count} motions under {scan_root}, got {len(motions)}")
    return motions[:count]


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": variance**0.5}


def _add_policy_summary(result_json: Path, result_csv: Path, run_rows: list[dict[str, str | int]]) -> dict[str, object]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    result_rows = payload["rows"]
    if len(result_rows) != len(run_rows):
        raise RuntimeError(f"Expected {len(run_rows)} result rows, got {len(result_rows)}")

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result_row, run_row in zip(result_rows, run_rows, strict=True):
        result_path = Path(str(result_row["path"])).resolve()
        run_path = Path(str(run_row["trajectory_path"])).resolve()
        if result_path != run_path:
            raise RuntimeError(
                "Result row order does not match run manifest: "
                f"result={result_path}, run={run_path}"
            )
        result_row["policy"] = run_row["policy"]
        result_row["motion_index"] = run_row["motion_index"]
        result_row["trajectory_path"] = run_row["trajectory_path"]
        groups[str(run_row["policy"])].append(result_row)

    per_policy: dict[str, dict[str, object]] = {}
    for policy_name, rows in groups.items():
        per_policy[policy_name] = {
            "count": len(rows),
            "progress_mean": _mean_std([float(row["progress"]) for row in rows])["mean"],
            "progress_std": _mean_std([float(row["progress"]) for row in rows])["std"],
            "global_root_tracking_error_mean": _mean_std(
                [float(row["global_root_tracking_error"]) for row in rows]
            )["mean"],
            "global_root_tracking_error_std": _mean_std(
                [float(row["global_root_tracking_error"]) for row in rows]
            )["std"],
            "global_root_tracking_error_xy_mean": _mean_std(
                [float(row["global_root_tracking_error_xy"]) for row in rows]
            )["mean"],
            "global_root_tracking_error_xy_std": _mean_std(
                [float(row["global_root_tracking_error_xy"]) for row in rows]
            )["std"],
            "local_body_tracking_error_mean": _mean_std(
                [float(row["local_body_tracking_error"]) for row in rows]
            )["mean"],
            "local_body_tracking_error_std": _mean_std(
                [float(row["local_body_tracking_error"]) for row in rows]
            )["std"],
            "mpjpe_mean": _mean_std([float(row["mpjpe"]) for row in rows])["mean"],
            "mpjpe_std": _mean_std([float(row["mpjpe"]) for row in rows])["std"],
            "root_final_error_norm_mean": _mean_std(
                [float(row["root_final_error_norm"]) for row in rows]
            )["mean"],
            "root_final_error_norm_std": _mean_std(
                [float(row["root_final_error_norm"]) for row in rows]
            )["std"],
            "root_final_error_xy_norm_mean": _mean_std(
                [float(row["root_final_error_xy_norm"]) for row in rows]
            )["mean"],
            "root_final_error_xy_norm_std": _mean_std(
                [float(row["root_final_error_xy_norm"]) for row in rows]
            )["std"],
        }

    payload["per_policy_summary"] = per_policy
    result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with result_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    return {"global": payload["summary"], "per_policy": per_policy}


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = _policy_map(args.policy)
    motions = _motion_paths(Path(args.motions_root), args.num_motions, args.motion_list)

    rows: list[dict[str, str | int]] = []
    for policy_name, policy_config in policies.items():
        for motion_index, motion_path in enumerate(motions):
            output_motion_index = args.motion_index_offset + motion_index
            motion_slug = motion_path.stem
            for seed in args.seeds:
                traj_path = (
                    output_dir
                    / "trajectories"
                    / policy_name
                    / f"seed_{seed}"
                    / f"{output_motion_index:02d}_{motion_slug}.npz"
                )
                if not (args.skip_existing and traj_path.is_file()):
                    cmd = [
                        sys.executable,
                        "-m",
                        "sim2real.sim_env.integrated_sim2sim",
                        "--robot",
                        args.robot,
                        "--policy-config",
                        policy_config,
                        "--motion-path",
                        str(motion_path),
                        "--headless",
                        "--run-once",
                        "--initial-pause-s",
                        str(args.initial_pause_s),
                        "--trajectory-output",
                        str(traj_path),
                        "--seed",
                        str(seed),
                    ]
                    if args.max_runtime_s is not None:
                        cmd.extend(["--max-runtime-s", str(args.max_runtime_s)])
                    _run(cmd)
                rows.append(
                    {
                        "policy": policy_name,
                        "policy_config": policy_config,
                        "motion_index": output_motion_index,
                        "motion_path": str(motion_path),
                        "seed": seed,
                        "trajectory_path": str(traj_path),
                    }
                )

    manifest_path = output_dir / "runs.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result_json = output_dir / "tracking_metrics.json"
    result_csv = output_dir / "tracking_metrics.csv"
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compute_tracking_metrics.py"),
            *[str(row["trajectory_path"]) for row in rows],
            "--output-json",
            str(result_json),
            "--output-csv",
            str(result_csv),
        ]
    )

    summary = _add_policy_summary(result_json, result_csv, rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
