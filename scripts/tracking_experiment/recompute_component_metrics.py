#!/usr/bin/env python3
"""Recompute one completed evaluation component from its saved trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_tracking_metrics_eval import _add_policy_summary, _is_readable_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    manifest = output_dir / "runs.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [
            {
                "policy": path.parents[1].name,
                "motion_index": int(path.name.split("_", 1)[0]),
                "trajectory_path": str(path),
            }
            for path in sorted((output_dir / "trajectories").glob("*/seed_*/*.npz"))
        ]
    paths = [Path(row["trajectory_path"]).resolve() for row in rows]
    if not rows or len(paths) != len(set(paths)) or not all(map(_is_readable_npz, paths)):
        raise RuntimeError(f"Incomplete or duplicate trajectories in {output_dir}")

    result_json = output_dir / "final_tracking_metrics.json"
    result_csv = output_dir / "final_tracking_metrics.csv"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("compute_tracking_metrics.py")),
            *map(str, paths),
            "--output-json",
            str(result_json),
            "--output-csv",
            str(result_csv),
        ],
        check=True,
    )
    summary = _add_policy_summary(result_json, result_csv, rows)
    (output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{output_dir.name}: {len(rows)} trajectories recomputed", flush=True)


if __name__ == "__main__":
    main()
