from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


POLICY_COLORS = {
    "sonic_release": "#4b7bec",
    "sonic_trained": "#20a486",
    "lafan": "#d65f5f",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts for root final error by motion and policy."
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/root_final_error_eval/root_final_error.csv",
        help="CSV produced by compute_root_final_error.py / run_root_final_error_eval.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/root_final_error_eval/root_final_error_bars.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metric",
        choices=["root_final_error_norm", "root_final_error_xy_norm"],
        default="root_final_error_norm",
        help="Metric to plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _motion_order(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["motion_index"]) for row in rows})


def _summarize(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[tuple[int, str], tuple[float, float, int]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["motion_index"]), row["policy"])].append(float(row[metric]))
    return {
        key: (float(np.mean(values)), float(np.std(values, ddof=0)), len(values))
        for key, values in grouped.items()
    }


def _motion_label(rows: list[dict[str, str]], motion_index: int) -> str:
    for row in rows:
        if int(row["motion_index"]) == motion_index:
            return f"M{motion_index + 1}\n{Path(row['motion_path']).stem.replace('xrobot_raw_', '')}"
    return f"M{motion_index + 1}"


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_csv)
    if not rows:
        raise RuntimeError(f"No rows in {input_csv}")

    motion_order = _motion_order(rows)
    policy_order = _ordered_unique([row["policy"] for row in rows])
    summary = _summarize(rows, args.metric)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(motion_order), dtype=np.float64)
    group_width = 0.82
    bar_width = group_width / max(1, len(policy_order))

    fig, ax = plt.subplots(figsize=(max(9.0, len(motion_order) * 1.3), 5.2), constrained_layout=True)
    for policy_idx, policy in enumerate(policy_order):
        offsets = x - group_width / 2 + bar_width * (policy_idx + 0.5)
        means: list[float] = []
        stds: list[float] = []
        labels: list[str] = []
        for motion_index in motion_order:
            mean, std, count = summary.get((motion_index, policy), (np.nan, 0.0, 0))
            means.append(mean)
            stds.append(std)
            labels.append(f"n={count}")
        bars = ax.bar(
            offsets,
            means,
            width=bar_width * 0.92,
            yerr=stds,
            capsize=3,
            label=policy,
            color=POLICY_COLORS.get(policy),
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, label in zip(bars, labels, strict=True):
            if np.isfinite(bar.get_height()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="#444444",
                )

    metric_label = "root final error (m)" if args.metric == "root_final_error_norm" else "root final error XY (m)"
    title = args.title or f"{metric_label} by motion and policy"
    ax.set_title(title)
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_motion_label(rows, motion_index) for motion_index in motion_order],
        fontsize=8,
        rotation=30,
        ha="right",
    )
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(ncols=max(1, len(policy_order)), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.savefig(output_path, dpi=args.dpi)
    print(output_path)


if __name__ == "__main__":
    main()
