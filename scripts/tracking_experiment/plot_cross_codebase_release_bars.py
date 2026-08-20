#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sim2real-cross-codebase")

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "assets/mimic_lite_cross_codebase_tracking_eval.csv"
OUTPUT_BASE = ROOT / "assets/mimic_lite_cross_codebase_tracking_eval"

# Canonical release-plot layout. Keep this block in sync with published figures.
FIGSIZE = (25.5, 7.1)
SUBPLOT_LAYOUT = {
    "left": 0.06,
    "right": 0.99,
    "top": 0.82,
}
RELEASE_BOTTOM = 0.18
ALL_POLICY_BOTTOM = 0.26
WSPACE = 0.18
LEGEND_Y = 0.03

for font_file in (
    Path.home() / ".local/share/fonts/windows-arial/arial.ttf",
    Path.home() / ".local/share/fonts/windows-arial/arialbd.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
):
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.titlesize": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    color: str


ALL_POLICIES = (
    Series("mimic_lite_huge", "MimicLite-Huge", "#5B2A86"),
    Series("mimic_lite_v1_1", "MimicLite-v1.1", "#1F77B4"),
    Series("g1_roa_huge_student_20260814", "MimicLite-Huge-ROA", "#F28E2B"),
    Series("mimic_lite_base", "MimicLite-Base", "#7B4AB8"),
    Series("mimic_lite_small", "MimicLite-Small", "#A77BD4"),
    Series("scalebfm_m", "ScaleBFM-M", "#C44E52"),
    Series("scalebfm_xl", "ScaleBFM-XL", "#E07B7B"),
    Series("sonic_g1", "SONIC", "#76B900"),
    Series("sonic_low_latency", "SONIC-Low-Latency", "#2E7D32"),
    Series("sonic_v1_1", "SONIC-v1.1", "#00897B"),
    Series("holomotion", "HoloMotion", "#17BECF"),
    Series("heft", "HEFT", "#54A24B"),
    Series("teleopit", "TeleopIt", "#B279A2"),
    Series("humanoid_gpt", "Humanoid-GPT", "#F58518"),
    Series("bfm_zero", "BFM-Zero", "#9D755D"),
    Series("twist2", "TWIST2", "#4C78A8"),
)
RELEASE_POLICIES = {
    "mimic_lite_huge",
    "mimic_lite_v1_1",
    "sonic_g1",
    "sonic_v1_1",
    "holomotion",
    "heft",
}

parser = argparse.ArgumentParser()
parser.add_argument("--release-only", action="store_true")
parser.add_argument("--reward-metric", choices=("return", "all-step"), default="return")
parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
args = parser.parse_args()
POLICIES = tuple(
    Series(policy.key, policy.label, "#F28E2B")
    if args.release_only and policy.key == "mimic_lite_v1_1"
    else policy
    for policy in ALL_POLICIES
    if not args.release_only or policy.key in RELEASE_POLICIES
)


with DATA_PATH.open(newline="", encoding="utf-8") as handle:
    rows = {row["policy"]: row for row in csv.DictReader(handle)}


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=4, width=0.8)


def grouped_bar(
    axis: plt.Axes,
    groups: list[str],
    fields: list[str],
    *,
    ylim: tuple[float, float],
) -> None:
    x_values = list(range(len(groups)))
    width = min(0.76 / len(POLICIES), 0.24)
    for index, item in enumerate(POLICIES):
        offset = (index - (len(POLICIES) - 1) / 2.0) * width
        axis.bar(
            [x + offset for x in x_values],
            [float(rows[item.key][field]) for field in fields],
            width=width * 0.92,
            color=item.color,
            label=item.label,
            alpha=0.92,
            edgecolor="#444444",
            linewidth=0.55,
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(groups)
    axis.set_ylim(*ylim)
    style_axis(axis)


selected_rows = [rows[policy.key] for policy in POLICIES]
progress_min = min(
    float(row[field])
    for row in selected_rows
    for field in ("lafan40_progress_pct", "phuma30_progress_pct")
)
progress_ymin = max(0.0, math.floor((progress_min - 5.0) / 10.0) * 10.0)
body_ymax = max(
    50.0,
    math.ceil(
        max(
            float(row[field])
            for row in selected_rows
            for field in ("phuma30_local_mm", "root90_local_mm")
        )
        * 1.15
        / 10.0
    )
    * 10.0,
)
reward_suffix = (
    "tracking_return" if args.reward_metric == "return" else "mean_tracking_reward"
)
reward_fields = [f"{split}_{reward_suffix}" for split in ("lafan40", "phuma30", "root90")]
reward_min = min(
    float(row[field])
    for row in selected_rows
    for field in reward_fields
)
reward_ymin = max(0.0, math.floor((reward_min - 0.05) * 10.0) / 10.0)

fig, axes = plt.subplots(
    1,
    4,
    figsize=FIGSIZE,
    gridspec_kw={"wspace": WSPACE},
)
fig.subplots_adjust(
    **SUBPLOT_LAYOUT,
    bottom=RELEASE_BOTTOM if args.release_only else ALL_POLICY_BOTTOM,
)
fig.suptitle("Unified Cross-Codebase Evaluation")

grouped_bar(
    axes[0],
    ["LAFAN-40", "PHUMA-30"],
    ["lafan40_progress_pct", "phuma30_progress_pct"],
    ylim=(progress_ymin, 100),
)
axes[0].set_title("Progress ↑ / %")

grouped_bar(
    axes[1],
    ["Forward", "Backward", "Sideward"],
    ["root90_forward_m", "root90_backward_m", "root90_sideward_m"],
    ylim=(0, 1.0 if args.release_only else 2.0),
)
axes[1].set_title("Root-90 Global Root Error ↓ / m")

grouped_bar(
    axes[2],
    ["PHUMA-30", "Root-90"],
    ["phuma30_local_mm", "root90_local_mm"],
    ylim=(0, body_ymax),
)
axes[2].set_title("Local Body Position Error ↓ / mm")

grouped_bar(
    axes[3],
    ["LAFAN-40", "PHUMA-30", "Root-90"],
    reward_fields,
    ylim=(reward_ymin, 2.0),
)
axes[3].set_title(
    "Normalized Tracking Return ↑"
    if args.reward_metric == "return"
    else "All-Step Mean Tracking Reward ↑"
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, LEGEND_Y),
    ncol=6 if args.release_only else 7,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#c8c8c8",
)
args.output_base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output_base.with_suffix(".pdf"))
fig.savefig(args.output_base.with_suffix(".png"), dpi=360)
plt.close(fig)
