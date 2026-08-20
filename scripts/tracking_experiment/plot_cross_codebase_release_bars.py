#!/usr/bin/env python3
from __future__ import annotations

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


POLICIES = (
    Series("mimic_lite_huge", "MimicLite-Huge", "#5B2A86"),
    Series("mimic_lite_v1_1", "MimicLite-v1.1", "#F28E2B"),
    Series("sonic", "SONIC", "#76B900"),
    Series("sonic_v1_1", "SONIC-v1.1", "#00897B"),
    Series("holomotion", "HoloMotion", "#17BECF"),
    Series("heft", "HEFT", "#54A24B"),
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


progress_min = min(
    float(row[field])
    for row in rows.values()
    for field in ("lafan40_progress_pct", "phuma30_progress_pct")
)
progress_ymin = max(0.0, math.floor((progress_min - 5.0) / 10.0) * 10.0)
body_ymax = max(
    50.0,
    math.ceil(
        max(
            float(row[field])
            for row in rows.values()
            for field in ("phuma30_local_mm", "root90_local_mm")
        )
        * 1.15
        / 10.0
    )
    * 10.0,
)

fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.5), gridspec_kw={"wspace": 0.32})
fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.30)
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
    ylim=(0, 1.0),
)
axes[1].set_title("Root-90 Global Root Error ↓ / m")

grouped_bar(
    axes[2],
    ["PHUMA-30", "Root-90"],
    ["phuma30_local_mm", "root90_local_mm"],
    ylim=(0, body_ymax),
)
axes[2].set_title("Local Body Position Error ↓ / mm")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.03),
    ncol=6,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="#c8c8c8",
)
fig.savefig(OUTPUT_BASE.with_suffix(".pdf"))
fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=360)
plt.close(fig)
