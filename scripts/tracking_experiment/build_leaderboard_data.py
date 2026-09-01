#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "assets/mimic_lite_cross_codebase_tracking_eval.csv"
MOTIONDECODE = ROOT / "assets/motiondecode_public_dataset_metrics.csv"
OUTPUT = ROOT / "docs/src/data/leaderboard.json"
LINKS = {
    "mimic_lite_ppo": "https://github.com/Roboparty/MimicLite",
    "mimic_lite_roa": "https://github.com/Roboparty/MimicLite",
    "scalebfm_m": "https://github.com/zengweishuai/ScaleBFM",
    "scalebfm_xl": "https://github.com/zengweishuai/ScaleBFM",
    "sonic_g1": "https://nvlabs.github.io/GEAR-SONIC/",
    "sonic_low_latency": "https://nvlabs.github.io/GEAR-SONIC/",
    "sonic_v1_1": "https://nvlabs.github.io/GEAR-SONIC/",
    "holomotion": "https://github.com/HorizonRobotics/HoloMotion",
    "heft": "https://heft.axell.top/",
    "teleopit": "https://github.com/BotRunner64/Teleopit",
    "humanoid_gpt": "https://github.com/GalaxyGeneralRobotics/Humanoid-GPT",
    "bfm_zero": "https://lecar-lab.github.io/BFM-Zero/",
    "twist2": "https://github.com/amazon-far/TWIST2",
    "grit_v0_0_1": "https://github.com/mrzuang/GRIT_teleop_deploy",
}
POLICY_ORDER = (
    "mimic_lite_roa",
    "mimic_lite_ppo",
    "heft",
    "holomotion",
    "sonic_g1",
    "sonic_low_latency",
    "sonic_v1_1",
    "grit_v0_0_1",
    "scalebfm_xl",
    "scalebfm_m",
    "bfm_zero",
    "teleopit",
    "humanoid_gpt",
    "twist2",
)


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, "", "—"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {key}: {value}")
    return parsed


def split_metric(row: dict[str, str], suffix: str) -> dict[str, float | None]:
    return {
        "lafan": optional_float(row, f"lafan40_{suffix}"),
        "phuma": optional_float(row, f"phuma30_{suffix}"),
        "root90": optional_float(row, f"root90_{suffix}"),
    }


def metric_entry(
    row: dict[str, str],
    suffix: str,
    motiondecode: dict[str, dict[str, str]],
    motiondecode_key: str,
    scale: float = 1.0,
    excluded: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "datasets": {
            **{
                dataset: value * scale
                if dataset not in excluded
                and (value := optional_float(values, motiondecode_key)) is not None
                else None
                for dataset, values in motiondecode.items()
            },
            **split_metric(row, suffix),
        }
    }


def main() -> None:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    with MOTIONDECODE.open(newline="", encoding="utf-8") as handle:
        motiondecode_source = list(csv.DictReader(handle))
    motiondecode = {
        policy: {row["dataset"]: row for row in motiondecode_source if row["policy"] == policy}
        for policy in POLICY_ORDER
    }
    assert all(set(values) == {"locomotion", "manipulation", "ground", "dance"} for values in motiondecode.values())
    assert set(POLICY_ORDER) == set(LINKS)
    source.sort(key=lambda row: POLICY_ORDER.index(row["policy"]))

    rows = []
    for row in source:
        key = row["policy"]
        if key == "g1_roa_huge_student_20260814":
            continue
        values = {
            name: float(value)
            for name, value in row.items()
            if name not in {"policy", "label", "gpu_hours_url"} and value not in {"", "—"}
        }
        assert key in LINKS and all(math.isfinite(value) for value in values.values())
        datasets = motiondecode[key]
        rows.append(
            {
                "key": key,
                "name": row["label"],
                "url": LINKS[key],
                "metrics": {
                    "bodyPos": metric_entry(row, "local_mm", datasets, "body_pos_m", 1000.0),
                    "bodyOri": metric_entry(row, "body_ori_rad", datasets, "body_ori_rad"),
                    "globalRoot": metric_entry(row, "global_root_m", datasets, "global_root_m"),
                    "gpuHours": {
                        "mean": optional_float(row, "gpu_hours"),
                        "sourceUrl": row.get("gpu_hours_url") or None,
                    },
                    "wristPos": metric_entry(row, "wrist_mm", datasets, "wrist_pos_m", 1000.0, ("ground", "dance")),
                    "wristOri": metric_entry(row, "wrist_ori_rad", datasets, "wrist_ori_rad", excluded=("ground", "dance")),
                    "trackingReturn": metric_entry(row, "tracking_return", datasets, "tracking_return"),
                    "progress": metric_entry(row, "progress_pct", datasets, "progress", 100.0),
                },
            }
        )

    assert len(rows) == len(LINKS) == len({row["key"] for row in rows})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
