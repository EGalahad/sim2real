#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "assets/mimic_lite_cross_codebase_tracking_eval.csv"
WRISTS = ROOT / "assets/mimic_lite_phuma30_wrist_error.csv"
OUTPUT = ROOT / "docs/src/data/leaderboard.json"
LINKS = {
    "mimic_lite_huge": "https://github.com/EGalahad/mimic-lite",
    "mimic_lite_v1_1": "https://github.com/EGalahad/mimic-lite",
    "g1_roa_huge_student_20260814": "https://github.com/EGalahad/mimic-lite",
    "mimic_lite_base": "https://github.com/EGalahad/mimic-lite",
    "mimic_lite_small": "https://github.com/EGalahad/mimic-lite",
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
}


def mean(*values: float) -> float:
    return sum(values) / len(values)


def main() -> None:
    with WRISTS.open(newline="", encoding="utf-8") as handle:
        wrists = {row["policy"]: float(row["wrist_pos_mm"]) for row in csv.DictReader(handle)}
    with METRICS.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))

    rows = []
    for row in source:
        key = row["policy"]
        values = {
            name: float(value)
            for name, value in row.items()
            if name not in {"policy", "label"}
        }
        assert key in LINKS and all(math.isfinite(value) for value in values.values())
        rows.append(
            {
                "key": key,
                "name": row["label"],
                "url": LINKS[key],
                "trackingReturn": mean(
                    values["lafan40_tracking_return"],
                    values["phuma30_tracking_return"],
                    values["root90_tracking_return"],
                ),
                "localError": mean(values["phuma30_local_mm"], values["root90_local_mm"]),
                "wristError": wrists.get(key),
                "globalRootError": values["root90_mean_m"],
                "lafanReturn": values["lafan40_tracking_return"],
                "phumaReturn": values["phuma30_tracking_return"],
                "root90Return": values["root90_tracking_return"],
                "lafanProgress": values["lafan40_progress_pct"],
                "phumaProgress": values["phuma30_progress_pct"],
                "phumaLocal": values["phuma30_local_mm"],
                "root90Local": values["root90_local_mm"],
                "rootForward": values["root90_forward_m"],
                "rootBackward": values["root90_backward_m"],
                "rootSideward": values["root90_sideward_m"],
            }
        )

    assert len(rows) == len(LINKS) == len({row["key"] for row in rows})
    assert set(wrists) <= {row["key"] for row in rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
