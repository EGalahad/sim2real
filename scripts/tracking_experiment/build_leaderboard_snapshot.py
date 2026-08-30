#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SPLITS = {"lafan40": (40, (0,)), "phuma30": (30, (0,)), "root90": (90, (0, 1, 2))}
POLICIES = {
    "bfm_zero": ("BFM-Zero", "/bfm-zero/"),
    "heft": ("HEFT", "/heft/"),
    "holomotion": ("HoloMotion", "holomotion_root90_refresh"),
    "humanoid_gpt": ("Humanoid-GPT", "/humanoid-gpt/"),
    "mimic_lite_base": ("MimicLite-Base", "/mimic-lite/4x8192-large/"),
    "mimic_lite_huge": ("MimicLite-Huge", "/mimic-lite/32x8192-huge/"),
    "mimic_lite_small": ("MimicLite-Small", "/mimic-lite/8x8192-huge/"),
    "scalebfm_m": ("ScaleBFM-M", "/scalebfm/humanoid_transformer_m/"),
    "scalebfm_xl": ("ScaleBFM-XL", "/scalebfm/humanoid_transformer_xl/"),
    "sonic_g1": ("SONIC", "/sonic/release/"),
    "sonic_low_latency": ("SONIC-Low-Latency", "/sonic/low_latency/"),
    "teleopit": ("TeleopIt", "/teleopit/"),
    "twist2": ("TWIST2", "/twist2/"),
}
FORCED_POLICIES = {
    "sonic_v1_1": "SONIC-v1.1",
    "mimic_lite_v1_1": "MimicLite-v1.1",
}
METRICS = {
    "progress_pct": ("progress", 100.0, False),
    "local_mm": ("local_body_tracking_error", 1000.0, False),
    "wrist_mm": ("wrist_tracking_error", 1000.0, False),
    "global_root_m": ("global_root_tracking_error", 1.0, False),
    "tracking_return": ("normalized_tracking_return", 1.0, True),
    "mean_tracking_reward": ("mean_tracking_reward", 1.0, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the three-split leaderboard snapshot.")
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--sonic-v1-1-dir", type=Path, required=True)
    parser.add_argument("--mimic-v1-1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def policy_key(policy_config: str) -> str:
    matches = [key for key, (_, needle) in POLICIES.items() if needle in policy_config]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify policy from {policy_config!r}: {matches}")
    return matches[0]


def read_split(
    root: Path,
    split: str,
    forced_policy: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    with (root / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["seed"]) in set(SPLITS[split][1])
        ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = forced_policy or policy_key(row["policy_config"])
        groups[key].append(row)
    return groups


def validate(key: str, split: str, rows: list[dict[str, str]]) -> None:
    motions, seeds = SPLITS[split]
    if len(rows) != motions * len(seeds):
        raise RuntimeError(f"{key}/{split}: expected {motions * len(seeds)} rows, got {len(rows)}")
    motion_names = {Path(row["motion_path"]).stem for row in rows}
    if len(motion_names) != motions or {int(row["seed"]) for row in rows} != set(seeds):
        raise RuntimeError(f"{key}/{split}: motion or seed set mismatch")
    numeric = {
        "progress", "local_body_tracking_error", "wrist_tracking_error",
        "global_root_tracking_error", "normalized_tracking_return",
        "mean_tracking_reward", "root_final_error_xy_norm",
    }
    for row in rows:
        for field in numeric:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"{key}/{split}: non-finite {field}")


def mean(rows: list[dict[str, str]], field: str, *, weighted: bool) -> float:
    values = [float(row[field]) for row in rows]
    weights = [max(1, int(row["motion_length"]) - 1) for row in rows]
    return (
        sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)
        if weighted
        else sum(values) / len(values)
    )


def direction(row: dict[str, str]) -> str:
    stem = Path(row["motion_path"]).stem.lower()
    for name in ("forward", "backward", "sideward"):
        if stem.startswith(name):
            return name
    raise ValueError(stem)


def main() -> None:
    args = parse_args()
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    sources = (
        (args.original_dir, None),
        (args.sonic_v1_1_dir, "sonic_v1_1"),
        (args.mimic_v1_1_dir, "mimic_lite_v1_1"),
    )
    for root, forced in sources:
        for split in SPLITS:
            for key, rows in read_split(root, split, forced).items():
                if split in grouped[key]:
                    raise RuntimeError(f"Duplicate {key}/{split}")
                validate(key, split, rows)
                grouped[key][split] = rows

    expected = set(POLICIES) | set(FORCED_POLICIES)
    if set(grouped) != expected or any(set(splits) != set(SPLITS) for splits in grouped.values()):
        raise RuntimeError(f"Policy/split mismatch: {sorted(grouped)}")
    for split in SPLITS:
        motion_seed_sets = {
            key: {(Path(row["motion_path"]).stem, int(row["seed"])) for row in splits[split]}
            for key, splits in grouped.items()
        }
        reference = next(iter(motion_seed_sets.values()))
        mismatched = [key for key, keys in motion_seed_sets.items() if keys != reference]
        if mismatched:
            raise RuntimeError(f"{split}: motion/seed sets differ for {mismatched}")

    rows_out = []
    for key in sorted(grouped):
        label = FORCED_POLICIES.get(key, POLICIES.get(key, (None,))[0])
        output: dict[str, str | float] = {"policy": key, "label": str(label)}
        for split, rows in grouped[key].items():
            for suffix, (field, scale, weighted) in METRICS.items():
                output[f"{split}_{suffix}"] = scale * mean(rows, field, weighted=weighted)
        root_rows = grouped[key]["root90"]
        for name in ("forward", "backward", "sideward"):
            selected = [row for row in root_rows if direction(row) == name]
            output[f"root90_{name}_m"] = mean(selected, "root_final_error_xy_norm", weighted=False)
        output["root90_mean_m"] = sum(
            float(output[f"root90_{name}_m"])
            for name in ("forward", "backward", "sideward")
        ) / 3.0
        rows_out.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {len(rows_out)} policies to {args.output}")


if __name__ == "__main__":
    main()
