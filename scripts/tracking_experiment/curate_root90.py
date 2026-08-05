#!/usr/bin/env python3
"""Build a direction-clean, displacement-balanced Root-90 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DIRECTION_CODES = {
    "FW": "forward",
    "BW": "backward",
    "SW": "sideward",
    "SR": "sideward",
}
QPOS_PATTERN = re.compile(r"_(FW|BW|SW|SR)__")
DISTANCE_BANDS = ((1.5, 2.0), (2.0, 2.5), (2.5, 3.000001))


@dataclass(frozen=True)
class Candidate:
    direction: str
    style: str
    source_path: Path
    start: int
    end: int
    duration_s: float
    displacement_m: float
    straightness: float
    heading_fraction: float
    semantic_fraction: float
    initial_direction_error_deg: float
    cross_track_m: float

    @property
    def score(self) -> tuple[float, ...]:
        return (
            self.duration_s,
            self.semantic_fraction,
            self.heading_fraction,
            self.straightness,
            -self.initial_direction_error_deg,
            -self.cross_track_m,
            -abs(self.displacement_m - 2.25),
        )


def _candidate_metrics(qpos: np.ndarray, direction: str) -> dict[str, float]:
    xy = qpos[:, :2]
    delta = 25
    displacements = xy[delta:] - xy[:-delta]
    net = xy[-1] - xy[0]
    distance = float(np.linalg.norm(net))
    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    unit = net / max(distance, 1e-9)
    along = displacements @ unit
    cross = np.abs(displacements[:, 0] * unit[1] - displacements[:, 1] * unit[0])
    moving = np.linalg.norm(displacements, axis=1) > 0.025
    heading_ok = (along > 0) & (
        cross <= np.tan(np.deg2rad(20.0)) * np.maximum(along, 1e-9)
    )

    relative_xy = xy - xy[0]
    cross_track = np.abs(
        relative_xy[:, 0] * unit[1] - relative_xy[:, 1] * unit[0]
    )

    qw, qx, qy, qz = (qpos[:-delta, index] for index in range(3, 7))
    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    facing = np.column_stack((np.cos(yaw), np.sin(yaw)))
    forward = np.sum(displacements * facing, axis=1)
    sideward = displacements[:, 0] * facing[:, 1] - displacements[:, 1] * facing[:, 0]
    semantic_angle = np.tan(np.deg2rad(35.0))
    if direction == "forward":
        semantic_ok = (forward > 0) & (np.abs(sideward) <= semantic_angle * forward)
    elif direction == "backward":
        semantic_ok = (-forward > 0) & (np.abs(sideward) <= semantic_angle * -forward)
    else:
        semantic_ok = np.abs(forward) <= semantic_angle * np.abs(sideward)

    net_angle = np.arctan2(net[1], net[0])
    initial_relative_angle = np.rad2deg(
        np.arctan2(np.sin(net_angle - yaw[0]), np.cos(net_angle - yaw[0]))
    )
    if direction == "forward":
        initial_direction_error = abs(initial_relative_angle)
    elif direction == "backward":
        initial_direction_error = abs(abs(initial_relative_angle) - 180.0)
    else:
        initial_direction_error = abs(abs(initial_relative_angle) - 90.0)

    return {
        "displacement_m": distance,
        "straightness": distance / max(path_length, 1e-9),
        "heading_fraction": float(heading_ok[moving].mean()) if moving.any() else 0.0,
        "semantic_fraction": float(semantic_ok[moving].mean()) if moving.any() else 0.0,
        "initial_direction_error_deg": float(initial_direction_error),
        "cross_track_m": float(cross_track.max()),
    }


def _find_candidates(source_root: Path) -> list[Candidate]:
    manifest = json.loads((source_root / "manifest.json").read_text())
    timestep = float(manifest["timestep"])
    if not np.isclose(timestep, 0.02):
        raise ValueError(f"Root-90 curation requires 50 Hz qpos, got timestep={timestep}")

    best: dict[tuple[str, str, int], Candidate] = {}
    for source_path in sorted((source_root / "motions").rglob("*.npz")):
        match = QPOS_PATTERN.search(source_path.name)
        if match is None:
            continue
        direction = DIRECTION_CODES[match.group(1)]
        style = source_path.relative_to(source_root / "motions").parts[0]
        qpos = np.load(source_path)["qpos"]

        for start in range(0, max(0, len(qpos) - 150), 25):
            for duration_s in range(3, 13):
                end = start + duration_s * 50
                if end >= len(qpos):
                    break
                metrics = _candidate_metrics(qpos[start : end + 1], direction)
                if not (
                    1.5 <= metrics["displacement_m"] <= 3.0
                    and metrics["straightness"] >= 0.97
                    and metrics["heading_fraction"] >= 0.90
                    and metrics["semantic_fraction"] >= 0.80
                    and metrics["initial_direction_error_deg"] <= 15.0
                    and metrics["cross_track_m"] <= 0.20
                ):
                    continue
                candidate = Candidate(
                    direction=direction,
                    style=style,
                    source_path=source_path,
                    start=start,
                    end=end,
                    duration_s=float(duration_s),
                    **metrics,
                )
                band_index = next(
                    index
                    for index, (lower, upper) in enumerate(DISTANCE_BANDS)
                    if lower <= candidate.displacement_m < upper
                )
                key = (direction, style, band_index)
                if key not in best or candidate.score > best[key].score:
                    best[key] = candidate
    return list(best.values())


def _select(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    for direction in ("forward", "backward", "sideward"):
        direction_candidates = [item for item in candidates if item.direction == direction]
        bands = []
        for band_index, (lower, upper) in enumerate(DISTANCE_BANDS):
            band_candidates = [
                item
                for item in direction_candidates
                if lower <= item.displacement_m < upper
            ]
            bands.append((band_index, lower, upper, band_candidates))

        used_styles: set[str] = set()
        for _, lower, upper, band in sorted(bands, key=lambda item: len(item[3])):
            band.sort(key=lambda item: item.score, reverse=True)
            band = [item for item in band if item.style not in used_styles]
            if len(band) < 10:
                raise RuntimeError(
                    f"Only {len(band)} clean {direction} candidates in [{lower}, {upper}) m"
                )
            chosen = band[:10]
            selected.extend(chosen)
            used_styles.update(item.style for item in chosen)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_dataset(source_root: Path, output_root: Path, selected: list[Candidate]) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty output: {output_root}")
    motions_root = output_root / "motions"
    motions_root.mkdir(parents=True, exist_ok=True)

    records = []
    for direction in ("forward", "backward", "sideward"):
        rows = sorted(
            (item for item in selected if item.direction == direction),
            key=lambda item: (item.displacement_m, item.style, str(item.source_path)),
        )
        for index, candidate in enumerate(rows):
            source_qpos = np.load(candidate.source_path)["qpos"]
            qpos = np.asarray(
                source_qpos[candidate.start : candidate.end + 1], dtype=np.float32
            )
            safe_style = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate.style)
            filename = f"{direction}__{index:02d}__{safe_style}.npz"
            output_path = motions_root / filename
            np.savez_compressed(output_path, qpos=qpos)
            records.append(
                {
                    "name": filename,
                    "direction": direction,
                    "style": candidate.style,
                    "source_dataset": source_root.name,
                    "source_motion": str(
                        candidate.source_path.relative_to(source_root / "motions")
                    ),
                    "source_start_frame": candidate.start,
                    "source_end_frame": candidate.end,
                    "frames": len(qpos),
                    "duration_s": candidate.duration_s,
                    "root_xy_displacement_m": candidate.displacement_m,
                    "straightness": candidate.straightness,
                    "heading_fraction": candidate.heading_fraction,
                    "semantic_fraction": candidate.semantic_fraction,
                    "initial_direction_error_deg": candidate.initial_direction_error_deg,
                    "cross_track_m": candidate.cross_track_m,
                    "sha256": _sha256(output_path),
                }
            )

    source_manifest = json.loads((source_root / "manifest.json").read_text())
    manifest = {
        "format_version": 2,
        "dataset_name": "root90_clean",
        "mjcf": source_manifest["mjcf"],
        "motions_subdir": "motions",
        "timestep": 0.02,
        "qpos_dim": source_manifest["qpos_dim"],
        "qpos_names": source_manifest["qpos_names"],
        "num_motions": len(records),
        "source": {
            "description": "Direction-clean, displacement-balanced Root-90 evaluation set.",
            "direction_counts": {direction: 30 for direction in ("forward", "backward", "sideward")},
            "distance_bands_m": [[lower, min(upper, 3.0)] for lower, upper in DISTANCE_BANDS],
            "thresholds": {
                "root_xy_displacement_m": [1.5, 3.0],
                "straightness_min": 0.97,
                "heading_fraction_min": 0.90,
                "semantic_fraction_min": 0.80,
                "initial_direction_error_deg_max": 15.0,
                "cross_track_m_max": 0.20,
            },
            "motions": records,
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "README.md").write_text(
        "# Clean Root-90 evaluation dataset\n\n"
        "Thirty forward, backward, and sideward clips with direction-consistent "
        "root motion and 1.5--3.0 m net XY displacement. See `manifest.json` for "
        "selection thresholds and source intervals.\n"
    )
    with (output_root / "audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    _write_audit_plot(output_root)


def _write_audit_plot(output_root: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, direction in zip(axes, ("forward", "backward", "sideward")):
        for path in sorted((output_root / "motions").glob(f"{direction}__*.npz")):
            qpos = np.load(path)["qpos"]
            xy = qpos[:, :2]
            xy = xy - xy[0]
            qw, qx, qy, qz = qpos[0, 3:7]
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            cosine, sine = np.cos(yaw), np.sin(yaw)
            local_xy = np.column_stack(
                (
                    cosine * xy[:, 0] + sine * xy[:, 1],
                    -sine * xy[:, 0] + cosine * xy[:, 1],
                )
            )
            axis.plot(local_xy[:, 0], local_xy[:, 1], alpha=0.65, linewidth=1.0)
        axis.set_title(f"{direction.title()} (n=30)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.set_xlabel("root-forward / m")
    axes[0].set_ylabel("root-left / m")
    fig.suptitle("Clean Root-90 XY trajectories")
    fig.savefig(output_root / "audit_xy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    candidates = _find_candidates(args.source_root.resolve())
    selected = _select(candidates)
    _write_dataset(args.source_root.resolve(), args.output_root.resolve(), selected)
    print(f"Wrote {len(selected)} clean Root-90 clips to {args.output_root}")


if __name__ == "__main__":
    main()
