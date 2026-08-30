import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_leaderboard_snapshot_has_three_finite_splits() -> None:
    with (ROOT / "assets/mimic_lite_cross_codebase_tracking_eval.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source = {row["policy"]: row for row in csv.DictReader(handle)}
    page = json.loads((ROOT / "docs/src/data/leaderboard.json").read_text())

    assert len(source) == 15
    assert len(page) == 14
    assert {"mimic_lite_huge", "mimic_lite_base", "mimic_lite_v1_1"} <= {
        row["key"] for row in page
    }
    assert {"mimic_lite_small", "g1_roa_huge_student_20260814"}.isdisjoint(
        row["key"] for row in page
    )

    for row in page:
        values = [
            row[f"{split}{metric}"]
            for split in ("lafan", "phuma", "root90")
            for metric in ("Return", "Progress", "Local", "Wrist", "GlobalRoot")
        ]
        assert all(math.isfinite(value) for value in values)
        assert math.isclose(
            row["localError"],
            sum(row[f"{split}Local"] for split in ("lafan", "phuma", "root90"))
            / 3.0,
        )

    v1_1 = next(row for row in page if row["key"] == "mimic_lite_v1_1")
    assert all(v1_1[f"{split}Wrist"] > 0 for split in ("lafan", "phuma", "root90"))
