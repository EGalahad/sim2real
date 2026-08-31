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
            row["metrics"][metric]["splits"][split]
            for metric in ("bodyPos", "globalRoot", "wristPos", "trackingReturn", "progress")
            for split in ("lafan", "phuma", "root90")
        ]
        assert all(math.isfinite(value) for value in values)
        assert math.isclose(
            row["metrics"]["bodyPos"]["mean"],
            sum(row["metrics"]["bodyPos"]["splits"][split] for split in ("lafan", "phuma", "root90"))
            / 3.0,
        )

    v1_1 = next(row for row in page if row["key"] == "mimic_lite_v1_1")
    assert all(v1_1["metrics"]["wristPos"]["splits"][split] > 0 for split in ("lafan", "phuma", "root90"))
    assert v1_1["metrics"]["bodyOri"]["mean"] is None
    assert v1_1["metrics"]["wristOri"]["mean"] is None
    assert v1_1["metrics"]["gpuHours"]["mean"] is None
