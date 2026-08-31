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

    assert len(source) == 13
    assert len(page) == 13
    assert {"mimic_lite_ppo", "mimic_lite_roa"} <= {
        row["key"] for row in page
    }
    assert {"mimic_lite_huge", "mimic_lite_base", "mimic_lite_v1_1", "mimic_lite_small", "g1_roa_huge_student_20260814"}.isdisjoint(
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

    roa = next(row for row in page if row["key"] == "mimic_lite_roa")
    assert all(roa["metrics"]["wristPos"]["splits"][split] > 0 for split in ("lafan", "phuma", "root90"))
    assert roa["metrics"]["bodyOri"]["mean"] is not None
    assert roa["metrics"]["wristOri"]["mean"] is not None
    assert roa["metrics"]["gpuHours"]["mean"] is not None
    assert roa["metrics"]["gpuHours"]["sourceUrl"].startswith("https://")

    sonic = next(row for row in page if row["key"] == "sonic_g1")
    assert sonic["metrics"]["gpuHours"]["mean"] == 21000.0
    assert sonic["metrics"]["gpuHours"]["sourceUrl"].startswith("https://")
