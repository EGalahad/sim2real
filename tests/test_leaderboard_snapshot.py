import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_leaderboard_snapshot_preserves_legacy_and_adds_motiondecode() -> None:
    with (ROOT / "assets/mimic_lite_cross_codebase_tracking_eval.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source = {row["policy"]: row for row in csv.DictReader(handle)}
    with (ROOT / "assets/motiondecode_public_dataset_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        motiondecode = {
            (row["policy"], row["dataset"]): row for row in csv.DictReader(handle)
        }
    page = json.loads((ROOT / "docs/src/data/leaderboard.json").read_text())

    assert len(source) == 14
    assert len(page) == 14
    assert {"mimic_lite_ppo", "mimic_lite_roa", "grit_v0_0_1"} <= {
        row["key"] for row in page
    }
    assert {"mimic_lite_huge", "mimic_lite_base", "mimic_lite_v1_1", "mimic_lite_small", "g1_roa_huge_student_20260814"}.isdisjoint(
        row["key"] for row in page
    )

    dataset_keys = {"locomotion", "manipulation", "ground", "dance", "lafan", "phuma", "root90"}
    for row in page:
        for metric in ("bodyPos", "bodyOri", "globalRoot", "wristPos", "wristOri", "trackingReturn", "progress"):
            values = row["metrics"][metric]["datasets"]
            assert set(values) == dataset_keys
            assert all(
                value is None or math.isfinite(value) for value in values.values()
            )
        assert all(
            row["metrics"]["globalRoot"]["datasets"][dataset] is None
            for dataset in ("manipulation", "ground", "dance")
        )
        assert all(
            row["metrics"][metric]["datasets"][dataset] is None
            for metric in ("wristPos", "wristOri")
            for dataset in ("ground", "dance")
        )
        assert math.isclose(
            row["metrics"]["bodyPos"]["datasets"]["locomotion"],
            float(motiondecode[row["key"], "locomotion"]["body_pos_m"]) * 1000.0,
        )
        assert math.isclose(
            row["metrics"]["bodyPos"]["datasets"]["lafan"],
            float(source[row["key"]]["lafan40_local_mm"]),
        )

    roa = next(row for row in page if row["key"] == "mimic_lite_roa")
    assert all(
        roa["metrics"]["wristPos"]["datasets"][split] > 0
        for split in dataset_keys - {"ground", "dance"}
    )
    assert all(roa["metrics"]["bodyOri"]["datasets"][split] is not None for split in dataset_keys)
    assert all(
        roa["metrics"]["wristOri"]["datasets"][split] is not None
        for split in dataset_keys - {"ground", "dance"}
    )
    assert roa["metrics"]["gpuHours"]["mean"] is not None
    assert roa["metrics"]["gpuHours"]["sourceUrl"].startswith("https://")

    sonic = next(row for row in page if row["key"] == "sonic_g1")
    assert sonic["metrics"]["gpuHours"]["mean"] == 21000.0
    assert sonic["metrics"]["gpuHours"]["sourceUrl"].startswith("https://")

    heft = next(row for row in page if row["key"] == "heft")
    assert heft["metrics"]["gpuHours"]["mean"] == 116.01
    assert heft["metrics"]["gpuHours"]["sourceUrl"] == "https://heft.axell.top/"
