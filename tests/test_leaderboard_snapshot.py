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

    assert len(source) == 20
    assert len(page) == 20
    assert {
        "mimic_lite_v1_1",
        "mimic_lite_ppo",
        "mimic_lite_roa",
        "grit_v0_0_1",
        "partymimic_mlp",
        "supermimic_repro",
        "nablamimic_repro",
        "blinkmimic_repro",
        "xmimic_repro",
    } <= {row["key"] for row in page}
    assert [row["key"] for row in page][:3] == ["partymimic_mlp", "supermimic_repro", "mimic_lite_v1_1"]
    assert {row["key"] for row in page} <= {
        policy for policy, _dataset in motiondecode
    }
    assert {"mimic_lite_huge", "mimic_lite_base", "mimic_lite_small", "g1_roa_huge_student_20260814"}.isdisjoint(
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
        legacy_lafan = source[row["key"]]["lafan40_local_mm"]
        if legacy_lafan:
            assert math.isclose(
                row["metrics"]["bodyPos"]["datasets"]["lafan"],
                float(legacy_lafan),
            )
        else:
            assert row["metrics"]["bodyPos"]["datasets"]["lafan"] is None

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

    v1_1 = next(row for row in page if row["key"] == "mimic_lite_v1_1")
    assert v1_1["name"] == "Mimic Lite v1.1"
    assert math.isclose(v1_1["metrics"]["progress"]["datasets"]["locomotion"], 99.32033527696792)
    assert math.isclose(v1_1["metrics"]["progress"]["datasets"]["ground"], 65.5418433385652)
    assert math.isclose(v1_1["metrics"]["progress"]["datasets"]["dance"], 55.83687444369809)
    assert math.isclose(v1_1["metrics"]["wristOri"]["datasets"]["manipulation"], 0.09997126250527799)
    assert math.isclose(v1_1["metrics"]["gpuHours"]["mean"], 50.10579994295982)
    assert all(v1_1["metrics"]["bodyPos"]["datasets"][split] is None for split in ("lafan", "phuma", "root90"))

    sonic = next(row for row in page if row["key"] == "sonic_g1")
    assert sonic["metrics"]["gpuHours"]["mean"] == 21000.0
    assert sonic["metrics"]["gpuHours"]["sourceUrl"].startswith("https://")
    # Global Root uses root_final_error_norm (initial_xy_yaw_aligned_endpoint_xyz_v1)
    # for every row, rescored from the saved MotionDecode trajectories.
    assert math.isclose(sonic["metrics"]["globalRoot"]["datasets"]["locomotion"], 0.5071716477913818)

    heft = next(row for row in page if row["key"] == "heft")
    assert heft["metrics"]["gpuHours"]["mean"] == 116.01
    assert heft["metrics"]["gpuHours"]["sourceUrl"] == "https://heft.axell.top/"
