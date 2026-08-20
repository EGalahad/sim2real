from pathlib import Path

from scripts.tracking_experiment.run_tracking_metrics_eval import _motion_paths


def test_motion_manifest_paths_are_relative_to_dataset_root(tmp_path: Path) -> None:
    motion = tmp_path / "motions" / "example.npz"
    motion.parent.mkdir()
    motion.touch()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("motions/example.npz\n", encoding="utf-8")

    assert _motion_paths(tmp_path, None, str(manifest)) == [motion]


def test_canonical_manifest_sizes() -> None:
    manifest_dir = (
        Path(__file__).parents[1] / "scripts" / "tracking_experiment" / "manifests"
    )
    for name, expected in (("lafan40", 40), ("phuma30", 30), ("root90", 90)):
        paths = (manifest_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        assert len(paths) == expected
        assert len(set(paths)) == expected
        assert all(not Path(path).is_absolute() for path in paths)
