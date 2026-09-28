"""Download and evaluate the frozen public MotionDecode G1 benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = SCRIPT_DIR / "manifests" / "motiondecode.json"


def verify_dataset(root: Path, count: int) -> None:
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {path}")
    motions = (root / "motion-list.txt").read_text().splitlines()
    if len(motions) != count or len(set(motions)) != count:
        raise ValueError(f"Expected {count} unique ordered motions in {root}")
    for name in motions:
        if name not in checksums:
            raise ValueError(f"Motion missing checksum: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets/motiondecode"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=["locomotion", "manipulation", "ground", "dance"], default=["locomotion", "manipulation"])
    parser.add_argument("--policy", action="append", required=True, help="name=deploy_yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify datasets and print evaluation commands.")
    args = parser.parse_args()
    benchmark = json.loads(BENCHMARK_FILE.read_text())
    summaries = {}
    for split in dict.fromkeys(args.splits):
        spec = benchmark[split]
        root = args.datasets_root.resolve() / split
        if args.download:
            from huggingface_hub import snapshot_download
            snapshot_download(spec["repo_id"], repo_type="dataset", revision=spec["revision"], local_dir=root)
        verify_dataset(root, spec["count"])
        output = args.output_dir.resolve() / split
        cmd = [sys.executable, str(SCRIPT_DIR / "run_tracking_metrics_eval.py"),
               "--motions-root", str(root), "--motion-list", str(root / "motion-list.txt"),
               "--output-dir", str(output), "--max-workers", str(args.max_workers),
               "--seeds", "0", "--initial-pause-s", "0.0"]
        for policy in args.policy:
            cmd += ["--policy", policy]
        if args.skip_existing:
            cmd.append("--skip-existing")
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            summaries[split] = json.loads((output / "summary.json").read_text())
    if summaries:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "motiondecode_summary.json").write_text(json.dumps({
            "datasets": {split: benchmark[split] for split in summaries},
            "splits": summaries,
            "aggregation": "Equal motion weights within each split; primary headline is the equal mean of locomotion and manipulation. Root is locomotion only.",
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
