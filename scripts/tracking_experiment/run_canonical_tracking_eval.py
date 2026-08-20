from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
MANIFEST_DIR = SCRIPT_DIR / "manifests"
PROTOCOLS = (
    ("lafan40", "lafan_root", (0,)),
    ("phuma30", "phuma_root", (0,)),
    ("root90", "root90_root", (0,)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical G1 LAFAN-40, PHUMA-30, and Root-90 evaluation."
    )
    parser.add_argument("--lafan-root", required=True)
    parser.add_argument("--phuma-root", required=True)
    parser.add_argument("--root90-root", required=True)
    parser.add_argument("--policy", action="append", required=True, help="Policy as name=path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summaries: dict[str, object] = {}
    for protocol, root_arg, seeds in PROTOCOLS:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_tracking_metrics_eval.py"),
            "--motions-root",
            str(getattr(args, root_arg)),
            "--motion-list",
            str(MANIFEST_DIR / f"{protocol}.txt"),
            "--output-dir",
            str(output_dir / protocol),
            "--max-workers",
            str(args.max_workers),
            "--seeds",
            *[str(seed) for seed in seeds],
        ]
        for policy in args.policy:
            cmd.extend(("--policy", policy))
        if args.skip_existing:
            cmd.append("--skip-existing")
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=PROJECT_DIR)
        summaries[protocol] = json.loads(
            (output_dir / protocol / "summary.json").read_text(encoding="utf-8")
        )

    if not args.dry_run:
        (output_dir / "canonical_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
