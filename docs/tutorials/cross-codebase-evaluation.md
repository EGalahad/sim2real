# Cross-Codebase Tracking Evaluation

The [leaderboard](../leaderboard.mdx) uses frozen MotionDecode subsets and a shared
integrated MuJoCo evaluator. Every policy runs with CPU ONNX Runtime; complete
trajectories are saved before metrics are computed offline.

## Install and adapt a policy

```bash
uv sync --extra inference-cpu
```

For a new policy, use the
[adapt-policy-to-sim2real skill](https://github.com/EGalahad/sim2real/blob/main/.agents/skills/adapt-policy-to-sim2real/SKILL.md).
It covers ONNX export, observation/history and action semantics, a deploy YAML,
and sim2sim validation. A policy ONNX alone is insufficient: its observations,
joint order, gains and action scaling must match training. See also
[Run External Policies](run-external-policies.md).

## Frozen MotionDecode datasets

| Set | Motions | Role | Download |
|---|---:|---|---|
| Locomotion-80 | 80 | Primary; includes global root | [Hugging Face](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-locomotion) |
| Manipulation-48 | 48 | Primary; body/wrist tracking | [Hugging Face](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-manipulation) |
| Ground-60 | 60 | Secondary | [Hugging Face](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-ground) |
| Dance-40 | 40 | Secondary | [Hugging Face](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-dance) |

Source: **ChingMu**, [CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode),
revision `80b489e0378b60bb44d495d5437475ce0e084283`. Original ChingMu access terms
apply. These are small research evaluation subsets, not the full dataset.
The G1 trajectories are converted to 120 Hz NPZ without a new retargeting pass.
Each repository contains the frozen ordered `motion-list.txt`, source selection,
converted motion hashes, `manifest.json` and `checksums.json`.
Do not resample the sets or reinterpret G1 joints for another robot.

Exact published dataset revisions are pinned in
[`motiondecode.json`](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/manifests/motiondecode.json).
The runner downloads those revisions and validates every file checksum.

## Evaluate

Run from the repository root, replacing the deploy YAML with your adapted policy:

```bash
uv run scripts/tracking_experiment/run_motiondecode_eval.py \
  --download \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_motiondecode \
  --max-workers 8
```

The default evaluates Locomotion-80 and Manipulation-48. To evaluate all four sets:

```bash
uv run scripts/tracking_experiment/run_motiondecode_eval.py \
  --download --splits locomotion manipulation ground dance \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_motiondecode_all \
  --max-workers 8
```

Repeat `--policy name=path` to compare policies. Downloads are cached under
`datasets/motiondecode`; use `--datasets-root` to change the location.
`--dry-run` downloads/verifies data and prints commands without simulating.
`--skip-existing` reuses readable trajectories and recomputes metrics; use it only
when policy, simulator and dataset versions are unchanged.

Each set writes trajectories, `runs.csv`, `tracking_metrics.csv`,
`tracking_metrics.json` and `summary.json`. `motiondecode_summary.json` collects
all split summaries and dataset revisions. Keep the Git commit, dependency lock,
policy files and raw trajectories with results.

## Shared protocol and metrics

- Seed 0, 50 Hz policy frames, `initial_pause_s=0.0`, full motions without a runtime cap.
- Shared single-frame failure: pelvis-height error greater than 0.3 m or projected
  gravity Z error greater than 0.8. No root-position termination. These are offline
  scoring conditions, independent of the policy's training termination settings.
- Progress and Tracking Return stop at the same failure frame; return excludes
  that frame. Return uses BeyondMimic body position/orientation reward scales
  0.3/0.4 and divides by the full reference length (range 0–2).
- Local body/wrist position errors are in metres; orientation errors are quaternion
  angular distances in radians. Local alignment removes current pelvis XY and yaw,
  retaining absolute Z. Errors average frames before failure; wrist metrics use
  the two wrist-yaw links. Manipulation does not measure object-task success.
- Global Root is reported only for Locomotion: `root_final_error_norm` is XYZ distance
  at the last recorded valid same-time frame after fixed initial XY/yaw alignment,
  preserving absolute world Z. The terminal sentinel is excluded. XY and absolute Z
  are components, not alternate headline metrics. Always report progress beside it:
  an early failure endpoint is not the end of the full reference motion.
- New results identify the root convention as
  `initial_xy_yaw_aligned_endpoint_xyz_v1`. Historical results with the old
  trajectory-mean/full-quaternion convention must be rescored from raw trajectories;
  do not relabel their values or compare them directly to endpoint root error.
- Every motion has equal weight within its set, including return. A combined primary
  score is the equal mean of Locomotion and Manipulation means. Keep Ground/Dance
  separate and never include Manipulation root error in the primary root score.

## Legacy evaluation

LAFAN-40, PHUMA-30 and Root-90 are secondary historical panels. Their ordered lists
remain in `scripts/tracking_experiment/manifests/`, and
`run_canonical_tracking_eval.py` runs them with `--lafan-root`, `--phuma-root` and
`--root90-root`. Public datasets are
[LAFAN](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-lafan),
[PHUMA-30](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-phuma30) and
[Root-90](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-root90).
They are not substitutes for the MotionDecode leaderboard sets. Historical plotting
scripts render frozen CSV aggregates; rerendering those plots does not rerun policies.
