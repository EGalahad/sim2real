# Cross-Codebase Tracking Evaluation

This is the reproducible evaluation used by the published MimicLite comparison.
It runs every policy through the same integrated MuJoCo simulator, starts from
motion frame 0, and records complete body trajectories before computing metrics
offline.

## Protocol

| Split | Motions | Seeds |
|---|---:|---|
| LAFAN-40 | 40 | 0 |
| PHUMA-30 | 30 | 0 |
| Root-90 | 90 | 0 |

The exact ordered motion lists live in
`scripts/tracking_experiment/manifests/`. Entries are relative to their dataset
root, so the datasets may be stored anywhere.

LAFAN-40 is available from
[`elijahgalahad/any4hdmi-g1-lafan`](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-lafan):

```bash
hf download elijahgalahad/any4hdmi-g1-lafan \
  --repo-type dataset \
  --local-dir datasets/lafan40
```

PHUMA-30 and Root-90 motions are not redistributed by this repository. Point
the command below at locally prepared dataset roots containing every relative
path in `phuma30.txt` and `root90.txt`.

## Evaluate a New Policy

Run the complete protocol with one command from the repository root:

```bash
uv run scripts/tracking_experiment/run_canonical_tracking_eval.py \
  --lafan-root datasets/lafan40 \
  --phuma-root /path/to/phuma30 \
  --root90-root /path/to/root90 \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_canonical \
  --max-workers 8
```

Repeat `--policy name=path` to compare multiple policies in the same run. Add
`--skip-existing` to recompute metrics and resume only missing or unreadable
trajectory files. Use `--dry-run` to inspect all three generated commands
without launching MuJoCo.

Each split writes trajectories, `tracking_metrics.csv`,
`tracking_metrics.json`, and `summary.json`. The output root also contains
`canonical_summary.json` with all three split summaries.

## Metrics

- Progress and tracking errors use the common report failure conditions.
- `normalized_tracking_return` uses only the pinned BeyondMimic relative
  body-position (`std=0.3`) and body-orientation (`std=0.4`) rewards. The common
  termination is torso-anchor Z error `> 0.25 m` or projected-gravity Z
  difference `> 0.8`. The terminating frame and later frames contribute zero,
  while the denominator remains the full reference length.
- `mean_tracking_reward` ignores termination and averages the same reward over
  every recorded motion step.

Policy summaries weight both reward metrics by the number of reference steps,
equivalent to dividing the total reward across all motions by the total number
of reference steps.

## Reproduce the Published Plots

The committed CSV contains the exact aggregates used by the figures. Plotting
does not require the full simulator environment:

```bash
uv run --no-project --with matplotlib \
  scripts/tracking_experiment/plot_cross_codebase_release_bars.py
```

Generate the seven-policy release figure or the termination-free
companion:

```bash
uv run --no-project --with matplotlib \
  scripts/tracking_experiment/plot_cross_codebase_release_bars.py \
  --release-only \
  --output-base outputs/cross_codebase_eval_seven_policy

uv run --no-project --with matplotlib \
  scripts/tracking_experiment/plot_cross_codebase_release_bars.py \
  --release-only \
  --reward-metric all-step \
  --output-base outputs/cross_codebase_eval_seven_policy_all_step_reward
```

The canonical layout constants are grouped at the top of
`plot_cross_codebase_release_bars.py`: release-view `bottom=0.18`, all-policy
`bottom=0.26`, and `wspace=0.18`.
