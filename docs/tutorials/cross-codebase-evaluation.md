# Cross-Codebase Tracking Evaluation

This is the reproducible evaluation used by the published MimicLite comparison.
It runs every policy through the same integrated MuJoCo simulator, starts from
motion frame 0, and records complete body trajectories before computing metrics
offline.

## Protocol

Install the CPU inference dependencies from a fresh clone:

```bash
uv sync --extra inference-cpu
```

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
  --repo-type dataset --revision 7937b18a2cbe8a2d354a57524f25c7799134cf15 \
  --local-dir datasets/lafan40
```

PHUMA-30 and Root-90 are published in the
[Any4HDMI collection](https://huggingface.co/collections/elijahgalahad/any4hdmi):

```bash
hf download elijahgalahad/any4hdmi-g1-phuma30 \
  --repo-type dataset --revision 469997c66a71f2bf9c1b0da349178d581fa4ed8e \
  --local-dir datasets/phuma30
hf download elijahgalahad/any4hdmi-g1-root90 \
  --repo-type dataset --revision 43ce2d2e12eba6af3cea60f8f89bf0086bdcfa33 \
  --local-dir datasets/root90
```

Each repository includes the ordered manifest and SHA-256 checksums. The
committed sim2real manifests use the same relative paths.

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

- Progress and `normalized_tracking_return` use the same termination: pelvis
  Z error `> 0.25 m` or projected-gravity Z difference `> 0.8`. The terminating
  frame and later frames contribute zero reward, and progress stops at that frame.
- `normalized_tracking_return` uses the pinned BeyondMimic relative
  body-position (`std=0.3`) and body-orientation (`std=0.4`) rewards. The common
  denominator remains the full reference length.
- `mean_tracking_reward` ignores termination and averages the same reward over
  every recorded motion step.
- Local-body, wrist, and global-root tracking errors are reported independently
  on LAFAN-40, PHUMA-30, and Root-90 using the same pre-termination window.

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
