# Integrated MuJoCo Tracking Evaluation

This directory contains the scripts used to evaluate tracking policies in one
MuJoCo process. The evaluator runs the exported policy, the MuJoCo simulation,
and the motion reference in the same Python process, then saves trajectory data
for metric computation.

Run all commands from the `sim2real/` repository root.

## Core Files

- `sim2real/sim_env/integrated_sim2sim.py`
  - Runs one policy on one motion in MuJoCo.
  - Initializes the robot from motion frame 0, waits for `--initial-pause-s`,
    starts policy tracking, and can stop when the motion reaches the last frame.
  - Saves root-only trajectories with `--root-trajectory-output`.
  - Saves full root/body trajectories with `--trajectory-output`.
- `run_tracking_metrics_eval.py`
  - Batch runner for multiple policies, motions, and seeds.
  - Calls `integrated_sim2sim.py` once per `(policy, motion, seed)`.
  - Writes `runs.csv`, trajectory `.npz` files, `tracking_metrics.csv`,
    `tracking_metrics.json`, and `summary.json`.
- `compute_tracking_metrics.py`
  - Computes motion progress, global root tracking, and local body tracking from
    saved full trajectory `.npz` files.
- `run_root_final_error_eval.py` and `compute_root_final_error.py`
  - Legacy root-final-displacement-only evaluation path. Keep this for older
    plots, but use `run_tracking_metrics_eval.py` for new tracking comparisons.
- `convert_xrobot_raw_zip_to_any4hdmi.py`
  - Converts XRobot raw G1 motion dumps into any4hdmi `.npz` motion files.
- `visualize_root_trajectory.py` and `plot_root_final_error_bars.py`
  - Helpers for inspecting root trajectories and old root-final-error outputs.

## Metrics

`compute_tracking_metrics.py` reports one row per rollout.

- `progress`
  - Motion completion ratio at the first tracking failure.
  - Failure conditions match the report protocol:
    `root_ori_error >= 1.2` for 25 consecutive policy frames,
    local body position error `>= 0.4 m` for 5 consecutive policy frames, or
    local body orientation error `>= 1.2 rad` for 5 consecutive policy frames.
  - If no failure occurs before motion end, progress is `1.0`.
- `global_root_tracking_error`
  - Mean 3D root trajectory error before the first failure.
  - Both robot and reference root trajectories are converted to their own
    start-frame local coordinate systems before differencing.
- `global_root_tracking_error_xy`
  - Same as `global_root_tracking_error`, but measured only in the horizontal
    plane.
- `local_body_tracking_error`
  - Mean local body position error before the first failure over the configured
    tracking bodies.
  - `mpjpe` is kept as an alias for compatibility with existing plotting code.
- `root_final_error_norm` and `root_final_error_xy_norm`
  - Legacy final root displacement errors. These compare only the final
    start-frame-relative displacement and are not the primary global tracking
    metric for new evaluations.

## Run One Rollout

Use `integrated_sim2sim.py` directly when debugging a single policy/motion pair:

```bash
uv run python -m sim2real.sim_env.integrated_sim2sim \
  --robot g1 \
  --policy-config checkpoints/example_policy/policy.yaml \
  --motion-path ../any4hdmi/output/lafan/motions/example_motion.npz \
  --headless \
  --run-once \
  --initial-pause-s 5.0 \
  --trajectory-output outputs/example_eval/trajectory.npz \
  --seed 0
```

Drop `--headless` to inspect the rollout in the MuJoCo viewer. In viewer mode,
pressing space after the final-frame hold restarts the motion from frame 0.

## Run Batched Tracking Metrics

Evaluate one or more policies over a motion directory:

```bash
uv run python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root ../any4hdmi/output/lafan/motions \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --policy sonic=checkpoints/sonic/g1/policy.yaml \
  --num-motions 40 \
  --seeds 0 1 2 \
  --output-dir outputs/tracking_eval/lafan40
```

Recompute tables from existing trajectory files without rerunning MuJoCo:

```bash
uv run python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root ../any4hdmi/output/lafan/motions \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --policy sonic=checkpoints/sonic/g1/policy.yaml \
  --num-motions 40 \
  --seeds 0 1 2 \
  --output-dir outputs/tracking_eval/lafan40 \
  --skip-existing
```

Output layout:

```text
outputs/tracking_eval/lafan40/
  runs.csv
  tracking_metrics.csv
  tracking_metrics.json
  summary.json
  trajectories/
    <policy>/
      seed_<seed>/
        <motion_index>_<motion_slug>.npz
```

## Compute Metrics From Existing Trajectories

Use this when another script already produced full trajectory `.npz` files:

```bash
uv run python scripts/tracking_experiment/compute_tracking_metrics.py \
  "outputs/tracking_eval/lafan40/trajectories/*/seed_*/*.npz" \
  --output-csv outputs/tracking_eval/lafan40/tracking_metrics.csv \
  --output-json outputs/tracking_eval/lafan40/tracking_metrics.json
```

## Legacy Root Final Error

The older root-final-error path is still available:

```bash
uv run python scripts/tracking_experiment/run_root_final_error_eval.py \
  --no-default-policies \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --motions-root ../any4hdmi/output/xrobot_raw_20260524/motions \
  --num-motions 8 \
  --seeds 0 1 2 \
  --output-dir outputs/root_final_error_eval/mimic_lite_ppo
```

This path writes root-only trajectories and computes
`root_final_error_norm` / `root_final_error_xy_norm`; it does not compute
motion progress or local body tracking.
