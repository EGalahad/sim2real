---
title: Offline Motion Tracking
sidebar_position: 1
---

This tutorial uses the root project tracking policy with an offline motion clip.

## Sim2Sim

Start the MuJoCo execution process. It prints the mjviser URL after startup:

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

In a second terminal, start the tracking policy:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

Process roles:

- `sim2real/sim_env/base_sim.py` executes `low_cmd` in MuJoCo and publishes `low_state`.
- `sim2real/rl_policy/tracking.py` consumes `low_state`, runs the exported policy, and publishes the next `low_cmd`.

After both processes are up, press `]` in the policy terminal to start. Use the Elastic Band controls in the mjviser UI to disable or tune the virtual gantry.

## Integrated Sim2Sim

Use the integrated runner when the policy and MuJoCo should live in one process. It loads the policy immediately, sets the robot to the first frame of the motion, waits five seconds, tracks until the motion ends, and then holds the last frame. Elastic band is disabled by default for this runner, and the runner prints its mjviser URL after startup.

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy_config checkpoints/sonic_groot_6k/policy-sonic-groot-006000.yaml \
  --motion_path ../any4hdmi/output/sonic/motions/240529/macarena_001__A545.npz
```

Add `--headless` for non-visual runs. The mjviser scene updates once per environment step when a browser client is connected. In mjviser mode, use the `Restart motion` button after the final-frame hold to reset the robot to the first frame and repeat the wait-track-hold sequence.

For quantitative evaluation, add `--trajectory-output <path>.npz` and compute
motion progress, global root tracking, and local body tracking with the scripts
under `scripts/tracking_experiment/`.

## Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). For example, the tracking policy still starts with:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
