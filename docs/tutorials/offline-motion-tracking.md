---
title: Offline Motion Tracking
sidebar_position: 1
---

This tutorial uses the root project tracking policy with an offline motion clip.

## Sim2Sim

Start the MuJoCo execution process:

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

In a second terminal, start the tracking policy:

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

Process roles:

- `sim2real/sim_env/base_sim.py` executes `low_cmd` in MuJoCo and publishes `low_state`.
- `sim2real/rl_policy/tracking.py` consumes `low_state`, runs the exported policy, and publishes the next `low_cmd`.

After both processes are up, press `]` in the policy terminal to start, then press `9` in the MuJoCo viewer to disable the virtual gantry.

## Sim2Real

Replace the MuJoCo execution process with the real bridge:

```bash
uv run scripts/real_bridge.py
```

In a second terminal, run the same tracking policy:

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

Process roles:

- `scripts/real_bridge.py` bridges Unitree DDS `low_state` / `low_cmd` to the shared ZMQ runtime.
- `sim2real/rl_policy/tracking.py` stays unchanged between sim2sim and sim2real.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
