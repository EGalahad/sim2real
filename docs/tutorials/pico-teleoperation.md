---
title: Pico Teleoperation
sidebar_position: 2
---

This tutorial uses the teleop publisher for live Pico / XR retargeting, the realtime viewer to inspect the retargeted G1 motion, and the root project tracking policy for execution.

## 1. Start the Pico retarget publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

## 2. Inspect the retarget in realtime

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://127.0.0.1:28701 \
  --viewer_hz 50
```

Keep the viewer open until the retargeted G1 motion looks correct.

## 3. Choose the execution backend

### Sim2Sim

Start the MuJoCo execution process:

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

In another terminal, start the tracking policy against the live motion stream:

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion_backend zmq \
  --motion_zmq_connect tcp://127.0.0.1:28701
```

### Sim2Real

Replace the MuJoCo execution process with the real bridge:

```bash
uv run scripts/real_bridge.py
```

Run the same tracking policy command:

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion_backend zmq \
  --motion_zmq_connect tcp://127.0.0.1:28701
```

## Notes

- `pico_retarget_pub.py` publishes the live motion stream consumed by both the realtime viewer and the tracking policy.
- `sim2real/sim_env/base_sim.py` is the sim2sim execution backend.
- `scripts/real_bridge.py` is the sim2real execution backend.
- If you want the policy control mode to come from the Pico controller topic instead of the keyboard, add `--controller pico` to the tracking command.
- If the publisher and policy run on different machines, replace `tcp://127.0.0.1:28701` with the publisher machine's IP.

## Next Steps

- [Motion Recording](/tutorials/motion-recording)
