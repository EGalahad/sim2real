---
title: Pico Teleoperation
sidebar_position: 2
---

This tutorial uses the teleop publisher for live Pico / XR retargeting, its built-in mjviser server to inspect the retargeted G1 motion, and the root project tracking policy for execution.

## 1. Start the Pico retarget publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py
```

Open the mjviser URL printed by the publisher and keep it open until the retargeted G1 motion looks correct.

## 2. Choose the execution backend

### Sim2Sim

Start the MuJoCo execution process:

```bash
uv run sim2real/sim_env/base_sim.py
```

In another terminal, start the tracking policy against the live motion stream:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion-backend zmq \
  --controller pico
```

### Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). The Pico-specific policy flags stay the same:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion-backend zmq \
  --controller pico
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Pico Controls

- Press `A` to enter the init pose.
- Press `A` + `B` to enter policy mode.
- Press `X` to unpause the motion flow.

## Notes

- `pico_retarget_pub.py` publishes the live motion stream consumed by the tracking policy and opens the retarget mjviser server.
- `sim2real/sim_env/base_sim.py` is the sim2sim execution backend.
- For real hardware, [Robot I/O](/reference/robot-io) lists the inline and bridge deployment modes.
- If the publisher and policy run on different machines, add `--motion-zmq-connect tcp://<publisher_ip>:28701`.

## Next Steps

- [Motion Recording](/tutorials/motion-recording)
