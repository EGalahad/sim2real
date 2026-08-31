---
title: Pico Teleoperation
sidebar_position: 2
---

This tutorial uses the teleop publisher for live Pico / XR retargeting, its built-in mjviser server to inspect the retargeted G1 motion, and the root project tracking policy for execution.

## 1. Start the Pico retarget publisher

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py
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
  --policy-config checkpoints/mimic-lite/roa/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

### Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). The Pico-specific policy flags stay the same:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/roa/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Pico Controls

- Press `A` to enter the init pose.
- Press `A` + `B` to enter policy mode.
- Press `X` to unpause the motion flow.

## DH116S hand control

The Pico publisher also sends the left and right analog trigger values as
normalized hand-grip commands on TCP port `5593`. The left trigger controls the
left DH116S and the right trigger controls the right DH116S.

Install the repo-local SDK once on the computer with the DH116S CANFD adapters:

```bash
./third_party/dh116s_sdk/install.sh
uv sync --project venv/dh116s
```

The installer detects `aarch64` or `x86_64` and writes the runtime under
`third_party/dh116s_sdk/python`. Then start the hand process from the root
project environment:

```bash
uv run --project venv/dh116s --no-sync scripts/dh116s_control.py \
  --hand-dir double \
  --connect tcp://<pico-publisher-ip>:5593
```

:::warning
The hardware process enables and homes every requested hand during startup.
Clear the workspace around both hands before launching it. If either hand fails
to initialize in `double` mode, the process disconnects and exits.
:::

The default hardware mapping is left hand `can0` / node `1` and right hand
`can1` / node `1`. To validate the ZMQ stream and the conservative 40% grasp
mapping without importing the SDK or moving hardware, run:

```bash
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_control.py --dry-run --hand-dir double
```

When the grip stream becomes stale, the process keeps the last commanded hand
pose. Stopping the process disconnects the SDK without automatically opening
the hands. Use `--hand-dir left` or `--hand-dir right` for staged bring-up.

## Notes

- `pico_retarget_pub.py` publishes the live motion stream consumed by the tracking policy and opens the retarget mjviser server.
- Hand-grip messages use their own ZMQ port and do not change the existing Pico button protocol.
- `sim2real/sim_env/base_sim.py` is the sim2sim execution backend.
- For real hardware, [Robot I/O](/reference/robot-io) lists the inline and bridge deployment modes.
- If the publisher and policy run on different machines, add `--motion-zmq-connect tcp://<publisher_ip>:28701`.

## Next Steps

- [Motion Recording](/tutorials/motion-recording)
