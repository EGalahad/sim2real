---
title: Robot I/O
slug: /reference/robot-io
---

# Robot I/O Modes

`robot_io` controls how the policy reads robot state and sends low-level
commands. It is separate from:

- `inference_backend`, which controls ONNX / TensorRT policy inference.
- `motion_backend`, which controls the reference motion source.

For sim2sim with `sim_env/base_sim.py`, keep the policy on `--robot-io zmq`.
For sim2real deployment, choose one of the three modes below.

## Quick Choice

| Mode | Processes | Use when |
| --- | --- | --- |
| `--robot-io inline` | policy only | Preferred real-robot path when the policy runs on a machine with `unitree_interface`; avoids the extra ZMQ bridge hop. |
| `--robot-io zmq` + `scripts/real_bridge.py` | policy + Python DDS bridge | Use when you want the original split-process bridge based on `unitree_sdk2py`, or when `unitree_interface` is unavailable. |
| `--robot-io zmq` + `scripts/real_bridge_cpp.py` | policy + `unitree_interface` bridge | Use when you want the split-process ZMQ contract but prefer the `unitree_interface` robot binding in the bridge. |

:::tip
If hardware tests show violent joint shaking, or high-dynamic / high-speed
motions look much worse than expected, suspect unstable ZMQ I/O latency first.
Try `--robot-io inline` before retuning the policy or gains.
:::

## Inline

Inline mode creates the Unitree robot object inside `BasePolicy`. The policy
loop calls `robot.read_low_state()` and `robot.write_low_command()` directly.
No real bridge process is started.

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot-io inline \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

Use this path first for real deployment when latency jitter matters. Add
`--robot-interface <robot_network_interface>` only when the robot network
interface is not the default `eth0`.

## ZMQ With `real_bridge.py`

This mode keeps the policy and robot bridge in separate processes. The bridge
uses `unitree_sdk2py`, publishes `low_state` over ZMQ, and applies `low_cmd`
from ZMQ to the robot.

Terminal 1:

```bash
uv run scripts/real_bridge.py
```

Terminal 2:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

Add `--interface <robot_network_interface>` to the bridge command only when the
robot network interface is not the default `eth0`.

## ZMQ With `real_bridge_cpp.py`

This mode keeps the same ZMQ contract as `real_bridge.py`, but the bridge uses
`unitree_interface` for robot I/O.

Terminal 1:

```bash
uv run scripts/real_bridge_cpp.py
```

Terminal 2:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

Add `--interface <robot_network_interface>` to the bridge command only when the
robot network interface is not the default `eth0`.

## Rule Of Thumb

Use `inline` for the normal real-robot deploy path. Use ZMQ bridge mode when
you need process isolation, want to debug the policy and robot bridge
separately, or are running sim2sim with `base_sim.py`.
