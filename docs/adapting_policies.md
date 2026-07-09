---
title: Adapting Policies
slug: /reference/adapting-policies
---

# Adapting Policies

Most exported tracking policies use the same sim2real runtime. If the policy
already has a deploy YAML under `checkpoints/`, keep the normal deploy command
and only replace `--policy-config` with that YAML.

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot-io inline \
  --motion-backend zmq \
  --controller pico \
  --policy-config checkpoints/heft/pmg/policy.yaml
```

This applies to ordinary G1 tracking policies that consume the normal G1 motion
stream, such as HEFT, TeleopIT, Humanoid-GPT, and standard any4hdmi / SONIC G1
motion policies.

## Policy-Specific Runtime Requirements

Some adapted policies need a different motion source or extra runtime asset.

### BFM-Zero

BFM-Zero needs its checkpoint-specific MJCF for the MuJoCo FK used by its motion
observations. For direct NPZ playback this is stored in the policy YAML. For ZMQ
publishers, pass the same MJCF override to the publisher.

```bash
uv run sim2real/teleop/npz_pub.py \
  --motion_path ../any4hdmi/output/g1/lafan/motions/walk1_subject1.npz \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

### SONIC SMPL Mode

SONIC SMPL mode is not the normal G1 `motion_backend=zmq` stream. Use the SONIC
SMPL policy config, keep its `motion_backend: smpl_zmq` setting, or pass
`--motion-backend smpl_zmq`, and run the SMPL/XRobot publisher path.

Minimal sim2sim Pico test:

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py --publish-smpl
```

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sonic/smpl/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --controller pico
```

See [SONIC SMPL Input](/reference/sonic-smpl-input) for the data contract.

## Hardware Notes

Current notes from the G1 test setup:

- BFM-Zero works with the MJCF override.
- TeleopIT can walk well, but joint chatter has been observed and double-knee
  kneeling is not reliable yet; treat this as a deploy-infra / policy
  compatibility item before using that behavior on hardware.
- HEFT has shown light chatter but strong overall tracking behavior.
