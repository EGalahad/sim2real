---
title: Run External Policies
slug: /tutorials/run-external-policies
---

# Run External Policies

The modular design of sim2real lets the same runtime execute different tracking
policies as long as they expose a compatible deploy YAML and ONNX model. We have
already converted several external policies into this format, so they can often
be interchanged by keeping the normal deploy command and only replacing
`--policy-config` with the policy YAML.

## Converted Checkpoints

Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder first, then use any checkpoint path below as the `--policy-config`
value.

| Policy | Checkpoint YAML | Notes |
| --- | --- | --- |
| HEFT PMG | `checkpoints/heft/pmg/policy.yaml` | Normal G1 motion stream. |
| HEFT Compliance | `checkpoints/heft/compliance/policy.yaml` | Normal G1 motion stream; compliance flag is forced off in the observation. |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | Normal G1 motion stream. |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Normal G1 motion stream. |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Requires the checkpoint-specific MJCF override for ZMQ publishers. |
| SONIC release G1 | `checkpoints/sonic/release/g1/policy.yaml` | Normal G1 motion stream. |
| SONIC release SMPL | `checkpoints/sonic/release/smpl/policy.yaml` | Uses `motion_backend: smpl_zmq` and the SMPL publisher. |
| SONIC low-latency G1 | `checkpoints/sonic/low_latency/g1/policy.yaml` | Normal G1 motion stream with the low-latency checkpoint. |
| SONIC low-latency SMPL | `checkpoints/sonic/low_latency/smpl/policy.yaml` | Four-frame SMPL input horizon. |
| TWIST2 | `checkpoints/twist2/policy.yaml` | Normal G1 motion stream. |

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
  --policy-config checkpoints/sonic/release/smpl/policy.yaml \
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
