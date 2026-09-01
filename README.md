# sim2real

A lightweight and modular sim2sim and sim2real deployment stack.

Chinese version: [README_zh.md](./README_zh.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

If you're looking for the HDMI deployment stack, go to [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi).

## Runtime Artifacts

Large runtime artifacts are not stored in git. Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder and place `checkpoints/` and `third_party/` at the repo root.

See [Download Artifacts](./docs/artifacts.md) for the expected directory
layout and onboard dependency notes.

## Quick Start

```bash
uv sync --extra inference-cpu
```

For G1 onboard installation or repair, invoke the repository Codex skill
`$configure-g1-sim2real` from `.agents/skills/configure-g1-sim2real`.

Run offline motion tracking (sim2sim):

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/mimic-lite/roa/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

After both processes are up, press `]` in the policy terminal to start. Open the mjviser URL printed by `base_sim.py`, then use the Elastic Band controls in the viewer UI to disable or tune the virtual gantry.

## Migrating to sim2real

This repo includes a Codex skill for adapting policies trained in external codebases into `sim2real`:

```text
.agents/skills/adapt-policy-to-sim2real
```

Converted checkpoints are distributed through the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder.

Currently supported adapted / distributed checkpoint families:

| Policy family | Config path(s) | Notes |
| --- | --- | --- |
| MimicLite-ROA | `checkpoints/mimic-lite/roa/policy.yaml` | Latest 16x16384 PPO-ROA student release. |
| MimicLite-PPO | `checkpoints/mimic-lite/ppo/policy.yaml` | Latest 16x16384 Huge PPO release. |
| HEFT | `checkpoints/heft` | PMG and compliance variants. |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | Uses the official unmodified ONNX from [HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx); place it at `checkpoints/holomotion/v1_4_0/policy.onnx`. |
| SONIC release | `checkpoints/sonic/release` | Release G1 and SMPL encoder variants. |
| SONIC low-latency | `checkpoints/sonic/low_latency` | Low-latency G1 and SMPL variants. |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | G1 policy with heading-normalized reference orientation. |
| GRIT v0.0.1 | `checkpoints/grit/v0_0_1/policy.yaml` | Nine-frame reference context and ten-frame proprioceptive history. |
| ScaleBFM | `checkpoints/scalebfm` | ScaleBFM Humanoid Transformer XL and M ONNX exports from [WeishuaiZeng/ScaleBFM](https://huggingface.co/WeishuaiZeng/ScaleBFM). |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Latent-conditioned motion tracker. |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | TeleopIT policy wrapper. |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Humanoid-GPT policy wrapper. |
| TWIST2 | `checkpoints/twist2/policy.yaml` | TWIST2 policy wrapper. |

![Unified cross-codebase tracking evaluation](assets/mimic_lite_cross_codebase_tracking_eval.png)

For a fair comparison, we report the motion-lookahead latency required by each
policy, defined by its furthest required future-reference frame. All values use
the shared 50 Hz reference-motion contract.

| Policy | MimicLite-ROA | MimicLite-PPO | HEFT | HoloMotion | SONIC | SONIC low-latency | SONIC v1.1 | GRIT v0.0.1 | ScaleBFM XL | ScaleBFM M | BFM-Zero | TeleopIT | Humanoid-GPT | TWIST2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motion-lookahead latency | 0.08 s | 0.08 s | 0.12 s | 0.20 s | 0.90 s | 0.18 s | 0.90 s | 0.26 s | 0.10 s | 0.10 s | 0.12 s | 0.00 s | 0.02 s | 0.00 s |

## Real-robot Environments

Robot SDKs are kept out of the generic root environment. G1 inline deployment
uses `uv sync --extra inference-cpu --extra robot-g1`. See
[Robot I/O Modes](./docs/robot_io.md) for setup and deployment commands.

## Next Steps

- [Docs Home](./docs/README.md)
- [Getting Started](./docs/getting-started/README.md)
- [Offline Motion Tracking Tutorial](./docs/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation Tutorial](./docs/tutorials/pico-teleoperation.md)

## Citation

If you find sim2real useful in your research, please cite:

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
