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
# Specify device for inference
uv sync --group cpu     # Sim2sim
uv sync --group g1      # G1 Robot
uv sync --group g1-gpu  # G1 Robot using GPU
```

Run offline motion tracking (sim2sim):

```bash
# Use Mirror for Acceleration (China Mainland)
export HF_ENDPOINT=https://hf-mirror.com
# Terminal 1: Launch sim
uv run sim2real/sim_env/base_sim.py --robot g1
# Terminal 2: Launch policy
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

### Keyboard Controls

After both processes are up, press `]` in the policy terminal to start. Open the mjviser URL printed by `base_sim.py`, then use the Elastic Band controls in the viewer UI to disable or tune the virtual gantry.

Keys are read in the policy (`tracking.py`) terminal:

| Key | Function |
|-----|----------|
| `i` | Init mode: interpolate joints from the current pose to the default pose (~10 s ramp) |
| `o` | Zero mode: hold the current joint positions |
| `]` | Policy mode: reset and start RL policy inference |
| `Space` | Pause / resume reference motion playback (starts paused; `npz` motion backend only) |

## Migrating to sim2real

This repo includes a Codex skill for adapting policies trained in external codebases into `sim2real`:

```text
skills/adapt-policy-to-sim2real
```

Converted checkpoints are distributed through the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder.

Currently supported adapted checkpoint families:

- BFM-Zero: `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml`
- HEFT: `checkpoints/heft/pmg/policy.yaml`, `checkpoints/heft/compliance/policy.yaml`
- Humanoid-GPT: `checkpoints/humanoid-gpt/policy.yaml`
- SONIC G1: `checkpoints/sonic/g1/policy.yaml`
- SONIC SMPL: `checkpoints/sonic/smpl/policy.yaml`
- TeleopIT: `checkpoints/teleopit/policy.yaml`
- TWIST2: `checkpoints/twist2/policy.yaml`

## Next Steps

- [Docs Home](./docs/README.md)
- [Getting Started](./docs/getting-started/README.md)
- [Offline Motion Tracking Tutorial](./docs/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation Tutorial](./docs/tutorials/pico-teleoperation.md)
