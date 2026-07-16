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
uv sync
```

Run offline motion tracking (sim2sim):

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

After both processes are up, press `]` in the policy terminal to start. Open the mjviser URL printed by `base_sim.py`, then use the Elastic Band controls in the viewer UI to disable or tune the virtual gantry.

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
