# sim2real

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
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

After both processes are up, press `]` in the policy terminal to start. Open the mjviser URL printed by `base_sim.py`, then use the Elastic Band controls in the viewer UI to disable or tune the virtual gantry.

## Record and Visualize Motion

Record retargeted Pico / XR motion into an any4hdmi dataset from the root project:

```bash
uv run scripts/record_motion.py --connect tcp://127.0.0.1:28701
```

Replay the recorded qpos motion with the any4hdmi viewer wrapper:

```bash
uv run scripts/view_motion.py --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz
```

## Migrating to sim2real

This repo includes a Codex skill for adapting policies trained in external codebases into `sim2real`:

```text
skills/adapt-policy-to-sim2real
```

Converted checkpoints are distributed through the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder.

Install it into your local Codex skills directory:

```bash
mkdir -p ~/.codex/skills
cp -r skills/adapt-policy-to-sim2real ~/.codex/skills/
```

After installation, start a new Codex session and use the skill by asking for policy adaptation work, or explicitly refer to `adapt-policy-to-sim2real`.

## Next Steps

- [Docs Home](./docs/README.md)
- [Getting Started](./docs/getting-started/README.md)
- [Offline Motion Tracking Tutorial](./docs/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation Tutorial](./docs/tutorials/pico-teleoperation.md)
- [Motion Recording Tutorial](./docs/tutorials/motion-recording.md)
