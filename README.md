# sim2real

Chinese version: [README_zh.md](./README_zh.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

If you're looking for the HDMI deployment stack, go to [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi).

## Quick Start

```bash
uv sync
```

Run offline motion tracking (sim2sim):

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

After both processes are up, press `]` in the policy terminal to start, then press `9` in the MuJoCo viewer to disable the virtual gantry.

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

Converted SONIC checkpoints are available on Google Drive:
[SONIC checkpoints](https://drive.google.com/drive/folders/1KgkLnfRzth2ZHMq8I49qpCSbY977iUAK).

Converted TWIST2 checkpoints are available on Google Drive:
[TWIST2 checkpoints](https://drive.google.com/drive/folders/14vXXgymYgnh2pXcaCzJQtcxm0FI3ez3j).

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
