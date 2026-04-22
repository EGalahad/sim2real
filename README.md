# sim2real

Chinese version: [README_zh.md](./README_zh.md)

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

## Next Steps

- [Docs Home](./docs/README.md)
- [Getting Started](./docs/getting-started/README.md)
- [Offline Motion Tracking Tutorial](./docs/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation Tutorial](./docs/tutorials/pico-teleoperation.md)
- [Motion Recording Tutorial](./docs/tutorials/motion-recording.md)
