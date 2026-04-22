# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/teleop`。

English version: [README.md](./README.md)
Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)
如果你在找 HDMI 的 deployment stack，请看 [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi)。

## Quick Start

```bash
uv sync
```

## Offline Motion Tracking (Sim2Sim)

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后在 MuJoCo viewer 里按 `9` 关闭虚拟 gantry。

## Next Steps

- [文档首页](./docs/zh_CN/README.md)
- [Getting Started](./docs/zh_CN/getting-started/README.md)
- [Root Project Setup](./docs/zh_CN/getting-started/root-project.md)
- [离线动作跟踪教程](./docs/zh_CN/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation 教程](./docs/zh_CN/tutorials/pico-teleoperation.md)
- [Motion Recording 教程](./docs/zh_CN/tutorials/motion-recording.md)
