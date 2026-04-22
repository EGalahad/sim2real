# Getting Started

English version: [../../getting-started/README.md](../../getting-started/README.md)

`sim2real` 分成两个环境：

- root project 负责 policy inference、MuJoCo simulation，以及 `scripts/real_bridge.py`
- `venv/teleop` 负责 Pico / XR retarget、realtime viewer，以及 motion recording

当前支持两种硬件布局：

- PC (`x86_64`) 运行 teleop 工具，通过网线控制 G1
- G1 onboard Orin 本地运行 teleop 工具，同时继续使用 root project 跑 policy 和 bridge runtime

## 选择你的路径

- 只需要 policy、sim2sim 或 real bridge 时，看 [Root Project](./root-project.md)
- Pico / XR 工具跑在 laptop / desktop 上时，看 [Teleop Project (x86_64 PC)](./teleop-x86-64.md)
- Pico / XR 工具跑在机载 Orin 上时，看 [Teleop Project (Onboard Orin)](./teleop-onboard-orin.md)

## Next Steps

- [Offline Motion Tracking](../tutorials/offline-motion-tracking.md)
- [Pico Teleoperation](../tutorials/pico-teleoperation.md)
- [Motion Recording](../tutorials/motion-recording.md)
