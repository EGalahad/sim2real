# Getting Started

Chinese version: [../zh_CN/getting-started/README.md](../zh_CN/getting-started/README.md)

`sim2real` is split into two environments:

- The root project is for inference, tracking policy, MuJoCo simulation, and the real bridge.
- `venv/teleop` is for Pico / XR retargeting, realtime viewing, and motion recording.

This project supports two hardware layouts:

- PC (`x86_64`) running the pipeline while controlling G1 over Ethernet.
- G1 onboard Orin running entire pipeline.

## Next Steps

- Use [Root Project](./root-project.md) if you only need policy, sim2sim, or the real bridge runtime.
- Use [Teleop Project (x86_64 PC)](./teleop-x86-64.md) if Pico / XR tools run on a laptop or desktop.
- Use [Teleop Project (Onboard Orin)](./teleop-onboard-orin.md) if teleop tooling runs on the robot.
