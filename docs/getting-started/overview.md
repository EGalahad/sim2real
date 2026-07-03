---
title: Getting Started
sidebar_position: 1
---

Start by cloning the repository:

```bash
git clone https://github.com/EGalahad/sim2real
```

`sim2real` is split into two environments:

- The root project is for policy inference, MuJoCo simulation, and robot I/O.
- `venv/teleop` is for Pico / XR retargeting with built-in mjviser viewing, and motion recording.

This project supports two hardware layouts:

- PC (`x86_64`) running the pipeline while controlling G1 over Ethernet.
- G1 onboard Orin running the entire pipeline locally on the robot.

## Runtime Architecture

The policy runtime is decoupled from the execution backend. In sim2sim, the
backend is MuJoCo. In sim2real, choose one of the [Robot I/O](/reference/robot-io)
modes for hardware.

### Sim2Sim

```mermaid
flowchart LR
    Teleop["teleop / live motion<br/>(optional)"] -. reference motion .-> Policy

    Policy["policy<br/>Tracking / BasePolicy"]
    Bridge["sim bridge<br/>SimulationBridge"]
    Mujoco["MuJoCo"]

    Policy -- low_cmd --> Bridge
    Bridge -- low_state --> Policy
    Mujoco -- sim state --> Bridge
    Bridge -- control --> Mujoco

    classDef policy fill:#ede9fe,stroke:#8b5cf6,color:#1f2937
    classDef bridge fill:#dcfce7,stroke:#22c55e,color:#1f2937
    classDef sim fill:#dbeafe,stroke:#3b82f6,color:#1f2937
    class Teleop,Policy policy
    class Bridge bridge
    class Mujoco sim
```

### Sim2Real

```mermaid
flowchart LR
    Teleop["teleop / live motion<br/>(optional)"] -. reference motion .-> Policy

    Policy["policy<br/>Tracking / BasePolicy"]
    RobotIO["robot I/O<br/>inline or bridge"]
    Robot["robot<br/>Unitree G1"]

    Policy -- command --> RobotIO
    RobotIO -- state --> Policy
    Robot -- low state --> RobotIO
    RobotIO -- low command --> Robot

    classDef policy fill:#ede9fe,stroke:#8b5cf6,color:#1f2937
    classDef bridge fill:#dcfce7,stroke:#22c55e,color:#1f2937
    classDef robot fill:#fef3c7,stroke:#f59e0b,color:#1f2937
    class Teleop,Policy policy
    class RobotIO bridge
    class Robot robot
```

## Next Steps

- Download runtime files from [Download Artifacts](/reference/artifacts).
- Choose a [Network Configuration](/getting-started/network-configuration) before running on hardware.
- Use [Root Project](/getting-started/root-project) if you only need policy, sim2sim, or robot I/O runtime.
- Choose the real-robot deploy path in [Robot I/O](/reference/robot-io).
- Use [Teleop Project (x86_64 PC)](/getting-started/teleop-x86-64) if Pico / XR tools run on a laptop or desktop.
- Use [Teleop Project (Onboard Orin)](/getting-started/teleop-onboard-orin) if teleop tooling runs on the robot.
