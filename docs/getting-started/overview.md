---
title: Getting Started
sidebar_position: 1
---

Start by cloning the repository:

```bash
git clone https://github.com/EGalahad/sim2real
```

`sim2real` is split into two environments:

- The root project is for policy inference, MuJoCo simulation, and the real bridge.
- `venv/teleop` is for Pico / XR retargeting, realtime viewing, and motion recording.

This project supports two hardware layouts:

- PC (`x86_64`) running the pipeline while controlling G1 over Ethernet.
- G1 onboard Orin running the entire pipeline locally on the robot.

## Runtime Architecture

The policy runtime is decoupled from the execution backend. In sim2sim, the
backend is MuJoCo. In sim2real, the same policy talks to the robot through the
real bridge.

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
    Bridge["real bridge<br/>Unitree DDS <-> ZMQ"]
    Robot["robot<br/>Unitree G1"]

    Policy -- low_cmd --> Bridge
    Bridge -- low_state --> Policy
    Robot -- rt/lowstate --> Bridge
    Bridge -- rt/lowcmd --> Robot

    classDef policy fill:#ede9fe,stroke:#8b5cf6,color:#1f2937
    classDef bridge fill:#dcfce7,stroke:#22c55e,color:#1f2937
    classDef robot fill:#fef3c7,stroke:#f59e0b,color:#1f2937
    class Teleop,Policy policy
    class Bridge bridge
    class Robot robot
```

## Next Steps

- Choose a [Network Configuration](/getting-started/network-configuration) before running on hardware.
- Use [Root Project](/getting-started/root-project) if you only need policy, sim2sim, or the real bridge runtime.
- Use [Teleop Project (x86_64 PC)](/getting-started/teleop-x86-64) if Pico / XR tools run on a laptop or desktop.
- Use [Teleop Project (Onboard Orin)](/getting-started/teleop-onboard-orin) if teleop tooling runs on the robot.
