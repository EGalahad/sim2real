# Getting Started

开始前先克隆仓库：

```bash
git clone https://github.com/EGalahad/sim2real
```

`sim2real` 分成两个环境：

- root project 负责 policy inference、MuJoCo simulation，以及 `scripts/real_bridge.py`
- `venv/teleop` 负责 Pico / XR retarget、内置 mjviser viewing，以及 motion recording

当前支持两种硬件布局：

- PC (`x86_64`) 运行 teleop 工具，通过网线控制 G1
- G1 onboard Orin 本地运行整个 pipeline

## Runtime Architecture

policy runtime 和执行 backend 是解耦的。sim2sim 里 backend 是 MuJoCo；
sim2real 里同一个 policy 通过 real bridge 和真机通信。

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

- 上硬件前先选择 [Network Configuration](./network-configuration.md)
- 只需要 policy、sim2sim 或 real bridge 时，看 [Root Project](./root-project.md)
- Pico / XR 工具跑在 laptop / desktop 上时，看 [Teleop Project (x86_64 PC)](./teleop-x86-64.md)
- Pico / XR 工具跑在机载 Orin 上时，看 [Teleop Project (Onboard Orin)](./teleop-onboard-orin.md)
