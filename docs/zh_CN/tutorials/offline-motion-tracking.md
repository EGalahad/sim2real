# Offline Motion Tracking

English version: [../../tutorials/offline-motion-tracking.md](../../tutorials/offline-motion-tracking.md)

这个教程使用 root project 里的 tracking policy 和离线动作参考。

## Sim2Sim

先启动 MuJoCo 执行进程：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

在第二个终端启动 tracking policy：

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

两个进程各自负责：

- `sim2real/sim_env/base_sim.py` 在 MuJoCo 里执行 `low_cmd`，并发布 `low_state`
- `sim2real/rl_policy/tracking.py` 消费 `low_state`，跑导出的 policy，再发出下一帧 `low_cmd`

两个进程都起来后，在 policy 终端按 `]` 开始跟踪，然后在 MuJoCo viewer 里按 `9` 关闭虚拟 gantry。

## Sim2Real

把 MuJoCo 执行进程换成 real bridge：

```bash
uv run scripts/real_bridge.py
```

在第二个终端运行同一个 tracking policy：

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

两个进程各自负责：

- `scripts/real_bridge.py` 把 Unitree DDS 的 `low_state` / `low_cmd` 接到统一 ZMQ runtime
- `sim2real/rl_policy/tracking.py` 在 sim2sim 和 sim2real 两种模式下保持不变

## Next Steps

- [Pico Teleoperation](./pico-teleoperation.md)
