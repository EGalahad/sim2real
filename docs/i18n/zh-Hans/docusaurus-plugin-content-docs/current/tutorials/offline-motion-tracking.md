# Offline Motion Tracking

这个教程使用 root project 里的 tracking policy 和离线动作参考。

## Sim2Sim

先启动 MuJoCo 执行进程。启动后终端会打印 mjviser URL：

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

两个进程都起来后，在 policy 终端按 `]` 开始跟踪。虚拟 gantry / elastic band 的开关和长度在 mjviser UI 里调。

## Integrated Sim2Sim

如果希望 policy 和 MuJoCo 在同一个进程里运行，用 integrated runner。它会立即加载 policy，把机器人设置到 motion 第一帧，等待 5 秒后开始跟踪；motion 结束后会停在最后一帧。这个 runner 默认关闭 elastic band，启动后也会打印 mjviser URL。

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy_config checkpoints/sonic_groot_6k/policy-sonic-groot-006000.yaml \
  --motion_path ../any4hdmi/output/sonic/motions/240529/macarena_001__A545.npz
```

非可视化运行加 `--headless`。有浏览器 client 连接时，mjviser scene 会每个 env step 更新一次。在 mjviser 模式里，停在最后一帧后点击 `Restart motion` 按钮会回到第一帧，并重新执行等待、跟踪、停在最后一帧的流程。

如果要做定量评测，可以加 `--trajectory-output <path>.npz` 保存完整轨迹，再用
`scripts/tracking_experiment/` 里的脚本计算动作进度、全局根部跟踪和局部身体跟踪指标。

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
