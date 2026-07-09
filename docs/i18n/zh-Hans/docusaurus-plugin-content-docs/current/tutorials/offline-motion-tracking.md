# Offline Motion Tracking

这个教程使用 root project 里的 tracking policy 和离线动作参考。

默认 motion：

```text
hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

## Sim2Sim

先启动 MuJoCo 执行进程。启动后终端会打印 mjviser URL：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

在第二个终端启动 tracking policy：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

两个进程各自负责：

- `sim2real/sim_env/base_sim.py` 在 MuJoCo 里执行 `low_cmd`，并发布 `low_state`
- `sim2real/rl_policy/tracking.py` 消费 `low_state`，跑导出的 policy，再发出下一帧 `low_cmd`

两个进程都起来后，在 policy 终端按 `]` 开始跟踪。虚拟 gantry / elastic band 的开关和长度在 mjviser UI 里调。

## Sim2Real

上真机前，先在 [Robot I/O](/reference/robot-io) 里选择部署路径。例如 tracking policy 仍然这样启动：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Integrated Sim2Sim

如果希望 policy 和 MuJoCo 在同一个进程里运行，用 integrated runner。它会立即加载 policy，把机器人设置到 motion 第一帧，等待 5 秒后开始跟踪；motion 结束后会停在最后一帧。这个 runner 默认关闭 elastic band，启动后也会打印 mjviser URL。

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

非可视化运行加 `--headless`。有浏览器 client 连接时，mjviser scene 会每个 env step 更新一次。在 mjviser 模式里，停在最后一帧后点击 `Restart motion` 按钮会回到第一帧，并重新执行等待、跟踪、停在最后一帧的流程。

如果要做定量评测，可以加 `--trajectory-output <path>.npz` 保存完整轨迹，再用
`scripts/tracking_experiment/` 里的脚本计算动作进度、全局根部跟踪和局部身体跟踪指标。

## Next Steps

- [Pico Teleoperation](./pico-teleoperation.md)
