# Pico Teleoperation

这个教程使用 teleop publisher 提供实时 Pico / XR retarget，用它内置的 mjviser server 检查 retarget 结果，再用 root project 的 tracking policy 做执行。

## 1. 启动 Pico retarget publisher

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py
```

打开 publisher 打印出来的 mjviser URL。先确认 viewer 里的 G1 retarget 动作是对的，再继续执行。

## 2. 选择执行后端

### Sim2Sim

启动 MuJoCo 执行进程：

```bash
uv run sim2real/sim_env/base_sim.py
```

在另一个终端，把 tracking policy 接到实时 motion stream：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

### Sim2Real

上真机前，先在 [Robot I/O](/reference/robot-io) 里选择部署路径。Pico 相关的 policy 参数保持一样：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Pico 按键

- 按 `A` 进入 init pose。
- 同时按 `A` + `B` 进入 policy mode。
- 按 `X` 解除 motion flow 暂停。

## DH116S 灵巧手控制

Pico publisher 还会把左右扳机的连续值归一化为 `[0,1]` 的手部 grip command，
并通过 TCP `5593` 端口发送。左扳机控制左侧 DH116S，右扳机控制右侧 DH116S。

在连接了 DH116S CANFD 适配器的电脑上，先把 SDK 安装到当前 repo：

```bash
./third_party/dh116s_sdk/install.sh
uv sync --project venv/dh116s
```

安装脚本会自动识别 `aarch64` 或 `x86_64`，并把运行文件放到
`third_party/dh116s_sdk/python`。然后使用根项目环境启动：

```bash
uv run --project venv/dh116s --no-sync scripts/dh116s_control.py \
  --hand-dir double \
  --connect tcp://<pico-publisher-ip>:5593
```

:::warning
硬件进程启动时会 enable 并自动 home 所有选中的手。启动前必须清空双手周围空间。
在 `double` 模式下，只要任意一只手初始化失败，进程就会断开并退出。
:::

默认硬件映射为左手 `can0` / node `1`，右手 `can1` / node `1`。如果只想验证
ZMQ 和最大 40% 的安全闭合映射，而不 import SDK 或移动硬件，运行：

```bash
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_control.py --dry-run --hand-dir double
```

grip stream 断流后，进程会保持最后一次下发的手部姿态。停止进程时只断开 SDK，
不会自动张开。分阶段 bring-up 时可使用 `--hand-dir left` 或 `--hand-dir right`。

## Notes

- `pico_retarget_pub.py` 发布实时 motion stream 给 tracking policy 使用，并自己创建 retarget mjviser server
- hand grip 使用独立的 ZMQ 端口，不会改变现有 Pico 按键协议
- `sim2real/sim_env/base_sim.py` 是 sim2sim 的执行后端
- 真机部署时，[Robot I/O](/reference/robot-io) 里列出了 inline 和 bridge 两类方式
- 如果 publisher 和 policy 跑在不同机器上，再加 `--motion-zmq-connect tcp://<publisher_ip>:28701`

## Next Steps

- [Motion Recording](./motion-recording.md)
