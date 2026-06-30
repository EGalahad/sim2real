# Pico Teleoperation

这个教程使用 teleop publisher 提供实时 Pico / XR retarget，用它内置的 mjviser server 检查 retarget 结果，再用 root project 的 tracking policy 做执行。

## 1. 启动 Pico retarget publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py
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
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion-backend zmq \
  --controller pico
```

### Sim2Real

上真机前，先在 [Robot I/O](../reference/robot-io.md) 里选择部署路径。Pico 相关的 policy 参数保持一样：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --motion-backend zmq \
  --controller pico
```

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Pico 按键

- 按 `A` 进入 init pose。
- 同时按 `A` + `B` 进入 policy mode。
- 按 `X` 解除 motion flow 暂停。

## Notes

- `pico_retarget_pub.py` 发布实时 motion stream 给 tracking policy 使用，并自己创建 retarget mjviser server
- `sim2real/sim_env/base_sim.py` 是 sim2sim 的执行后端
- 真机部署时，[Robot I/O](../reference/robot-io.md) 里列出了 inline 和 bridge 两类方式
- 如果 publisher 和 policy 跑在不同机器上，再加 `--motion-zmq-connect tcp://<publisher_ip>:28701`

## Next Steps

- [Motion Recording](./motion-recording.md)
