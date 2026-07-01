---
title: Robot I/O
slug: /reference/robot-io
---

# Robot I/O 模式

`robot_io` 控制 policy 怎么读取机器人低层状态、发送低层命令。它和下面两个选项是分开的：

- `inference_backend`：控制 ONNX / TensorRT policy 推理后端。
- `motion_backend`：控制参考动作来源。

用 `sim_env/base_sim.py` 跑 sim2sim 时，policy 保持 `--robot-io zmq`。上真机时，在下面三种方式里选一种。

## 快速选择

| 模式 | 进程 | 什么时候用 |
| --- | --- | --- |
| `--robot-io inline` | 只跑 policy | 推荐的真机部署路径。policy 跑在有 `unitree_interface` 的机器上，少一跳 ZMQ bridge，延迟波动更小。 |
| `--robot-io zmq` + `scripts/real_bridge.py` | policy + Python DDS bridge | 需要原来的分进程 bridge，或者当前环境没有 `unitree_interface` 时用。bridge 基于 `unitree_sdk2py`。 |
| `--robot-io zmq` + `scripts/real_bridge_cpp.py` | policy + `unitree_interface` bridge | 想保留 ZMQ 分进程协议，但 bridge 侧使用 `unitree_interface` 机器人绑定时用。 |

:::tip
如果真机测试时关节剧烈抖动，或者高动态、高速度 motion 效果明显不好，优先怀疑
ZMQ I/O 延迟不稳定。建议先试 `--robot-io inline`，再考虑重新调 policy 或 PD 参数。
:::

## Inline

inline 模式会在 `BasePolicy` 里直接创建 Unitree robot object。policy loop 直接调用
`robot.read_low_state()` 和 `robot.write_low_command()`，不需要启动 real bridge。

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot-io inline \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

真机部署时如果关心延迟稳定性，优先用这个模式。只有机器人网卡不是默认 `eth0`
时，才额外加 `--robot-interface <robot_network_interface>`。

## ZMQ + `real_bridge.py`

这个模式把 policy 和 robot bridge 分成两个进程。bridge 使用 `unitree_sdk2py`，
把机器人状态发布成 ZMQ `low_state`，再把 ZMQ `low_cmd` 下发到机器人。

终端 1：

```bash
uv run scripts/real_bridge.py
```

终端 2：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

只有机器人网卡不是默认 `eth0` 时，才在 bridge 命令里加
`--interface <robot_network_interface>`。

## ZMQ + `real_bridge_cpp.py`

这个模式和 `real_bridge.py` 使用同一套 ZMQ 协议，但 bridge 里用 `unitree_interface`
做机器人 I/O。

终端 1：

```bash
uv run scripts/real_bridge_cpp.py
```

终端 2：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

只有机器人网卡不是默认 `eth0` 时，才在 bridge 命令里加
`--interface <robot_network_interface>`。

## 使用建议

正常真机部署优先用 `inline`。如果需要进程隔离、想分开 debug policy 和 robot bridge，
或者正在用 `base_sim.py` 跑 sim2sim，就用 ZMQ bridge 模式。
