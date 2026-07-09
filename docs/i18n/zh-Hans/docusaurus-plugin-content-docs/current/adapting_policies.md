---
title: Adapting Policies
slug: /reference/adapting-policies
---

# Adapting Policies

大多数导出的 tracking policy 都使用同一套 sim2real runtime。只要这个 policy 已经在
`checkpoints/` 下面有 deploy YAML，部署命令通常保持不变，只需要把
`--policy-config` 换成对应 YAML。

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot-io inline \
  --motion-backend zmq \
  --controller pico \
  --policy-config checkpoints/heft/pmg/policy.yaml
```

这适用于普通 G1 tracking policy，也就是消费正常 G1 motion stream 的 policy，
例如 HEFT、TeleopIT、Humanoid-GPT，以及普通 any4hdmi / SONIC G1 motion policy。

## Policy 特殊运行条件

少数 adapted policy 需要不同的 motion source 或额外 runtime asset。

### BFM-Zero

BFM-Zero 的 motion observation 里用 MuJoCo FK，所以需要使用它 checkpoint 对应的
MJCF。direct NPZ playback 时，这个路径写在 policy YAML 里；如果通过 ZMQ publisher
发 motion，就把同一个 MJCF override 传给 publisher。

```bash
uv run sim2real/teleop/npz_pub.py \
  --motion_path ../any4hdmi/output/g1/lafan/motions/walk1_subject1.npz \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

### SONIC SMPL Mode

SONIC SMPL mode 不是普通的 G1 `motion_backend=zmq` stream。使用 SONIC SMPL
policy config，保留它的 `motion_backend: smpl_zmq` 设置；或者启动 policy 时显式传
`--motion-backend smpl_zmq`。这条链路需要 SMPL/XRobot publisher。

最简单的 sim2sim Pico 测试：

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py --publish-smpl
```

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sonic/smpl/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --controller pico
```

数据 contract 见 [SONIC SMPL Input](/reference/sonic-smpl-input)。

## 真机测试记录

当前 G1 真机测试记录：

- BFM-Zero 可以正常跑，但需要注意传 MJCF override。
- TeleopIT 走路表现不错，但观察到关节会有剧烈响动，双膝跪地目前不可靠；这块先按
  deploy infra / policy compatibility 问题处理。
- HEFT 有轻微响动，但整体 tracking 表现很好。
