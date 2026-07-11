---
title: Run External Policies
slug: /tutorials/run-external-policies
---

# Run External Policies

sim2real 的模块化设计允许同一套 runtime 执行不同的 tracking policy，只要这个
policy 提供兼容的 deploy YAML 和 ONNX model。我们已经把几个外部工作转换成了这个
格式，所以多数情况下可以保持正常部署命令不变，只把 `--policy-config` 换成对应
policy YAML。

## 已转换 Checkpoints

先下载共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录，然后把下面任意 checkpoint 路径作为 `--policy-config` 使用。

| Policy | Checkpoint YAML | Notes |
| --- | --- | --- |
| HEFT PMG | `checkpoints/heft/pmg/policy.yaml` | 正常 G1 motion stream。 |
| HEFT Compliance | `checkpoints/heft/compliance/policy.yaml` | 正常 G1 motion stream；observation 里 compliance flag 固定为 off。 |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | 正常 G1 motion stream。 |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | 正常 G1 motion stream。 |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | ZMQ publisher 需要传 checkpoint 对应的 MJCF override。 |
| SONIC G1 | `checkpoints/sonic/g1/policy.yaml` | 正常 G1 motion stream。 |
| SONIC SMPL | `checkpoints/sonic/smpl/policy.yaml` | 使用 `motion_backend: smpl_zmq` 和 SMPL publisher。 |
| TWIST2 | `checkpoints/twist2/policy.yaml` | 正常 G1 motion stream。 |

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
