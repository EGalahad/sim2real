# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/teleop`。

English version: [README.md](./README.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

如果你在找 HDMI 的部署栈，请看 [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi)。

## Runtime Artifacts

大文件不放在 git 里。先从共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
下载，把 `checkpoints/` 和 `third_party/` 放到 repo 根目录。

目录结构和 onboard 依赖说明见 [Download Artifacts](./docs/artifacts.md)。

## 快速开始

```bash
# 根据推理设备选择依赖组
uv sync --group cpu     # Sim2sim
uv sync --group g1      # G1 机器人
uv sync --group g1-gpu  # G1 机器人（GPU 推理）
```

运行离线动作跟踪（sim2sim）：

```bash
# 使用镜像加速（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com
# 终端 1：启动仿真
uv run sim2real/sim_env/base_sim.py --robot g1
# 终端 2：启动 policy
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

### 键盘控制

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

按键在 policy（`tracking.py`）终端里读取：

| 按键 | 功能 |
|------|------|
| `i` | Init 模式：关节从当前姿态插值到默认姿态（约 10 s 缓动） |
| `o` | Zero 模式：保持当前关节位置 |
| `]` | Policy 模式：reset 并开始 RL policy 推理 |
| `Space` | 暂停 / 恢复参考动作播放（启动时为暂停状态；仅 `npz` motion backend 有效） |

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
skills/adapt-policy-to-sim2real
```

已经转好的 checkpoints 统一放在共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录里。

目前已经支持的 adapted checkpoint：

- BFM-Zero: `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml`
- HEFT: `checkpoints/heft/pmg/policy.yaml`, `checkpoints/heft/compliance/policy.yaml`
- Humanoid-GPT: `checkpoints/humanoid-gpt/policy.yaml`
- SONIC G1: `checkpoints/sonic/g1/policy.yaml`
- SONIC SMPL: `checkpoints/sonic/smpl/policy.yaml`
- TeleopIT: `checkpoints/teleopit/policy.yaml`
- TWIST2: `checkpoints/twist2/policy.yaml`

安装到本机 Codex skills 目录：

```bash
mkdir -p ~/.codex/skills
cp -r skills/adapt-policy-to-sim2real ~/.codex/skills/
```

安装后重新打开一个 Codex session，即可通过 policy adaptation 相关请求触发；也可以显式提到 `adapt-policy-to-sim2real`。

## 下一步

- [文档首页](https://egalahad.github.io/sim2real/zh-Hans/)
- [快速上手](https://egalahad.github.io/sim2real/zh-Hans/getting-started/overview)
- [Root Project Setup](https://egalahad.github.io/sim2real/zh-Hans/getting-started/root-project)
- [离线动作跟踪教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/offline-motion-tracking)
- [Pico Teleoperation 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/pico-teleoperation)
