# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/pico`。

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
uv sync --extra inference-cpu
```

在 G1 上安装或修复环境时，可以调用 repo 内置的 Codex skill
`$configure-g1-sim2real`；它位于 `.agents/skills/configure-g1-sim2real`。

运行离线动作跟踪（sim2sim）：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/mimic-lite/roa/policy.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
.agents/skills/adapt-policy-to-sim2real
```

已经转好的 checkpoints 统一放在共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录里。

目前已经支持的 adapted / distributed checkpoint：

| Policy family | Config path(s) | 说明 |
| --- | --- | --- |
| MimicLite-ROA | `checkpoints/mimic-lite/roa/policy.yaml` | 最新 16x16384 PPO-ROA student release。 |
| MimicLite-PPO | `checkpoints/mimic-lite/ppo/policy.yaml` | 最新 16x16384 Huge PPO release。 |
| HEFT | `checkpoints/heft` | PMG 和 compliance 两个版本。 |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | 使用官方未修改 ONNX：[HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx)，下载后放到 `checkpoints/holomotion/v1_4_0/policy.onnx`。 |
| SONIC release | `checkpoints/sonic/release` | Release G1 和 SMPL encoder variants。 |
| SONIC low-latency | `checkpoints/sonic/low_latency` | Low-latency G1 和 SMPL variants。 |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | 使用 heading-normalized reference orientation 的 G1 policy。 |
| GRIT v0.0.1 | `checkpoints/grit/v0_0_1/policy.yaml` | 九帧参考上下文和十帧本体感知历史。 |
| ScaleBFM | `checkpoints/scalebfm` | [WeishuaiZeng/ScaleBFM](https://huggingface.co/WeishuaiZeng/ScaleBFM) 的 Humanoid Transformer XL 和 M ONNX exports。 |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Latent-conditioned motion tracker。 |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | TeleopIT policy wrapper。 |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Humanoid-GPT policy wrapper。 |
| TWIST2 | `checkpoints/twist2/policy.yaml` | TWIST2 policy wrapper。 |

![统一的跨代码库动作跟踪评测](assets/mimic_lite_cross_codebase_tracking_eval.png)

图中使用 14 个 policy variants 的全新结果，数据集为 LAFAN-40、PHUMA-30
和清洗后的 Root-90。Root-90 每段沿标注的前进、后退或侧移方向持续运动，
root XY 位移为 1.5--3.0 m。

为了公平比较，我们报告每个 policy 所需的 motion-lookahead latency，并将
其定义为最远 future reference frame 对应的时间。所有数值均采用统一的
50 Hz reference-motion contract。

| Policy | MimicLite-ROA | MimicLite-PPO | HEFT | HoloMotion | SONIC | SONIC low-latency | SONIC v1.1 | GRIT v0.0.1 | ScaleBFM XL | ScaleBFM M | BFM-Zero | TeleopIT | Humanoid-GPT | TWIST2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motion-lookahead latency | 0.08 s | 0.08 s | 0.12 s | 0.20 s | 0.90 s | 0.18 s | 0.90 s | 0.26 s | 0.10 s | 0.10 s | 0.12 s | 0.00 s | 0.02 s | 0.00 s |

## 真机环境

机器人 SDK 不安装进通用 root 环境。G1 inline 部署使用
`uv sync --extra inference-cpu --extra robot-g1`。安装与部署命令见
[Robot I/O 模式](./docs/robot_io.md)。

Repo skills 统一放在 `.agents/skills/`，无需手动复制到
`~/.codex/skills/`。可以在 Codex 中显式调用
`$adapt-policy-to-sim2real`。

## 下一步

- [文档首页](https://egalahad.github.io/sim2real/zh-Hans/)
- [快速上手](https://egalahad.github.io/sim2real/zh-Hans/getting-started/overview)
- [Root Project Setup](https://egalahad.github.io/sim2real/zh-Hans/getting-started/root-project)
- [离线动作跟踪教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/offline-motion-tracking)
- [Pico Teleoperation 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/pico-teleoperation)

## Citation

如果 sim2real 对你的研究有所帮助，请引用：

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
