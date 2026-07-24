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

运行离线动作跟踪（sim2sim）：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
skills/adapt-policy-to-sim2real
```

已经转好的 checkpoints 统一放在共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录里。

目前已经支持的 adapted / distributed checkpoint：

| Policy family | Config path(s) | 说明 |
| --- | --- | --- |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Latent-conditioned motion tracker。 |
| HEFT | `checkpoints/heft/pmg/policy.yaml`, `checkpoints/heft/compliance/policy.yaml` | PMG 和 compliance 两个版本。 |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | 使用官方未修改 ONNX：[HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx)，下载后放到 `checkpoints/holomotion/v1_4_0/policy.onnx`。 |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Humanoid-GPT policy wrapper。 |
| Mimic-Lite | `checkpoints/mimic-lite/4x8192-large/policy.yaml`, `checkpoints/mimic-lite/8x8192-huge/policy.yaml`, `checkpoints/mimic-lite/32x8192-huge/policy.yaml` | Native mimic-lite tracking checkpoints。 |
| SONIC release | `checkpoints/sonic/release/g1/policy.yaml`, `checkpoints/sonic/release/smpl/policy.yaml` | Release G1 和 SMPL encoder variants。 |
| SONIC low-latency | `checkpoints/sonic/low_latency/g1/policy.yaml`, `checkpoints/sonic/low_latency/smpl/policy.yaml` | Low-latency G1 和 SMPL variants。 |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | TeleopIT policy wrapper。 |
| TWIST2 | `checkpoints/twist2/policy.yaml` | TWIST2 policy wrapper。 |
| 2026-07-23 deploy pipeline | `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/g1_8gpu/ppo/policy.yaml`, `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/g1_8gpu/ppo_roa_student/policy.yaml`, `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/g1_32gpu/ppo/policy.yaml`, `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/g1_32gpu/ppo_roa_student/policy.yaml`, `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/rp1_24dof/ppo/policy.yaml`, `checkpoints/deploy_2026_07_23/huge_mixture_pipeline/rp1_24dof/ppo_roa_student/policy.yaml` | G1/RP1 PPO 和 ROA student deploy exports。 |

## 真机环境

机器人 SDK 不安装进通用 root 环境。G1 inline 部署使用
`uv sync --extra inference-cpu --extra robot-g1`。安装与部署命令见
[Robot I/O 模式](./docs/robot_io.md)。

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
