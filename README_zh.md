# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/teleop`。

English version: [README.md](./README.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

如果你在找 HDMI 的部署栈，请看 [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi)。

## 快速开始

```bash
uv sync
```

运行离线动作跟踪（sim2sim）：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

## 录制和回放 Motion

在 root project 里把 Pico / XR retarget 后的 motion 录成 any4hdmi dataset：

```bash
uv run scripts/record_motion.py --connect tcp://127.0.0.1:28701
```

用 any4hdmi viewer wrapper 回放录好的 qpos motion：

```bash
uv run scripts/view_motion.py --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz
```

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
skills/adapt-policy-to-sim2real
```

已经转好的 SONIC checkpoints 放在 Google Drive：
[SONIC checkpoints](https://drive.google.com/drive/folders/1KgkLnfRzth2ZHMq8I49qpCSbY977iUAK)。

已经转好的 TWIST2 checkpoints 放在 Google Drive：
[TWIST2 checkpoints](https://drive.google.com/drive/folders/14vXXgymYgnh2pXcaCzJQtcxm0FI3ez3j)。

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
- [Motion Recording 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/motion-recording)
