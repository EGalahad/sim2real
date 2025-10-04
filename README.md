# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

<div align="center">
<a href="https://hdmi-humanoid.github.io/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=GvIBzM7ieaA&list=PL0WMh2z6WXob0roqIb-AG6w7nQpCHyR0Z&index=12">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/pdf/2509.16757">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/EGalahad/sim2real/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/EGalahad/sim2real?style=social"/>
</a>
</div>

HDMI is a novel framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations.

This repository contains the official sim2sim and sim2real code of **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.

## TODO
- [x] Release pretrained models
- [x] Release sim2real code
- [x] Release sim2sim instructions
- [ ] Release sim2real instructions

## 🚀 Quickstart

```bash
conda create -n sim2real python=3.12
conda activate sim2real
pip install -r requirements.txt
```

## Sim2Sim Testing

The sim2sim environment consists of a MuJoCo simulation environment and a reinforcement learning policy, which is two python process that communicate through ZMQ.
After starting both processes, press ']' in the terminal running the policy to start the policy.
After the policy starts, press '9' in the mujoco viewer to disable a virtual gantry.

Test move suitcase.

```bash
# start mujoco
python sim_env/hdmi.py --robot_config config/robot/g1.yaml --scene_config config/scene/g1_29dof_rubberhand-suitcase.yaml
# start policy
python rl_policy/tracking.py --robot_config ./config/robot/g1.yaml --policy_config checkpoints/G1TrackSuitcase/policy-v55m8a23-final.yaml
```

Test open door.

```bash
python sim_env/hdmi.py --robot_config config/robot/g1.yaml --scene_config config/scene/g1_29dof_rubberhand-door.yaml
python rl_policy/tracking.py --robot_config ./config/robot/g1.yaml --policy_config checkpoints/G1PushDoorHand/policy-xg6644nr-final.yaml
```

Test roll ball.
```bash
python sim_env/hdmi.py --robot_config config/robot/g1.yaml --scene_config config/scene/g1_29dof_rubberhand-ball.yaml
python rl_policy/tracking.py --robot_config ./config/robot/g1.yaml --policy_config checkpoints/G1RollBall/policy-yte3rr8b-final.yaml 
```

## Sim2Real

### ONNX Inference Testing
```bash
python scripts/test_onnx_inference.py --policy_config checkpoints/G1TrackSuitcase/policy-v55m8a23-final.yaml
```

TODO...