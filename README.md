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

HDMI is a framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations. This repository contains the official sim2sim and sim2real code for **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.

## Environment Setup

```bash
uv sync
source .venv/bin/activate
```

If you prefer conda
```bash
conda create -n hdmi python=3.12
conda activate hdmi
pip install -e .
```

### FAQ

For unitree_sdk_python installation, if missing CYCLONEDDS, refer to https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq

```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

This error mentions that the cyclonedds path could not be found. First compile and install cyclonedds:
```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```
Then run the above setup command (uv or conda) again.


## Sim2Sim
The sim2sim setup runs a MuJoCo environment and a reinforcement-learning policy as two Python processes that communicate over ZMQ. After both processes are up, press `]` in the policy terminal to start, then press `9` in the MuJoCo viewer to disable the virtual gantry immediately.


```bash
# 1) Start sim
uv run sim2real/sim_env/base_sim.py --robot_config config/robot/g1.yaml --scene_config config/scene/g1_29dof_rubberhand.yaml
# 2) Start policy
uv run sim2real/rl_policy/tracking.py --robot_config ./config/robot/g1.yaml --policy_config checkpoints/lafan-aa/policy-kl5hrst6-final.yaml
```

To deploy the sim2sim policy on the real robot, replace the sim script with 

```bash
# 1) Start real bridge
uv run scripts/real_bridge.py
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EGalahad/sim2real&type=date&legend=top-left)](https://www.star-history.com/#EGalahad/sim2real&type=date&legend=top-left)
