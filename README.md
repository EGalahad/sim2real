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

## Setup

For inference / policy / sim, use the root project:

```bash
uv sync
```

For teleop, use `venv/teleop` and follow [sim2real/teleop/README.md](./sim2real/teleop/README.md).

### Validate Setup

1. Test ankle swing.

```bash
uv run scripts/ankle_swing.py
```

2. Test ONNX inference time.

```bash
uv run scripts/test_policy_inference.py --policy_config checkpoints/lafan-aa/policy-120e84e1_lafan_finetune-final.yaml --inference_backend onnx-cpu
```

3. Test PICO data reading, and use the viewer to check real-time G1 motion.

See [sim2real/teleop/README.md](./sim2real/teleop/README.md).

## Run Policy

### Sim2Sim
The sim2sim setup runs a MuJoCo environment and a reinforcement-learning policy as two Python processes that communicate over ZMQ. After both processes are up, press `]` in the policy terminal to start, then press `9` in the MuJoCo viewer to disable the virtual gantry immediately.

```bash
# 1) Run sim env
uv run sim2real/sim_env/base_sim.py --robot g1
# 2) Start policy
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/lafan-aa/policy-120e84e1_lafan_finetune-final.yaml
```

### Sim2Real
Run an additional real bridge:

```bash
# 1) Run additional real bridge
uv run scripts/real_bridge.py
# 2) Start policy
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/lafan-aa/policy-120e84e1_lafan_finetune-final.yaml
```

## Misc

Generate MuJoCo Python stubs 

```bash
uv run --with mypy stubgen -p mujoco -o stubs
```

Then add `./stubs` to VSCode settings "python.analysis.extraPaths"


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EGalahad/sim2real&type=date&legend=top-left)](https://www.star-history.com/#EGalahad/sim2real&type=date&legend=top-left)

## FAQ

### `uv sync` fails to fetch Git dependencies on onboard Orin

If the Orin system clock is wrong after boot (for example, it reset to 1970),
`uv sync` may fail while fetching Git dependencies from GitHub, sometimes with
TLS or other network-looking errors. Set the current time manually, then retry:

```bash
sudo date -s "2026-04-17 10:00:00"  # Replace with the current time
uv sync
```

If this keeps happening after reboot, check the device's time synchronization or
RTC configuration.

### Could not locate cyclonedds

For unitree_sdk_python installation, if missing CYCLONEDDS, refer to https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq

```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

This error means that the cyclonedds path could not be found. First compile and install cyclonedds:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

Then run the matching setup command above again, or repeat your conda setup.

### ImportError: cannot allocate memory in static TLS block

When running on onboard Orin, Python may fail to import a native library and
report that `libc10.so` or `libGLdispatch.so.0` cannot allocate memory in the
static TLS block.

This can happen when large native libraries such as PyTorch or OpenGL are loaded
after other dependencies have already consumed the available static thread-local
storage slots. Force the problematic library to load first with `LD_PRELOAD`.
For a `libGLdispatch.so.0` error on `aarch64`, run this before starting the
Python script:

```bash
export LD_PRELOAD=/home/elijah/sim2real/venv/teleop/.venv/lib/python3.10/site-packages/torch/lib/libtorch.so:/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
```

You can also try moving `import torch` to the first import in the Python script.
