---
title: Root Project
sidebar_position: 2
---

Use the root project for inference, tracking policy, MuJoCo simulation, and robot I/O.

## Setup

```bash
uv sync
```

If this machine will run `--robot-io inline` or `scripts/real_bridge_cpp.py`,
install the G1 dependency group as well:

```bash
uv sync --group g1
```

If `unitree_sdk2py` setup cannot locate `cyclonedds`, refer to the upstream
FAQ:

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

Typical error:

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

Compile and install CycloneDDS first:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

Then rerun your environment setup.

## Verify Installation

### Test ankle swing

```bash
uv run scripts/ankle_swing.py
```

### Test inference time

```bash
uv run scripts/test_policy_inference.py \
  --policy_config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --inference_backend onnx-cpu
```

## Next Steps

- [Offline Motion Tracking](/tutorials/offline-motion-tracking)
- [Robot I/O](/reference/robot-io)
