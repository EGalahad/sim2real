# Root Project

root project 用来跑 inference、tracking policy、MuJoCo simulation，以及 robot I/O。

## Setup

```bash
uv sync
```

如果这台机器要跑 `--robot-io inline` 或 `scripts/real_bridge_cpp.py`，还需要安装
G1 dependency group：

```bash
uv sync --group g1
```

如果 `unitree_sdk2py` setup 找不到 `cyclonedds`，可以参考上游 FAQ：

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

常见报错：

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

先编译并安装 CycloneDDS：

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

然后重新执行环境 setup。

## Verify Installation

### Test ankle swing

```bash
uv run scripts/ankle_swing.py
```

### Test inference time

```bash
uv run scripts/test_policy_inference.py \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
  --inference_backend onnx-cpu
```

## Next Steps

- [Offline Motion Tracking](../tutorials/offline-motion-tracking.md)
- [Robot I/O](/reference/robot-io)
