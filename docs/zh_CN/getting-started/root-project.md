# Root Project

English version: [../../getting-started/root-project.md](../../getting-started/root-project.md)

root project 用来跑 inference、tracking policy、MuJoCo simulation，以及 `scripts/real_bridge.py`。

## Setup

```bash
uv sync
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
  --policy_config checkpoints/lafan-aa/policy-33e770c9_lafan_finetune-final.yaml \
  --inference_backend onnx-cpu
```

## Next Steps

- [Offline Motion Tracking](../tutorials/offline-motion-tracking.md)
