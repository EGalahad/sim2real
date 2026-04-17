# Teleop 使用说明

English version: [README.md](./README.md)

## Setup

### 1. uv

```bash
uv --project venv/teleop sync
```

JetPack 5 的预编译包在这里：

<https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e?usp=sharing>

下载后解压到 repo 根目录，确保下面的 `prebuilt/` 路径存在。

### 2. xrobot app

在 laptop / desktop 上，从 <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases> 下载 `.deb` 并安装：

```bash
sudo apt install -y ./XRoboToolkit_PC_Service_*.deb
```

在 laptop 上，从桌面 / 应用列表里的 `XRoboToolkit` / `XRobot` 图标启动。

在 G1 机载 Orin（`aarch64`, Ubuntu 20.04）上，安装仓库里提供的预编译包：

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

在 onboard Orin 上，用 `bash /opt/apps/roboticsservice/runService.sh` 启动。

### 3. xrobotoolkit_sdk

#### Clone

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

##### Orin 额外步骤

把 SDK 仓库切到 `orin`：

```bash
(cd external/XRoboToolkit-PC-Service && git checkout orin)
```

在 onboard Orin / JetPack 5 上，上游 aarch64 gRPC 可能和 Ubuntu 20.04
不兼容。先按 [docs/xrobot_grpc_jetpack5.md](../../docs/xrobot_grpc_jetpack5.md)
准备 JetPack 5 兼容包，再用仓库里的预编译版本替换上游目录：

```bash
export sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

##### Build 和 copy

`amd64` / `x86_64`：

```bash
(cd external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK && bash build.sh)

mkdir -p external/XRoboToolkit-PC-Service-Pybind/include
mkdir -p external/XRoboToolkit-PC-Service-Pybind/lib

cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h \
  external/XRoboToolkit-PC-Service-Pybind/include/
rm -rf external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
cp -r external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann \
  external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so \
  external/XRoboToolkit-PC-Service-Pybind/lib/
```

`aarch64`：

```bash
(cd external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK && bash build.sh)

mkdir -p external/XRoboToolkit-PC-Service-Pybind/include/aarch64
mkdir -p external/XRoboToolkit-PC-Service-Pybind/lib/aarch64

cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h \
  external/XRoboToolkit-PC-Service-Pybind/include/aarch64/
rm -rf external/XRoboToolkit-PC-Service-Pybind/include/aarch64/nlohmann
cp -r external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann \
  external/XRoboToolkit-PC-Service-Pybind/include/aarch64/nlohmann
cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so \
  external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/
```

检查 `amd64` / `x86_64` 文件：

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
```

检查 `aarch64` 文件：

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/aarch64/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/aarch64/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
```

##### 安装 Python 包

```bash
export pybind11_DIR=$(uv --project venv/teleop run python -c "import pybind11; print(pybind11.get_cmake_dir())")
uv --project venv/teleop pip uninstall --python venv/teleop/.venv/bin/python xrobotoolkit_sdk
uv --project venv/teleop pip install --python venv/teleop/.venv/bin/python -e external/XRoboToolkit-PC-Service-Pybind
```

### 4. pico

1. 戴好腿部 trackers。
2. 完成 whole-body tracking 校准（可选做 ground floor calibration）。
3. 打开 `XRoboToolkit`，输入 laptop 或 onboard 的 IP 并连接。
4. 打开 whole-body streaming。

### 5. validate your setup

先检查 import：

```bash
uv --project venv/teleop run python - <<'PY'
import torch
import general_motion_retargeting
import xrobotoolkit_sdk
import zmq
from loop_rate_limiters import RateLimiter
print("general_motion_retargeting: OK")
print("xrobotoolkit_sdk: OK")
print("pyzmq: OK")
print("loop_rate_limiters: OK")
PY
```

再检查实时 XR 数据：

```bash
uv --project venv/teleop run python - <<'PY'
import xrobotoolkit_sdk as xrt

xrt.init()
print("Body data available:", xrt.is_body_data_available())
print("Headset pose:", xrt.get_headset_pose())
print("Left controller pose:", xrt.get_left_controller_pose())
print("Right controller pose:", xrt.get_right_controller_pose())
xrt.close()
PY
```

## Run

### Publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

### Viewer

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://127.0.0.1:28701 \
  --viewer_hz 50
```

## Offline Retargeting

### 1. record smplx

```bash
uv --project venv/teleop run sim2real/teleop/record_smplx.py \
  --output sim2real/teleop/xrobot_smplx_$(date +%Y%m%d_%H%M%S).npz \
  --sample_fps 30 \
  --actual_human_height 1.80
```

用 `Ctrl-C` 结束。

### 2. benchmark

```bash
uv --project venv/teleop run sim2real/teleop/benchmark_smplx_retarget.py \
  --input sim2real/teleop/xrobot_smplx_20260321_000000.npz \
  --actual_human_height 1.80 \
  --warmup_frames 10
```
