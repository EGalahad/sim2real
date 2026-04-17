# Teleop 使用说明

English version: [README.md](./README.md)

## Setup

### 1. uv

```bash
uv --project venv/teleop sync
```

### 2. xrobot app

在 laptop / desktop 上，从 <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases> 下载 `.deb`，安装：

```bash
sudo apt install -y ./XRoboToolkit_PC_Service_*.deb
```

在 laptop 上，从桌面 / 应用列表里的 `XRoboToolkit` / `XRobot` 图标启动。

在 G1 机载 Orin（`aarch64`, Ubuntu 20.04）上，从 repo 根目录安装仓库里带的包：

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

在 onboard Orin 上，用 `bash /opt/apps/roboticsservice/runService.sh` 启动。

### 3. xrobotoolkit_sdk

GMR 会通过 `uv --project venv/teleop sync` 安装，不需要手动 clone。

如果你之前把 SDK 仓库 clone 在 `sim2real/teleop/` 下面，先迁移到 `external/`:

```bash
mkdir -p external
mv sim2real/teleop/XRoboToolkit-PC-Service-Pybind external/XRoboToolkit-PC-Service-Pybind
mv external/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service external/XRoboToolkit-PC-Service
```

clone:

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
mkdir -p external/XRoboToolkit-PC-Service-Pybind/include
mkdir -p external/XRoboToolkit-PC-Service-Pybind/lib
```

如果是在 onboard Orin，上游 SDK 仓库切到 `orin`:

```bash
(cd external/XRoboToolkit-PC-Service && git checkout orin)
```

如果是在 onboard Orin / JetPack 5，上游 aarch64 gRPC 可能和 Ubuntu 20.04
不兼容。先按 [docs/xrobot_grpc_jetpack5.md](../../docs/xrobot_grpc_jetpack5.md)
里的说明准备仓库内的 JetPack 5 兼容 gRPC 包，再替换上游目录：

```bash
export xrobot_root=external/XRoboToolkit-PC-Service
export sdk_grpc="$xrobot_root/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

`amd64` / `x86_64` 的 build 和 copy:

```bash
(cd external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK && bash build.sh)

cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h \
  external/XRoboToolkit-PC-Service-Pybind/include/
rm -rf external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
cp -r external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann \
  external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
cp external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so \
  external/XRoboToolkit-PC-Service-Pybind/lib/
```

`aarch64`，例如 G1 机载 Orin 或通过 SSH 连到这台机器时的 build 和 copy:

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

`amd64` / `x86_64` 检查文件:

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
```

`aarch64` 检查文件:

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/aarch64/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/aarch64/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
```

安装 python 包:

```bash
export pybind11_DIR=$(uv --project venv/teleop run python -c "import pybind11; print(pybind11.get_cmake_dir())")
uv --project venv/teleop pip uninstall --python venv/teleop/.venv/bin/python xrobotoolkit_sdk
uv --project venv/teleop pip install --python venv/teleop/.venv/bin/python \
  -e external/XRoboToolkit-PC-Service-Pybind
```

### 4. pico

1. 戴好腿部 trackers。
2. 完成 whole-body tracking 校准（可选做 ground floor calibration）。
3. 打开 `XRoboToolkit`，输入 laptop 或 onboard 的 IP 并连接。
4. 打开 whole-body streaming。

### 5. validate your setup

下面命令都使用 `venv/teleop`。

先检查 import:

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

再检查实时 XR 数据:

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
  --actual_human_height 1.70
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
  --actual_human_height 1.70
```

用 `Ctrl-C` 结束。

### 2. benchmark

```bash
uv --project venv/teleop run sim2real/teleop/benchmark_smplx_retarget.py \
  --input sim2real/teleop/xrobot_smplx_20260321_000000.npz \
  --actual_human_height 1.70 \
  --warmup_frames 10
```
