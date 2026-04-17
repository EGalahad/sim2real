# Teleop Instructions

Chinese version: [README_zh.md](./README_zh.md)

## Setup

### 1. uv

```bash
uv --project venv/teleop sync
```

Prebuilt JetPack 5 packages are available here:

<https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e?usp=sharing>

Download and extract them at the repo root so the `prebuilt/` paths below exist.

### 2. xrobot app

On laptop / desktop, download the `.deb` from <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases> and install it:

```bash
sudo apt install -y ./XRoboToolkit_PC_Service_*.deb
```

On laptop, start from the desktop/app icon `XRoboToolkit` / `XRobot`.

On G1 onboard Orin (`aarch64`, Ubuntu 20.04), install the repo-provided prebuilt package:

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

On onboard Orin, start with `bash /opt/apps/roboticsservice/runService.sh`.

### 3. xrobotoolkit_sdk

#### Clone

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

##### Additionnal steps for onboard Orin

Switch the SDK repo to `orin`:

```bash
(cd external/XRoboToolkit-PC-Service && git checkout orin)
```

On onboard Orin / JetPack 5, the upstream aarch64 gRPC package can be
incompatible with Ubuntu 20.04. Build a JetPack 5 compatible package first as
described in [docs/xrobot_grpc_jetpack5.md](../../docs/xrobot_grpc_jetpack5.md).

Replace the upstream aarch64 gRPC directory with the repo-provided package:

```bash
export sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

##### Build and copy

For `amd64` / `x86_64`:

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

For `aarch64`:

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

Check files for `amd64` / `x86_64`:

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so
```

Check files for `aarch64`:

```bash
ls -l external/XRoboToolkit-PC-Service-Pybind/include/aarch64/PXREARobotSDK.h
ls -ld external/XRoboToolkit-PC-Service-Pybind/include/aarch64/nlohmann
ls -l external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
ldd external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so
```

##### Install python package

```bash
export pybind11_DIR=$(uv --project venv/teleop run python -c "import pybind11; print(pybind11.get_cmake_dir())")
uv --project venv/teleop pip uninstall --python venv/teleop/.venv/bin/python xrobotoolkit_sdk
uv --project venv/teleop pip install --python venv/teleop/.venv/bin/python -e external/XRoboToolkit-PC-Service-Pybind
```

### 4. pico

1. Wear the leg trackers.
2. Finish whole-body tracking calibration (optionally ground floor calibration).
3. Open `XRoboToolkit` and enter the laptop or onboard IP and connect.
4. Enable whole-body streaming.

### 5. validate your setup

Import check:

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

Live XR check:

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

Stop with `Ctrl-C`.

### 2. benchmark

```bash
uv --project venv/teleop run sim2real/teleop/benchmark_smplx_retarget.py \
  --input sim2real/teleop/xrobot_smplx_20260321_000000.npz \
  --actual_human_height 1.80 \
  --warmup_frames 10
```
