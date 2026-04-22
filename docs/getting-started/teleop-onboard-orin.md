---
title: Teleop Project (Onboard Orin)
sidebar_position: 4
---

Use this path when Pico / XR tooling runs directly on the G1 onboard Orin. The root project still provides the policy runtime and `scripts/real_bridge.py`.

## Setup

```bash
uv --project venv/teleop sync
```

Download the [JetPack 5 prebuilt package bundle](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e?usp=sharing) and extract it at the repo root so `prebuilt/` exists.

### Install XRoboToolkit PC Service

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

Start the service:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

### Clone the Pico SDK repos

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
git -C external/XRoboToolkit-PC-Service checkout orin
```

### Replace the upstream aarch64 gRPC package

Build the JetPack 5 compatible package first by following [XRobot gRPC JetPack 5](/reference/xrobot-grpc-jetpack5), then replace the upstream directory:

```bash
export sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

### Build and install `xrobotoolkit_sdk`

```bash
bash scripts/setup/setup_xrobot_pybind.sh --arch aarch64
```

## Verify Installation

Start the live retarget publisher on the G1 onboard Orin:

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

In a second terminal, run the realtime viewer and point it at the G1 Orin IP:

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://<g1-orin-ip>:28701 \
  --viewer_hz 50
```

If the viewer updates with live G1 retargeted motion, the onboard teleop stack is ready.

## Notes

- Run `uv sync` in the repo root as well when the onboard machine also runs the policy and bridge processes.
- When the Pico publisher runs on a separate PC, point policy-side ZMQ arguments at that machine's IP.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
