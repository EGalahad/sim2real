---
title: Teleop Project (Onboard Orin)
sidebar_position: 4
---

Use this path when Pico / XR tooling runs directly on the G1 onboard Orin. The root project still provides the policy runtime and `scripts/real_bridge.py`.

## Setup

```bash
uv --project venv/teleop sync
```

Choose the setup path that matches the onboard Orin image:

### JetPack 5

Download the [JetPack 5 prebuilt package bundle](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e?usp=sharing) and extract it at the repo root so `prebuilt/` exists.

Install `XRoboToolkit PC Service` from the prebuilt bundle:

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

### JetPack 6

If the onboard Orin is already on JetPack 6, for example after flashing Sonic, do not download the JetPack 5 prebuilt bundle.

Download the arm64 `.deb` for `XRoboToolkit PC Service` from:

[https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)

```bash
sudo apt install -y ./XRoboToolkit*.deb
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

These repos are used to build `xrobotoolkit_sdk`. They are separate from the installable `XRoboToolkit PC Service` `.deb`.

### JetPack 5 only: replace the upstream aarch64 gRPC package

Build the JetPack 5 compatible package first by following [XRobot gRPC JetPack 5](/reference/xrobot-grpc-jetpack5), then replace the upstream directory:

```bash
export sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

If the onboard Orin is on JetPack 6, skip this section and keep the upstream `linux_aarch64/grpc` directory.

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

To record and replay a qpos clip after the live publisher is running, use the root project environment:

```bash
uv run scripts/record_motion.py --connect tcp://<g1-orin-ip>:28701
uv run scripts/view_motion.py --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz
```

## Notes

- Run `uv sync` in the repo root as well when the onboard machine also runs the policy and bridge processes.
- When the Pico publisher runs on a separate PC, point policy-side ZMQ arguments at that machine's IP.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
