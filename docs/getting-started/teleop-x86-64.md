---
title: Teleop Project (x86_64 PC)
sidebar_position: 3
---

Use `venv/teleop` for Pico / XR body tracking, realtime retarget inspection, and motion recording on a laptop or desktop.

## Setup

```bash
uv --project venv/teleop sync
```

### Install XRoboToolkit PC Service

Download the `.deb` from:

[https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)

```bash
sudo apt install -y ./XRoboToolkit_PC_Service_*.deb
```

Start `XRoboToolkit` / `XRobot` from the desktop or app launcher after installation.

### Clone the Pico SDK repos

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

### Build and install `xrobotoolkit_sdk`

```bash
bash scripts/setup/setup_xrobot_pybind.sh
```

The script expects the two repos above to exist under `external/`.

### Enable Pico streaming

1. Wear the leg trackers.
2. Finish whole-body tracking calibration.
3. Open `XRoboToolkit`, connect to this machine, and enable whole-body streaming.

## Verify Installation

Start the live retarget publisher:

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

In a second terminal, run the realtime viewer:

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://127.0.0.1:28701 \
  --viewer_hz 50
```

If the viewer updates with live G1 retargeted motion, the teleop stack is ready.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
- [Motion Recording](/tutorials/motion-recording)
