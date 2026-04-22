# Teleop Project (x86_64 PC)

`venv/teleop` 用来在 laptop / desktop 上跑 Pico / XR body tracking、realtime retarget 检查，以及 motion recording。

## Setup

```bash
uv --project venv/teleop sync
```

### 安装 XRoboToolkit PC Service

从下面的 release 页面下载 `.deb`：

[https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)

```bash
sudo apt install -y ./XRoboToolkit_PC_Service_*.deb
```

安装后，从桌面或应用列表启动 `XRoboToolkit` / `XRobot`。

### Clone Pico SDK 仓库

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

### Build 并安装 `xrobotoolkit_sdk`

```bash
bash scripts/setup/setup_xrobot_pybind.sh
```

这个脚本默认要求上面两个仓库已经位于 `external/` 下。

### 打开 Pico streaming

1. 戴好腿部 trackers。
2. 完成 whole-body tracking 校准。
3. 打开 `XRoboToolkit`，连接到当前机器。
4. 打开 whole-body streaming。

## Verify Installation

先启动 live retarget publisher：

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

在第二个终端里启动 realtime viewer：

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://127.0.0.1:28701 \
  --viewer_hz 50
```

如果 viewer 里能看到实时更新的 G1 retarget 动作，teleop 环境就通了。

## Next Steps

- [Pico Teleoperation](../tutorials/pico-teleoperation.md)
- [Motion Recording](../tutorials/motion-recording.md)
