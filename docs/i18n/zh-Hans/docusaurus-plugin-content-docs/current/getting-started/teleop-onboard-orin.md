# Teleop Project (Onboard Orin)

当 Pico / XR 工具直接跑在 G1 的 onboard Orin 上时，使用这条路径。root project 仍然负责 policy runtime 和 `scripts/real_bridge.py`。

## Setup

```bash
uv --project venv/teleop sync
```

先下载 [JetPack 5 预编译包](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e?usp=sharing) 并解压到 repo 根目录，保证 `prebuilt/` 存在。

### 安装 XRoboToolkit PC Service

```bash
sudo apt install -y \
  ./prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

启动服务：

```bash
bash /opt/apps/roboticsservice/runService.sh
```

### Clone Pico SDK 仓库

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
git -C external/XRoboToolkit-PC-Service checkout orin
```

### 替换上游 aarch64 gRPC 包

先按 [XRobot gRPC JetPack 5](/reference/xrobot-grpc-jetpack5) 准备 JetPack 5 兼容包，再替换目录：

```bash
export sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"

rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

### Build 并安装 `xrobotoolkit_sdk`

```bash
bash scripts/setup/setup_xrobot_pybind.sh --arch aarch64
```

## Verify Installation

先在 G1 onboard Orin 上启动 live retarget publisher：

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

在第二个终端里启动 realtime viewer，并把 `--connect` 指到 G1 Orin 的 IP：

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --connect tcp://<g1-orin-ip>:28701 \
  --viewer_hz 50
```

如果 viewer 里能看到实时更新的 G1 retarget 动作，说明 onboard teleop 环境已经打通。

## Notes

- 如果 onboard 机器也跑 policy 和 bridge，请同时在 repo 根目录执行 `uv sync`
- 如果 Pico publisher 跑在另一台 PC 上，记得把 policy 的 `--motion_zmq_connect` 指到那台机器

## Next Steps

- [Pico Teleoperation](../tutorials/pico-teleoperation.md)
