# VR Teleop Instructions

```bash
uv --project sim2real/teleop sync
```

实现细节、数据格式和参数说明见 [docs/teleop_impl.md](/home/elijah/Documents/projects/simple-tracking/sim2real/sim2real/teleop/docs/teleop_impl.md)。

## XRobot

### XRoboToolkit app

1. 从 <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases> 下载并安装对应的 `.deb` 包。
2. 安装完成后，在 Ubuntu 上启动 `XRoboToolkit` / `XRobot` app。
3. Linux 侧 app 能看到小人，通常说明 PC service 已经连通。

注意事项：

- Ubuntu 主机和 PICO 必须在同一局域网。
- PICO 必须能访问 Ubuntu 主机 IP。
- 本机防火墙不要拦截 XRoboToolkit 相关通信。

### xrobotoolkit_sdk

```bash
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  sim2real/teleop/XRoboToolkit-PC-Service-Pybind
mkdir -p sim2real/teleop/XRoboToolkit-PC-Service-Pybind/tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  sim2real/teleop/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service
mkdir -p sim2real/teleop/XRoboToolkit-PC-Service-Pybind/lib
mkdir -p sim2real/teleop/XRoboToolkit-PC-Service-Pybind/include
```

然后构建底层 SDK，并把生成的头文件和动态库放到 `XRoboToolkit-PC-Service-Pybind` 对应目录，再安装 pybind 包：

```bash
uv pip install --python sim2real/teleop/.venv/bin/python -e sim2real/teleop/XRoboToolkit-PC-Service-Pybind
```

建议先验证环境：

```bash
uv --project sim2real/teleop run python - <<'PY'
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

如果这里 import 失败，先不要继续跑 teleop，先把 SDK 安装问题解决干净。

## PICO

按下面顺序操作：

1. 戴好腿部 trackers。
2. 把 controllers 固定在手腕。
3. 在头显里启动 VR。
4. 完成 whole-body motion tracking 校准。
5. 打开 `XRoboToolkit / XRobot`。
6. 输入 Ubuntu 主机 IP 并连接。
7. 开启 whole-body streaming。

建议先在 Linux 侧确认已经收到 XR 数据：

```bash
uv --project sim2real/teleop run python - <<'PY'
import xrobotoolkit_sdk as xrt

xrt.init()
print("Body data available:", xrt.is_body_data_available())
print("Headset pose:", xrt.get_headset_pose())
print("Left controller pose:", xrt.get_left_controller_pose())
print("Right controller pose:", xrt.get_right_controller_pose())
xrt.close()
PY
```

如果 `Body data available` 仍然是 `False`，优先检查：

- PICO 是否已经连到正确的 Ubuntu 主机。
- trackers 是否完成校准。
- PICO 侧是否真的打开了 whole-body stream。
- XRoboToolkit PC service 是否还在运行。

## Run

### run publisher

先启动 publisher：

```bash
uv --project sim2real/teleop run python -m sim2real.teleop.pico_g1_zmq_publisher \
  --bind tcp://*:28701 \
  --topic g1 \
  --publish_hz 50 \
  --actual_human_height 1.70
```

注意事项：

- publisher 起不来时，先检查 XR 数据流是否已经连通。
- 如果要跨机器使用，`--bind` / `--connect` 地址要和实际网络拓扑一致。
- `--actual_human_height` 会影响 retarget 结果，实测时不要随便留默认值。

### real time viewer

再启动 viewer：

```bash
uv --project sim2real/teleop run python -m sim2real.teleop.pico_g1_zmq_viewer \
  --connect tcp://127.0.0.1:28701 \
  --topic g1 \
  --viewer_hz 50
```

注意事项：

- 同机调试可以直接连 `127.0.0.1`。
- 分机部署时，viewer 机器需要能访问 publisher 暴露的端口。
- 当前更推荐 publisher 和 viewer 分开启动，排查问题更直接。

### recording

录制原始 XRobot body pose：

```bash
uv --project sim2real/teleop run python -m sim2real.teleop.record_xrobot_smplx \
  --output sim2real/teleop/xrobot_smplx_$(date +%Y%m%d_%H%M%S).npz \
  --sample_fps 30 \
  --actual_human_height 1.70
```

`Ctrl-C` 结束录制并保存。

注意事项：

- 录制前先确认数据流稳定，否则前面会积累空帧或无效帧。
- `--sample_fps` 只影响采样频率，不会修复上游流本身的抖动。
- 录制文件建议单独保存，避免后续 benchmark 误用旧数据。

### benchmark

离线 benchmark retarget：

```bash
uv --project sim2real/teleop run python -m sim2real.teleop.benchmark_smplx_retarget \
  --input sim2real/teleop/xrobot_smplx_20260321_000000.npz \
  --actual_human_height 1.70 \
  --warmup_frames 10
```

注意事项：

- benchmark 输入应使用同一套录制参数生成的数据。
- `--actual_human_height` 最好和录制/在线 retarget 保持一致。
- benchmark 结果用于看 retarget 性能，不代表整条实时链路端到端延迟。
