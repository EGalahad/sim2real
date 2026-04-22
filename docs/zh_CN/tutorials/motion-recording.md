# Motion Recording

English version: [../../tutorials/motion-recording.md](../../tutorials/motion-recording.md)

这个教程把 `sim2real/teleop/pico_retarget_pub.py` 发布的 retargeted G1 motion stream 录成 any4hdmi 的 qpos motion clip。

## 1. 启动 live publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

## 2. 录制 motion stream

```bash
uv --project venv/teleop run sim2real/teleop/record_motion.py \
  --connect tcp://127.0.0.1:28701
```

用 `Ctrl-C` 停止录制并写出数据。

## Output

默认会生成一个时间戳目录，例如 `g1_motion_YYYYMMDD_HHMMSS/`，里面会写出：

- `motion.npz`
- 单条 motion 数据集 payload
- 自动生成的 manifest

终端会打印最终输出目录、frame 数、invalid frame 数，以及推断出的 FPS。

## 3. 可选：用 realtime viewer 回看保存的 motion

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --motion_backend npz \
  --motion_path g1_motion_YYYYMMDD_HHMMSS/motion.npz
```
