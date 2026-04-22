---
title: Motion Recording
sidebar_position: 3
---

This tutorial records the retargeted G1 motion stream published by `sim2real/teleop/pico_retarget_pub.py` and saves it as an any4hdmi qpos motion clip.

## 1. Start the live publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py \
  --bind tcp://*:28701 \
  --publish_hz 50 \
  --actual_human_height 1.80
```

## 2. Record the motion stream

```bash
uv --project venv/teleop run sim2real/teleop/record_motion.py \
  --connect tcp://127.0.0.1:28701
```

Press `Ctrl-C` to stop recording and write the dataset.

## Output

By default, the recorder creates a timestamped directory such as `g1_motion_YYYYMMDD_HHMMSS/` and writes:

- `motion.npz`
- the single-motion dataset payload
- the generated manifest

The terminal prints the final output directory, frame count, invalid frame count, and inferred FPS.

## 3. Optional: replay the saved motion in the realtime viewer

```bash
uv --project venv/teleop run sim2real/teleop/realtime_viewer.py \
  --motion_backend npz \
  --motion_path g1_motion_YYYYMMDD_HHMMSS/motion.npz
```
