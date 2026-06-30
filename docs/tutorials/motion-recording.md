---
title: Motion Recording
sidebar_position: 3
---

This tutorial records the retargeted G1 motion stream published by `sim2real/teleop/pico_retarget_pub.py` and saves it as an any4hdmi qpos motion clip.

## 1. Start the live publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py
```

## 2. Record the motion stream

```bash
uv run scripts/record_motion.py
```

Press `Ctrl-C` to stop recording and write the dataset.

## Output

By default, the recorder creates a timestamped directory such as `g1_motion_YYYYMMDD_HHMMSS/` and writes:

- `manifest.json`
- `motions/motion.npz`

The output directory is an any4hdmi dataset root. The terminal prints the final output directory, frame count, invalid frame count, and inferred FPS.

## 3. Optional: replay the saved motion with any4hdmi

```bash
uv run scripts/view_motion.py \
  --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz
```

The live retarget viewer is built into `sim2real/teleop/pico_retarget_pub.py`. It does not replay recorded `.npz` files.
