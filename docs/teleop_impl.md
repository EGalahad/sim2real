---
title: Teleop Implementation
slug: /reference/teleop-implementation
---

# Teleop Implementation Notes

这个文档只放实现细节，README 只保留操作步骤和注意事项。

## Publisher payload

`sim2real.teleop.pico_g1_zmq_publisher` 发布的消息是 JSON，主要字段包括：

- `seq`
- `t_ns`
- `publish_t_ns`
- `smplx_t_ns`
- `joint_pos`
- `body_pos_w`
- `body_quat_w`
- `qpos`
- `root_pos`
- `root_quat`
- `dof_pos`

字段约定：

- `qpos` 长度是 36。
- `joint_pos` 是 G1 的 29 维关节角。
- `body_pos_w` 形状是 `[num_bodies, 3]`。
- `body_quat_w` 形状是 `[num_bodies, 4]`。
- `root_pos` 是 3 维。
- `root_quat` 是 4 维。
- `dof_pos` 是兼容旧 subscriber 的冗余字段，等价于 `joint_pos`。

旋转格式：

- 四元数顺序统一使用 `wxyz`。

## Recording format

`sim2real.teleop.record_xrobot_smplx` 输出 `.npz`，主要字段包括：

- `metadata_json`
- `body_joint_names`
- `body_pos`
- `body_rot_wxyz`
- `capture_time_ns`
- `frame_valid`
- `frames`

字段约定：

- `body_pos` 形状是 `[T, J, 3]`。
- `body_rot_wxyz` 形状是 `[T, J, 4]`。
- 旋转四元数顺序是 `wxyz`。

## Benchmark output

`sim2real.teleop.benchmark_smplx_retarget` 常见输出包括：

- `overall_fps`
- `avg_ms_per_frame`
- `median_ms_per_frame`
- `p95_ms_per_frame`
- `min_ms_per_frame`
- `max_ms_per_frame`

## Common parameters

和 retarget 相关的常用参数：

- `--actual_human_height`
- `--publish_hz`
- `--viewer_hz`
- `--min_link_height`
- `--min_link_height_align_strategy`
- `--min_link_height_bootstrap_frames`

补充说明：

- `--actual_human_height` 会传给 GMR，影响人体尺度匹配。
- `--publish_hz` 是 publisher 主循环频率。
- `--viewer_hz` 是 viewer 刷新频率。
- `--min_link_height` 用于控制目标最小 link 高度。
- `--min_link_height_align_strategy` 支持 `none`、`startup_fixed`、`per_frame`。
- `--min_link_height_bootstrap_frames` 用于 `startup_fixed` 模式下估计固定高度偏移。
