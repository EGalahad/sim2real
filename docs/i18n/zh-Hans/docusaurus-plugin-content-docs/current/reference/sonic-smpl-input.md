---
title: SONIC SMPL Input
slug: /reference/sonic-smpl-input
---

# SONIC SMPL Input

这页记录 SONIC SMPL mode 的 runtime input contract。核心关系是：

```text
raw XRobot body poses
    -> SMPL local body rotations
    -> SONIC human_joints_info FK
    -> canonical root-local SMPL joints

GMR retarget
    -> robot joint_pos
    -> 同一个 SONIC encoder input 里需要的 wrist joint references
```

SMPL reference 和 GMR retarget 共用同一帧 live tracking 数据，但它们不是同一个表示。

## Encoder 组成

SONIC checkpoint config 里的 SMPL encoder input 是三块：

| Component | Shape | Runtime source | 含义 |
|---|---:|---|---|
| `smpl_joints_multi_future_local_nonflat` | `[10, 72]` | `motion_data.smpl_joint_pos_root` | 24 个 canonical SMPL joints 的 root-local xyz，flatten 后输入。 |
| `smpl_root_ori_b_multi_future` | `[10, 6]` | `motion_data.smpl_root_quat_w` + 当前 robot root quat | reference root 和 robot root 的相对朝向，用 rotation matrix 前两列表示。 |
| `joint_pos_multi_future_wrist_for_smpl` | `[10, 6]` | `motion_data.joint_pos` 里的 wrist slice | G1 robot wrist 的 roll、pitch、yaw joint angles，左右各 3 个。 |

当前 runtime 由 `sim2real/rl_policy/observations/sonic.py::sonic_smpl_official_encoder_input`
负责把这些字段 pack 到 encoder input。

## Runtime Payload Fields

SMPL ZMQ publisher 会发这些 reference fields：

| Payload field | Shape | 含义 |
|---|---:|---|
| `smpl_body_pose_aa` | `[N, 21, 3]` | SMPL body local axis-angle rotations，不包含 root。主要用于 traceability 和 buffer 完整性。 |
| `smpl_joint_pos_root` | `[N, 24, 3]` | canonical SMPL joints 在 SMPL root frame 下的位置；这是 encoder 实际读取的 SMPL joint field。 |
| `smpl_root_quat_w` | `[N, 4]` | SONIC root-frame conversion 后的 SMPL reference root quaternion，`wxyz` 顺序。 |
| `joint_pos` | `[N, num_robot_joints]` | retargeted G1 robot joint positions；encoder 只取其中 6 个 wrist joints。 |

6 个 wrist joint names 是：

- `left_wrist_roll_joint`
- `right_wrist_roll_joint`
- `left_wrist_pitch_joint`
- `right_wrist_pitch_joint`
- `left_wrist_yaw_joint`
- `right_wrist_yaw_joint`

这些是 robot joint names，不是 SMPL skeleton joint names。

## Raw XRobot Data

XRobot body tracking 每个 body 给一条 pose：

```text
[x, y, z, qx, qy, qz, qw]
```

这里的 rotation 按 global body orientation 处理，也就是 tracking/world frame 下每个 body 的朝向；它不是相对 parent 的 local rotation。

`coordinate_transform_unity_data` 不是 forward kinematics。它只是对每个 body pose 独立做 Unity-to-right-hand 坐标系转换：

```text
position -> position @ rotation_matrix.T
orientation -> coordinate_rotation * orientation
```

它不会沿 SMPL parent tree forward，也不会生成 canonical SMPL skeleton。

## GMR Path

GMR path 消费的是处理后的 body-pose dict：

```text
raw XRobot body_poses
    -> name mapping and xyzw-to-wxyz quaternion reorder
    -> coordinate_transform_unity_data
    -> live pelvis yaw/xy alignment
    -> min-height z offset
    -> GMR scale_human_data / offset_human_data
    -> robot IK
    -> robot qpos / joint_pos
```

GMR 直接把 tracker 给的 global positions 和 orientations 当作 IK target frames。
`scaled_human_data` 只是经过 scale/offset 后的 body-pose dict，用来设 IK target；
它不是 SONIC canonical SMPL joint set，也不使用 `human_joints_info.pkl`。

当前 runtime 仍然需要这条 path，因为 SONIC SMPL encoder contract 里包含
`joint_pos_multi_future_wrist_for_smpl`，也就是 robot wrist joint reference。

## SONIC SMPL Reference Path

SMPL reference path 应该和 GMR 保持分离：

```text
raw XRobot global body quaternions
    -> convert to local parent-relative SMPL rotations
    -> `smpl_body_pose_aa`
    -> use `human_joints_info.pkl` rest skeleton and parents
    -> FK over the SONIC canonical human skeleton
    -> select 24 output joints: [0..21, 39, 54]
    -> rotate positions into the SMPL root frame
    -> `smpl_joint_pos_root`
```

这里做的是 forward kinematics，不是 IK。它刻意不把 tracker 的 raw body positions
直接当成 `smpl_joint_pos_root`，因为 SONIC 训练时用的是 canonical SMPL preprocessing，
不是 live tracker body positions。

## 为什么 Wrist Joints 要单独给

SMPL joint positions 里已经包含人体的 `Left_Wrist`、`Right_Wrist`、
`Left_Hand`、`Right_Hand`。

额外的 wrist input 是另一类信息：它是 G1 robot wrist joint angles，不是 SMPL wrist body positions。
手的位置 reference 能告诉 policy 人手在哪里，但不能唯一决定机器人 wrist roll、pitch、yaw。
retarget 后的 robot wrist angles 给 policy 补了这个 robot-configuration reference。

所以在当前 ONNX input contract 下：

- SMPL reference fields 来自 raw XRobot rotations + SONIC SMPL FK。
- Robot wrist references 仍然来自 GMR retargeted `joint_pos`。
- 如果完全跳过 GMR，就需要用其他 solver 替代这 6 个 wrist joint references，或者重新改/export SONIC encoder input contract。

## Relevant Files

- `sim2real/teleop/smpl_stream.py`
- `sim2real/teleop/pico_retarget_pub.py`
- `sim2real/rl_policy/utils/motion_buffer.py`
- `sim2real/rl_policy/observations/sonic.py`
- `checkpoints/sonic_groot_6k/model_config.yaml`
