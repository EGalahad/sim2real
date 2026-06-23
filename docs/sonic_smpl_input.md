---
title: SONIC SMPL Input
slug: /reference/sonic-smpl-input
---

# SONIC SMPL Input

This note records the runtime contract for SONIC SMPL mode. The short version:

```text
raw XRobot body poses
    -> SMPL local body rotations
    -> SONIC human_joints_info FK
    -> canonical root-local SMPL joints

GMR retarget
    -> robot joint_pos
    -> wrist joint references used by the same SONIC encoder input
```

SMPL reference data and GMR retarget data share the same live tracking source,
but they are not the same representation.

## Encoder Components

The SONIC checkpoint config lists the SMPL encoder inputs as:

| Component | Shape | Runtime source | Meaning |
|---|---:|---|---|
| `smpl_joints_multi_future_local_nonflat` | `[10, 72]` | `motion_data.smpl_joint_pos_root` | 24 canonical SMPL joint positions, root-local, flattened as xyz. |
| `smpl_root_ori_b_multi_future` | `[10, 6]` | `motion_data.smpl_root_quat_w` + current robot root quat | Relative root orientation encoded as the first two rotation-matrix columns. |
| `joint_pos_multi_future_wrist_for_smpl` | `[10, 6]` | wrist slice from `motion_data.joint_pos` | G1 robot wrist joint angles: roll, pitch, yaw for both wrists. |

Current runtime code packs these fields in
`sim2real/rl_policy/observations/sonic.py::sonic_smpl_official_encoder_input`.

## Runtime Payload Fields

The SMPL ZMQ publisher sends these reference fields:

| Payload field | Shape | Meaning |
|---|---:|---|
| `smpl_body_pose_aa` | `[N, 21, 3]` | SMPL body local axis-angle rotations, excluding root. Published for traceability and buffer completeness. |
| `smpl_joint_pos_root` | `[N, 24, 3]` | Canonical SMPL joint positions in the SMPL root frame. This is the field consumed by the encoder. |
| `smpl_root_quat_w` | `[N, 4]` | SMPL reference root quaternion in `wxyz` order after SONIC root-frame conversion. |
| `joint_pos` | `[N, num_robot_joints]` | Retargeted G1 robot joint positions. The encoder only uses the six wrist joints. |

The six wrist names are:

- `left_wrist_roll_joint`
- `right_wrist_roll_joint`
- `left_wrist_pitch_joint`
- `right_wrist_pitch_joint`
- `left_wrist_yaw_joint`
- `right_wrist_yaw_joint`

These are robot joint names, not SMPL skeleton joint names.

## Raw XRobot Data

XRobot body tracking provides one pose per body in this layout:

```text
[x, y, z, qx, qy, qz, qw]
```

Those rotations are treated as global body orientations in the tracking/world
frame, not as local rotations relative to each parent.

`coordinate_transform_unity_data` is not forward kinematics. It applies the
same Unity-to-right-hand coordinate transform independently to each body pose:

```text
position -> position @ rotation_matrix.T
orientation -> coordinate_rotation * orientation
```

It does not traverse the SMPL parent tree and does not generate a canonical
SMPL skeleton.

## GMR Path

The GMR path consumes the processed body-pose dictionary:

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

GMR trusts the tracker-provided global positions and orientations as IK target
frames. `scaled_human_data` is only a scaled and offset body-pose dictionary for
IK. It is not the SONIC canonical SMPL joint set, and it does not use
`human_joints_info.pkl`.

The runtime still needs this path because the SONIC SMPL encoder contract
includes `joint_pos_multi_future_wrist_for_smpl`, which is a robot wrist-joint
reference.

## SONIC SMPL Reference Path

The SMPL reference path should stay separate from GMR:

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

This is forward kinematics, not IK. It deliberately does not use the raw
tracker body positions as `smpl_joint_pos_root`, because SONIC was trained on
canonical SMPL preprocessing rather than live tracker body positions.

## Why Wrist Joints Are Separate

The SMPL joint positions already include human wrist and hand points:
`Left_Wrist`, `Right_Wrist`, `Left_Hand`, and `Right_Hand`.

The extra wrist input is different. It contains G1 robot wrist joint angles, not
SMPL wrist body positions. Position references tell the policy where the human
hands are, but they do not uniquely determine the robot wrist roll, pitch, and
yaw. The retargeted robot wrist angles provide that missing robot-configuration
reference.

Therefore, under the current ONNX input contract:

- SMPL reference fields come from raw XRobot rotations plus SONIC SMPL FK.
- Robot wrist references still come from GMR retargeted `joint_pos`.
- Skipping GMR entirely would require replacing the six wrist joint references
  with another solver or changing/re-exporting the SONIC encoder input contract.

## Relevant Files

- `sim2real/teleop/smpl_stream.py`
- `sim2real/teleop/pico_retarget_pub.py`
- `sim2real/rl_policy/utils/motion_buffer.py`
- `sim2real/rl_policy/observations/sonic.py`
- `checkpoints/sonic_groot_6k/model_config.yaml`
