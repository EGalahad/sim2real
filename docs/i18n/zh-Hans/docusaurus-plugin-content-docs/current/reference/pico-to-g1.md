---
title: Pico To G1
slug: /reference/pico-to-g1
---

# PICO To G1 Mapping

This note summarizes the live `PICO/XRobot -> G1` retarget path used by
`sim2real/teleop/pico_retarget_pub.py`.

## Scope

The live publisher does not consume standard SMPLX parameter tensors such as
`betas`, `pose_body`, `root_orient`, or `trans`.

It consumes XRobot body tracking data, converts it into a body-pose dictionary,
then passes that dictionary into GMR with:

- `src_human="xrobot"`
- `tgt_robot="unitree_g1"`

Relevant sources:

- `sim2real/teleop/pico_retarget_pub.py`
- `venv/teleop/.venv/lib/python3.10/site-packages/general_motion_retargeting/xrobot_utils.py`
- `venv/teleop/.venv/lib/python3.10/site-packages/general_motion_retargeting/motion_retarget.py`
- `venv/teleop/.venv/lib/python3.10/site-packages/general_motion_retargeting/ik_configs/xrobot_to_g1.json`

## PICO Body Points

XRobot exposes 24 body points in this order:

- `Pelvis`
- `Left_Hip`
- `Right_Hip`
- `Spine1`
- `Left_Knee`
- `Right_Knee`
- `Spine2`
- `Left_Ankle`
- `Right_Ankle`
- `Spine3`
- `Left_Foot`
- `Right_Foot`
- `Neck`
- `Left_Collar`
- `Right_Collar`
- `Head`
- `Left_Shoulder`
- `Right_Shoulder`
- `Left_Elbow`
- `Right_Elbow`
- `Left_Wrist`
- `Right_Wrist`
- `Left_Hand`
- `Right_Hand`

Only a subset of these points is used by the G1 retarget config.

## Mapping Table

The table below is copied from `xrobot_to_g1.json`.

- `Phase-1 weight` means `ik_match_table1` `(position_cost, orientation_cost)`.
- `Phase-2 weight` means `ik_match_table2` `(position_cost, orientation_cost)`.
- `pos offset` is applied in the local frame before solving IK.
- `rot offset` is a quaternion in `(w, x, y, z)` order.

| PICO point | G1 body link | Phase-1 weight | Phase-2 weight | pos offset | rot offset |
|---|---|---|---|---|---|
| `Pelvis` | `pelvis` | `(0, 10)` | `(10, 5)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Left_Hip` | `left_hip_yaw_link` | `(0, 10)` | `(10, 5)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Left_Knee` | `left_knee_link` | `(0, 10)` | `(10, 5)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Left_Foot` | `left_toe_link` | `(100, 10)` | `(100, 10)` | `[0.05, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Right_Hip` | `right_hip_yaw_link` | `(0, 10)` | `(10, 5)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Right_Knee` | `right_knee_link` | `(0, 10)` | `(10, 5)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Right_Foot` | `right_toe_link` | `(100, 10)` | `(100, 10)` | `[0.05, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Spine3` | `torso_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[-0.5, 0.5, -0.5, -0.5]` |
| `Left_Shoulder` | `left_shoulder_yaw_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.7071067811865475, 0.0, 0.7071067811865475, 0.0]` |
| `Left_Elbow` | `left_elbow_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 1.0, 0.0]` |
| `Left_Wrist` | `left_wrist_yaw_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 1.0, 0.0]` |
| `Right_Shoulder` | `right_shoulder_yaw_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.0, 0.7071067811865475, 0.0, -0.7071067811865475]` |
| `Right_Elbow` | `right_elbow_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.0, 1.0, 0.0, 0.0]` |
| `Right_Wrist` | `right_wrist_yaw_link` | `(0, 10)` | `(0, 10)` | `[0.0, 0.0, 0.0]` | `[0.0, 1.0, 0.0, 0.0]` |

## Unused PICO Points

These PICO/XRobot body points exist in the stream but are not directly bound in
the current `xrobot_to_g1.json` IK task tables:

- `Spine1`
- `Spine2`
- `Left_Ankle`
- `Right_Ankle`
- `Neck`
- `Left_Collar`
- `Right_Collar`
- `Head`
- `Left_Hand`
- `Right_Hand`

## GMR Preprocessing Flow

The actual preprocessing logic lives in
`motion_retarget.py`.

For the live `xrobot -> unitree_g1` path, the flow is:

1. `pico_retarget_pub.py` builds a `dict[name] = [pos, quat_wxyz]` for all 24 XRobot body points.
2. GMR loads `xrobot_to_g1.json` from `params.py`.
3. GMR computes a global height ratio:
   `ratio = actual_human_height / human_height_assumption`.
4. GMR multiplies every entry in `human_scale_table` by that ratio.
5. `update_targets()` runs the following preprocessing chain:
   - `to_numpy()`
   - `scale_human_data()`
   - `offset_human_data()`
   - `apply_ground_offset()`
   - optionally `offset_human_data_to_ground()`
6. The processed human poses are then used as `FrameTask` targets for IK.

### Scaling

Scaling is position-only. Quaternions are not scaled or changed in
`scale_human_data()`.

The logic is:

- Read the human root from `human_root_name`
- Scale the root position directly by `human_scale_table[root]`
- For every other body listed in `human_scale_table`:
  - convert position to root-local coordinates:
    `local_pos = body_pos - root_pos`
  - scale that local vector by the per-body scale factor
  - transform back to world coordinates by adding the scaled root position
- Preserve the original quaternion for every retained body

Important consequence:

- Bodies that are not present in `human_scale_table` are dropped at this stage
- For `xrobot_to_g1.json`, that means the live 24-point PICO stream is reduced
  to:
  - `Pelvis`
  - `Spine3`
  - `Left_Hip`
  - `Right_Hip`
  - `Left_Knee`
  - `Right_Knee`
  - `Left_Foot`
  - `Right_Foot`
  - `Left_Shoulder`
  - `Right_Shoulder`
  - `Left_Elbow`
  - `Right_Elbow`
  - `Left_Wrist`
  - `Right_Wrist`

This is why the code can ingest the full 24-point body stream without failing in
`offset_human_data()`: unsupported points have already been removed by
`scale_human_data()`.

### Offset Application

Offsets are applied in `offset_human_data()`.

For each retained body:

1. Start from the scaled pose `(pos, quat)`.
2. Apply the configured rotation offset first:
   `updated_quat = quat * rot_offset`
3. Read the configured position offset in the body-local frame.
4. Rotate that local offset into world coordinates using `updated_quat`.
5. Add the rotated world offset to the body position.

So the position offset is not a raw world-space translation. It is a local-frame
offset that gets rotated by the already-offset body orientation.

In config loading, GMR also subtracts `ground_height * [0, 0, 1]` from every
`pos_offset` before storing it, so the effective stored local offset is:

- `stored_pos_offset = config_pos_offset - ground_vector`

### Ground Handling

There are two separate ground-related mechanisms:

- `apply_ground_offset()`
  - subtracts `self.ground_offset` from every body's world `z`
  - this only does anything if some caller previously invoked
    `set_ground_offset(...)`
- `offset_human_data_to_ground()`
  - optional, only runs when `retarget(..., offset_to_ground=True)`
  - finds the lowest body whose name contains `Foot` or `foot`
  - translates the whole body set so that this lowest foot lands at `z = 0.1`

In `pico_retarget_pub.py`, GMR is called with `offset_to_ground=False`, so this
second step is currently disabled in the live publisher path.

## Two-Phase IK Details

After preprocessing, GMR creates `mink.FrameTask` targets and solves IK in two
stages.

### Phase 1

Tasks come from `ik_match_table1`.

Typical intent in the `xrobot_to_g1.json` config:

- feet: strong position constraints
- pelvis / hips / knees: mainly orientation in phase 1
- torso / shoulders / elbows / wrists: mostly orientation-driven

GMR solves one IK step, integrates velocity into the current MuJoCo
configuration, then iterates while error keeps dropping by more than `0.001`,
up to `max_iter = 10`.

### Phase 2

Tasks come from `ik_match_table2`, and the solver continues from the phase-1
configuration. The configuration is not reset between phases.

For `xrobot_to_g1.json`, the main difference is that phase 2 adds position
weights to:

- `Pelvis`
- `Left_Hip`
- `Right_Hip`
- `Left_Knee`
- `Right_Knee`

Feet remain strongly position-constrained in both phases.

### Important Implementation Detail

Although GMR loads both:

- `pos_offsets1` / `rot_offsets1`
- `pos_offsets2` / `rot_offsets2`

`update_targets()` currently preprocesses the human data only once, using
`pos_offsets1` and `rot_offsets1`, and then feeds that same processed pose set
into both `tasks1` and `tasks2`.

So in the current implementation:

- phase 1 and phase 2 do use different task weights and potentially different
  task sets
- but phase 2 does not get its own separately offset human target poses

This is a property of the current code in `motion_retarget.py`, not just of the config files.

## Output Shape

After retargeting, GMR returns a full MuJoCo `qpos` for G1:

- `qpos[0:7]`: floating base pose
- `qpos[7:36]`: 29 scalar G1 joints

The live publisher then runs MuJoCo forward kinematics and publishes:

- `qpos`
- `joint_pos`
- `body_pos_w`
- `body_quat_w`

The canonical G1 joint order comes from `sim2real/config/robots/g1.py`.
