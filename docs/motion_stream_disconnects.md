---
title: Motion Stream Disconnects
slug: /reference/motion-stream-disconnects
---

# Motion Stream Disconnects

This page documents what happens when a ZMQ motion stream stops publishing and
later starts publishing again.

The liveness side of the motion stream contract is:

1. Motion sources publish explicit reference payloads when they have new data.
2. `RealtimeMotionBuffer` owns the stream timeline seen by policy.
3. If no new payload arrives, the buffer keeps serving the last transformed
   frame with zero velocities.

That means missing ZMQ packets, a killed publisher, or a temporary live-source
disconnect should stop the reference in place. It should not block the policy
process, empty the future motion window, or force policy-side observation reset.

## Missing Payloads

Missing payloads are different from explicit paused/default payloads.

If no new payload arrives, `RealtimeMotionBuffer` keeps returning the last
transformed frame it has seen:

```text
last joint_pos
last body_pos_w
last body_quat_w
zero joint_vel
zero body_lin_vel_w
zero body_ang_vel_w
```

This is the expected behavior after killing `npz_pub.py` or temporarily losing a
live motion source. The policy should not block waiting for a new packet.

## Explicit Paused Or Default Payloads

If a publisher sends a payload with `paused=true` or `source="default"`, that is
still an explicit reference frame.

The buffer accepts it and updates its internal reference timeline. This lets
`npz_pub.py` intentionally switch from motion to default pose, or pause on a
specific motion frame.

## Reconnect

When a motion source reconnects after silence, the source must decide whether
the new payload is continuous with the previous reference.

If it may jump in root `xy/yaw`, the source must publish:

```text
motion_first_frame = true
```

Then `RealtimeMotionBuffer` will attach the new segment to the previous
transformed output frame using yaw-only rotation plus `xy` translation.

If the source has already preserved continuity itself, it can continue with
`motion_first_frame=false`.

## Publisher Rules

For `.npz` streaming, `npz_pub.py` marks segment boundaries when:

- switching default pose and motion
- resetting `motion_t` to frame 0
- looping from the last frame to frame 0

For Pico retargeting, pause/unpause continuity is handled inside
`pico_retarget_pub.py` by anchoring the new live pelvis to the previous tracked
pelvis. Pause/unpause is not a reconnect boundary.

If live XR body data disappears while Pico is in live mode, the publisher treats
the next successful body frame as a reconnect. That recovered frame:

- reinitializes the live pelvis anchor
- reinitializes the SMPL heading anchor when SMPL publishing is enabled
- publishes `motion_first_frame=true`

The buffer then attaches the recovered segment to the previous transformed
output frame.
