---
title: NPZ Motion Publisher
slug: /reference/npz-motion-publisher
---

# NPZ Motion Publisher

`sim2real/teleop/npz_pub.py` replays an any4hdmi `.npz` motion as the same
canonical ZMQ motion stream used by live Pico retargeting.

It is useful for testing tracking policy behavior against a streamed source
without running the Pico stack.

## State Machine

The publisher has three playback states:

```text
DEFAULT
MOTION_PAUSED
MOTION_PLAYING
```

It also stores one motion cursor:

```text
motion_t
```

`FINISHED` is not a separate state. Reaching the end is represented as:

```text
state = MOTION_PAUSED
motion_t = last_frame
```

## Startup

Startup begins in `DEFAULT` for normal keyboard use. The publisher sends the
aligned default pose until the user requests motion playback.

The default pose is still an explicit reference segment, so the first default
payload is marked with `motion_first_frame=True`.

## Keyboard

| Key | Behavior |
| --- | --- |
| `]` | Reset `motion_t=0`, enter `MOTION_PAUSED`, and mark a new segment. |
| `space` | Toggle `MOTION_PAUSED` and `MOTION_PLAYING`. No effect in `DEFAULT`. |
| `x` | Enter `DEFAULT` and pause the current `motion_t`. |

Pressing `]` works from any state. It always returns to the first motion frame
and pauses there.

Pressing `x` during playback does not reset `motion_t`; it switches the
published reference back to default pose and pauses the cursor.

## Segment Boundaries

The publisher sets `motion_first_frame=True` when the reference can jump:

- startup default pose
- `DEFAULT -> MOTION_PAUSED` through `]`
- motion frame reset to `motion_t=0`
- motion loop from last frame back to frame 0
- `MOTION_* -> DEFAULT` through `x`

While paused on a segment boundary, the publisher may keep sending the boundary
flag on the repeated frame. This is intentional: ZMQ streams often use
conflation, so repeating the flag makes it more likely that the policy side sees
the current segment start.

## End Of Motion

With `hold_last=True`, the terminal frame is emitted once as an active motion
payload. After that payload is sent, the publisher switches to
`MOTION_PAUSED` and holds the final frame.

This matters because `RealtimeMotionBuffer` treats every received payload as an
explicit reference. The final frame must be sent before the state becomes
paused, otherwise ZMQ playback can freeze on the penultimate frame.
