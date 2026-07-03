---
title: Yaw Alignment
slug: /reference/yaw-alignment
---

# Yaw Alignment

Motion stream contract 有三条核心 invariant：

1. **Motion buffer handles motion continuity.** 它消费 `motion_first_frame` signal，并保证自己输出给 policy 的 streamed motion 在 `xy` 和 yaw 上连续。
2. **Policy handles robot-motion yaw alignment.** Policy 只在 observation `reset()` 时处理机器人当前 yaw 和 motion 初始 yaw 不一致的问题，之后不再因为 stream segment boundary 重新 align。
3. **Motion buffer maintains the last frame.** 如果 motion source 停止发包，buffer 继续返回最后一帧 transformed motion，并把速度置零，policy 不应该因为 stream 断开而 block。

也就是说，stream segment 之间的连续性和 stream liveness fallback 属于 motion buffer；robot 和 motion 初始朝向不一致的问题属于 policy observation `reset()`。

## Policy 做什么

Policy 只负责 **一开始** 的 robot-vs-motion yaw alignment。

在 observation `reset()` 时，policy 读取：

```text
当前 robot yaw
当前 motion yaw
```

然后计算一个固定 yaw offset：

```text
motion yaw * inverse(robot yaw)
```

之后 observation 每一步都用这个固定 offset，把 robot 和 motion 放到一致的 yaw frame 里比较。

这个 offset 不应该随着 motion stream 的 `motion_first_frame` 变化。否则 policy 会在 runtime 中途重新定义 observation frame，反而可能制造 reference 突变。

Policy 不做这些事：

- 不消费 `motion_first_frame`
- 不因为 motion stream reconnect 而 reset observation
- 不在 `.npz` reset / loop / default-pose 切换时重新 align yaw
- 不把 stream segment boundary 传进 observation hook

## Motion Buffer 做什么

Motion buffer 负责 streamed reference 自己的 `xy/yaw` 连续性，并负责在 stream 没有新 payload 时维持最后一个有效 reference。

当 publisher 发来：

```text
motion_first_frame = true
```

buffer 把这帧视为新的 reference segment。这个 segment 的 raw root pose 可能和上一个 segment 的最后输出帧不连续。

处理步骤是：

1. 取 incoming first frame 的 raw root `xy/yaw`。
2. 取上一帧已经输出过的 transformed root `xy/yaw`。
3. 计算 yaw-only rotation。
4. 计算 `xy` translation。
5. 把这个 transform 应用到当前 frame 和之后所有 frame，直到下一个 `motion_first_frame`。

因此，policy 看到的 streamed motion root `xy/yaw` 是连续的。

`z`、roll、pitch、joint pose 不参与这个 segment transform。这个逻辑只处理地面平面上的 root translation 和 heading。

如果 queue 在 cleanup 之后变空，buffer 不返回 empty motion，也不等待新的 ZMQ packet。它返回最后一次已经 transform 过的 frame：

```text
last transformed joint_pos
last transformed body_pos_w
last transformed body_quat_w
zero velocities
```

这保证 motion source 被 kill、短暂断联、或者未来窗口已经耗尽时，policy 仍然看到一个静止的 reference，而不是卡住或突然失去 motion。

## 为什么这样能避免突变

如果每个 publisher 都在 reference 可能跳变时发 `motion_first_frame`，那么 motion buffer 会把每个新 segment 接到上一帧 output 上。

只要 policy 在最开始把 robot yaw 和 motion yaw 对齐，后续 policy 看到的是：

```text
continuous motion reference
+ fixed robot-vs-motion yaw offset
+ last-frame fallback when stream stalls
```

这两个 transform 都是稳定的，所以不会在 stream reset、`.npz` loop、default/motion 切换时突然改变 policy observation frame。

## 相关边界

- Publisher 应在 reference `xy/yaw` 可能突变时发 `motion_first_frame`，例如 motion/default 切换、`.npz` reset、`.npz` loop。
- Stream 断开、没有新 payload、重新连上：见 [Motion Stream Disconnects](/reference/motion-stream-disconnects)。
- `npz_pub.py` 的键盘状态机：见 [NPZ Motion Publisher](/reference/npz-motion-publisher)。
