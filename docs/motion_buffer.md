---
title: Motion Buffer
slug: /reference/motion-buffer
---

# `sim2real/rl_policy/utils/motion_buffer.py` 工作逻辑

这篇文档单独解释 `RealtimeMotionBuffer` 是怎么工作的。

它的职责可以概括成一句话：

```text
持续接收 publisher 发来的实时 motion 数据
    -> 按时间戳缓存
    -> 在 policy 需要 observation 时按目标时间插值
    -> 返回统一的 MotionData
```

这个类主要服务 tracking policy 的 live / zmq motion 模式。

## 1. 它解决的核心问题

tracking policy 不是只需要“当前这一帧” motion。

它通常需要这样一组参考时间点：

```text
future_steps = [-4, -2, 0, 1, 2, 3, 4, 8, 16]
dt = 20ms
```

也就是说，policy 一次 forward 需要：

- 比“参考时刻”更早的几帧
- 参考时刻本身
- 未来几帧

但实时 VR teleop 的 motion 是通过 ZMQ 一帧一帧到达的，而且到达时间并不一定正好卡在这些目标时间点上。

所以需要一个 buffer 来做三件事：

1. 把收到的 motion 按时间存起来
2. 给 policy 提供一个“延迟后的稳定时间轴”
3. 对不在离散采样点上的目标时间做插值

## 2. 整体数据流

```text
PICO / XR body stream
    -> pico_g1_zmq_publisher.py
    -> ZMQ JSON payload
    -> RealtimeMotionBuffer
    -> MotionData
    -> StateProcessor
    -> track observations
    -> policy
```

更细一点：

```text
publisher
    发出:
        smplx_t_ns
        joint_pos
        body_pos_w
        body_quat_w

RealtimeMotionBuffer
    收到 payload
        -> 解析 JSON
        -> 按 smplx_t_ns 排序插入
        -> 保存到内部时间缓存

policy step
    当前时间 t
        -> 取参考时刻 t - delay
        -> 根据 future_steps 生成一组目标时间戳
        -> 从 buffer 里查找/插值
        -> 返回 MotionData
```

## 3. 初始化时做了什么

`RealtimeMotionBuffer.__init__()` 里会初始化这些状态：

- `joint_names`
  - 固定为 teleop 侧定义好的 canonical G1 joint 顺序
- `body_names`
  - 固定为 teleop 侧定义好的 canonical G1 body 顺序
- `future_steps`
  - policy 配置里要求的参考时间步列表
- `dt_s`
  - motion 的离散步长，tracking 里通常是 `0.02`
- `tolerance_s`
  - 给网络抖动 / retarget 延迟预留的安全余量
- `delay_s`
  - 如果没显式传，就自动按下面公式算：

```text
delay_s = max(future_steps) * dt_s + tolerance_s
```

例如：

```text
future_steps = [-4, -2, 0, 1, 2, 3, 4, 8, 16]
dt_s = 0.02
tolerance_s = 0.04

delay_s = 16 * 0.02 + 0.04 = 0.36 s
```

它还会准备两个内部容器：

- `_timestamps_ns`
  - 每一帧对应的时间戳
- `_frames`
  - 每一帧对应的 motion 数据

可以理解成：

```text
_timestamps_ns: [t0, t1, t2, t3, ...]
_frames:        [f0, f1, f2, f3, ...]
```

两者下标一一对应，而且始终按时间升序排列。

## 4. ZMQ 收包线程怎么工作

如果初始化时传了 `motion_zmq_connect`，buffer 会起一个后台线程：

```text
while True:
    recv payload from ZMQ
    parse payload
    append into sorted buffer
```

这部分逻辑在 `_start_motion_stream()` 里。

每次收到一条 payload，会交给 `__append_payload()`。

## 5. 一条 payload 进入 buffer 时发生了什么

`__append_payload()` 的逻辑可以概括成：

```text
raw string / bytes
    -> 去掉 topic 前缀
    -> JSON 反序列化
    -> 取出时间戳和 motion 字段
    -> 转成 numpy
    -> 插入有序时间轴
```

当前它主要读这些字段：

- `smplx_t_ns`
- `joint_pos`
- `body_pos_w`
- `body_quat_w`

兼容逻辑：

- 如果没有 `joint_pos`，也会尝试读 `dof_pos` 或 `qpos`
- 如果给了 `joint_names/body_names`，会检查它们是否和 canonical G1 顺序一致

插入时用的是 `bisect_right`，所以即使消息有轻微乱序，也会按时间顺序插进去。

内部每个 frame 现在只保留三类核心数据：

```text
frame = {
    joint_pos,
    body_pos_w,
    body_quat_w,
}
```

## 6. 为什么 policy 看到的是 `t - delay`

这是这个 buffer 最重要的设计点。

policy 在控制时刻 `t`，不能直接把 `t` 当作参考时刻，因为它还需要未来若干步的数据。

所以这里采用：

```text
参考时刻 = t - delay
```

这样，policy 实际上总是在看“一段更早、但已经收齐未来窗口”的 motion。

示意图：

```text
当前真实时间:                 t
                              |
                              v
时间轴:   ---- ---- ---- ---- ---- ---- ---- ---- ----
                    ^
                    |
                 t - delay

policy 需要的 future_steps:
[-4, -2, 0, 1, 2, 3, 4, 8, 16]

对应目标时间:
(t-delay) - 4*dt
(t-delay) - 2*dt
(t-delay) + 0*dt
(t-delay) + 1*dt
...
(t-delay) + 16*dt
```

所以虽然名字叫 `future_steps`，但从“当前真实时间 t”的视角看，它其实是：

- 一个经过延迟对齐后的未来窗口
- 用这个窗口来保证 observation 不会因为“未来帧还没到”而缺数据

## 7. `get_obs()` 怎么取一组 observation

`get_obs()` 每次被调用时，大致做这几步。

### 第一步：拿当前时间

```text
current_time_ns = time.time_ns()
```

### 第二步：清掉太旧的数据

先算一个 cutoff：

```text
cutoff = current_time
         - (delay + abs(min_future_step) * dt)
```

这表示：

- 比这个时间更早的帧
- 连最早那个负的 `future_step` 也用不上了

就可以删掉。

但实现上会保留 cutoff 前最后一帧，避免插值时丢失左端点。

示意图：

```text
时间轴:

old old old keep | useful useful useful useful
            ^
            |
          cutoff

清理规则:
    cutoff 之前不是全删
    而是保留 cutoff 前最后一帧
```

这是为了保证：

```text
目标时间落在 [旧帧, 新帧] 中间时
仍然能做插值
```

### 第三步：生成目标时间戳

先算参考基准时间：

```text
target_base = current_time - delay
```

再把 `future_steps` 展开成具体时间：

```text
target_times = target_base + future_steps * dt
```

例如：

```text
future_steps = [-4, -2, 0, 1, 2, 3, 4, 8, 16]
dt = 20ms

target_times =
    target_base - 80ms
    target_base - 40ms
    target_base
    target_base + 20ms
    ...
    target_base + 320ms
```

### 第四步：逐个目标时间取样

对每个 `target_time`，调用 `_sample_frame_locked()`。

### 第五步：堆成 `MotionData`

最后把所有目标时间对应的 frame 堆起来，组成：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_lin_vel_w`
- `body_quat_w`
- `body_ang_vel_w`

并包装成 `MotionData` 返回给上层。

## 8. `_sample_frame_locked()` 的三种情况

对某个目标时间 `ts`，它分三种情况处理。

### 情况 1：`ts` 早于最老帧

直接返回最老帧。

```text
ts   t0   t1   t2
|----|----|----|

返回 t0
```

### 情况 2：`ts` 晚于最新帧

直接返回最新帧。

```text
t0   t1   t2   ts
|----|----|----|

返回 t2
```

### 情况 3：`ts` 落在两帧中间

找到：

```text
t0 <= ts <= t1
```

然后计算：

```text
alpha = (ts - t0) / (t1 - t0)
```

再做插值：

- `joint_pos`
  - 线性插值
- `body_pos_w`
  - 线性插值
- `body_quat_w`
  - batch slerp

示意图：

```text
t0 -------- ts -------- t1
     alpha
```

## 9. quaternion 插值怎么做

`body_quat_w` 不能直接线性插值，所以单独用了 `_quat_slerp_batch()`。

现在的实现是纯 `numpy` 的 batch slerp，逻辑是：

1. 先把 quaternion 归一化
2. 计算 `dot(q0, q1)`
3. 如果 dot 小于 0，就把 `q1` 翻面，保证走最短弧
4. 如果两帧非常接近，就退化成 lerp + normalize
5. 否则走标准 slerp 公式

这样比之前逐 body 调一次 `scipy.Slerp` 要快很多。

## 10. 为什么 velocity 现在全是 0

当前 live tracking 路径里，真正被 observation 用到的是：

- `joint_pos`
- `body_pos_w`
- `body_quat_w`

而 velocity 相关量目前没有被这个 policy 真正消费。

所以现在实现里：

- `joint_vel`
- `body_lin_vel_w`
- `body_ang_vel_w`

都直接返回 0。

对应逻辑在：

- `_frame_with_zero_velocities()`
- `_sample_frame_locked()`

这样做的目的很直接：

- 避免在 `get_obs()` 里做不必要的导数计算
- 把 live buffer retrieval 的耗时压到足够低

## 11. 这个文件当前的几个重要假设

### 假设 1：publisher 的数组顺序已经是 canonical G1 顺序

也就是：

- `joint_pos` 的顺序和 `G1_JOINT_NAMES` 一致
- `body_pos_w/body_quat_w` 的顺序和 `G1_BODY_NAMES` 一致

buffer 现在主要做检查，不做复杂重排。

### 假设 2：publisher 的时间戳和 subscriber 所在机器的时钟大致一致

因为 `get_obs()` 用的是本机 `time.time_ns()`，而 payload 里用的是 publisher 写进去的 `smplx_t_ns`。

如果两边时钟差很多，就会影响 `t - delay` 对齐。

### 假设 3：live policy 只关心 pose，不关心 motion velocity

如果未来有 policy 真要消费：

- `joint_vel`
- `body_lin_vel_w`
- `body_ang_vel_w`

那就需要把这部分恢复成真实计算，而不是全 0。

## 12. 一句话总结

`RealtimeMotionBuffer` 本质上就是一个“带延迟的时间插值器”：

```text
实时收流
    -> 按时间缓存
    -> 用 t-delay 对齐参考时间轴
    -> 对 future_steps 对应的时间点做插值
    -> 返回 MotionData
```

它的核心价值不是“多存几帧”，而是：

- 把实时、抖动、非整点到达的 motion 流
- 变成 policy 可以稳定消费的固定时间栅格 observation
