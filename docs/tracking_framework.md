---
title: Tracking Framework
slug: /reference/tracking-framework
---

# Tracking 部分整体框架

本文里的命令默认使用 PC 根目录环境。

本文总结当前 `sim2real` 仓库里 tracking 推理链路的代码组织方式。分析入口是：

```bash
uv run sim2real/rl_policy/tracking.py --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml
uv run sim2real/sim_env/base_sim.py
```

核心结论很简单：tracking policy 只关心低层状态、参考动作序列和 ONNX 推理；执行侧可以是 MuJoCo，也可以是真机。真机部署时通过 `robot_io` 选择 inline 或 ZMQ bridge，具体选择集中放在 [Robot I/O](/reference/robot-io)。

## 1. 两个入口分别做什么

### `sim2real/rl_policy/tracking.py`

- 读取 `robot_config` 和 `policy_config`。
- 根据 `policy_config` 推导 ONNX 模型路径。
- 创建 `Tracking` 对象并调用 `run()`。
- `Tracking` 本身很薄，只是在 `BasePolicy` 上额外加了一个 `paused` 开关，支持空格键或手柄 `B` 键暂停/继续参考动作播放。
- `tracking.py` 还支持 `--onnx_provider cpu|gpu`，用于切换 ONNX Runtime 执行后端；要真正跑到 GPU，环境里必须有 `CUDAExecutionProvider`。

也就是说，tracking 的主要逻辑并不在 `tracking.py`，而在 `sim2real/rl_policy/base_policy.py`、`sim2real/rl_policy/utils/state_processor.py` 和 `sim2real/sim2real/rl_policy/observations/*`。

### `sim2real/sim_env/base_sim.py`

- 读取 `robot_config` 和 `scene_config`。
- 加载 MuJoCo 场景，创建 mjviser server。
- 创建 `SimulationBridge`，把 MuJoCo 状态发布成统一的 `low_state`，并从统一的 `low_cmd` 里取关节目标。
- 以固定仿真步长循环：
  - 发布低层状态
  - 计算 PD + 前馈力矩
  - 写入 `mj_data.ctrl`
  - 执行 `mujoco.mj_step`

因此，`base_sim.py` 并不知道 tracking 是什么任务，它只实现统一的机器人底层接口。

## 2. 总体数据流

tracking 主链路可以概括成下面这条：

```text
MuJoCo / Real Robot
    -> low_state (ZMQ)
    -> StateProcessor
    -> Observation update / compute
    -> obs_dict
    -> ONNX policy
    -> action
    -> q_target
    -> low_cmd (ZMQ)
    -> SimulationBridge / RealBridge
    -> PD torque
    -> MuJoCo / Real Robot
```

其中有两条并行输入：

1. 机器人当前状态
   - 来自 `low_state`
   - 包含 base quaternion、base gyro、joint pos、joint vel、joint torque

2. 参考 tracking 轨迹
   - 来自 any4hdmi dataset 中的 qpos `.npz`，或 live ZMQ motion stream
   - 在推理侧由 `MotionDataset` 读入并缓存
   - 由 `track.py` 中的观测类切成未来多帧参考量

## 3. 策略侧框架

### 3.1 `BasePolicy` 是真正的推理主循环

`Tracking` 继承自 `BasePolicy`，而 `BasePolicy` 负责以下几件事：

1. 初始化状态读入
   - `StateProcessor(robot_config)` 负责订阅 `low_state`

2. 初始化命令发送
   - `CommandSender(robot_config, policy_config)` 负责发布 `low_cmd`

3. 解析策略配置
   - 读取 `joint_names_simulation`
   - 读取 `policy_joint_names`
   - 根据 `default_joint_pos` 和 `action_scale` 把策略输出映射成实际关节目标

4. 加载 ONNX 策略
   - `setup_policy()` 内部通过 `ONNXModule` 调用 onnxruntime
   - ONNX 输出里的 `next_*` 状态会被写回 `state_dict`，因此它支持带隐状态的策略

5. 构建观测系统
   - `setup_observations(policy_config["observation"])`
   - 按配置生成多个 observation group

6. 以固定 `rl_rate` 运行 `_rl_step_scheduled()`

### 3.2 一次 RL step 在做什么

每个控制周期大致分成四段：

1. 读取底层状态
   - `state_processor._prepare_low_state()`
   - 从 `low_state` socket 中取最新一帧，写入 `root_quat_w`、`root_ang_vel_b`、`joint_pos`、`joint_vel`

2. 更新观测
   - `self.update()`
   - 依次调用 `state_processor.update()` 和所有 observation 的 `update()`
   - 然后 `prepare_obs_for_rl()` 把各个 observation group 拼成 ONNX 输入

3. 跑 ONNX 策略
   - `action, q_target, self.state_dict = self.policy(self.state_dict)`
   - `action` 是策略输出
   - `q_target = default_dof_angles + action * action_scale`

4. 规则式控制流收尾
   - 若处于 `get_ready_state`，则插值回默认站姿
   - 若未启用策略，则直接把当前关节角发回去，相当于“保持当前位置”
   - 否则发送策略生成的 `q_target`

最终统一通过：

```python
self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)
```

发给下游 bridge。

### 3.3 运行时状态机和按键

当前策略侧实际上有三个常用运行状态：

1. `use_policy_action = False`
   - 不使用策略输出
   - 直接把当前关节角发回去，机器人保持当前位置

2. `get_ready_state = True`
   - 从当前姿态插值回 `default_joint_pos`

3. `use_policy_action = True`
   - 使用策略输出的 `q_target`

对于 tracking，另外还有一个独立的 `paused` 标志控制参考轨迹是否前进：

- `]`
  - 启用策略
  - 调用 `reset()`
  - `reset()` 会把 `paused=True`

- `space`
  - 仅在 `Tracking` 中额外定义
  - 用来切换 `paused`

- `o`
  - 停止使用策略

- `i`
  - 回默认站姿

所以从当前代码行为看，tracking 不是按下 `]` 就立即开始“播放整段参考轨迹”，而是会先 reset 到第 0 帧并进入 pause；之后再通过空格切换参考轨迹播放。

## 4. tracking 特有部分：参考动作是怎么接进来的

tracking 和普通速度跟踪、站立控制的最大区别，在于策略不仅看“机器人当前状态”，还看“未来若干帧参考动作”。

### 4.1 参考动作由 `StateProcessor` 统一管理

`StateProcessor` 除了订阅 `low_state`，还额外承担 motion source manager 的角色。现在它支持两种后端：

1. 离线 any4hdmi qpos `.npz`
   - 第一次调用 `register_motion_request(...)` 时加载 `MotionDataset`
   - `update(...)` 里按 50 Hz 推进 `motion_t`
   - `get_motion_packet(name)` 返回当前 `motion_t` 对应的未来帧切片

2. 实时 VR teleop
   - 第一次调用 `register_motion_request(...)` 时创建 `RealtimeMotionBuffer`
   - 后台线程持续订阅 `pico_g1_zmq_publisher.py` 发出的 ZMQ motion 流
   - `get_motion_packet(name)` 按“当前时间减去 delay”对齐时间轴，再对 observation 需要的 `future_steps` 做插值

不管底层是离线文件还是实时流，对 observation 暴露的都是同一个 `MotionData` 接口。这意味着：motion 数据不是每个 observation 各自去读，而是集中到 `StateProcessor` 中统一加载或订阅，再切片多次。

### 4.2 `MotionDataset` 的职责

`sim2real/rl_policy/utils/motion.py` 负责把参考动作文件整理成统一格式：

- 通过 any4hdmi manifest 解析 dataset root、MJCF、body names、joint names 和 timestep
- 读取 `motions/*.npz` 中的 qpos，并通过 any4hdmi full-motion dataset 生成 body / joint motion fields
- 必要时插值到 50 Hz
- 把 joint 顺序重排到“Unitree 通用顺序优先，其余关节追加在后面”
- 支持 `get_slice(motion_ids, starts, steps)` 取某一时刻周围的多帧数据

当前实现有一个重要约束：

- `register_motion_request()` 中显式要求 `num_motions == 1`
- 也就是当前 tracking 推理默认只支持单条参考动作序列，而不是运行时切换多段 motion

### 4.3 实时 `RealtimeMotionBuffer` 的职责

`sim2real/rl_policy/utils/motion_buffer.py` 负责把 publisher 发来的实时 motion 数据整理成和离线 `MotionDataset` 一致的 `MotionData` 结构。

发布端当前会发出：

- `smplx_t_ns`
  - retarget 之前收到这帧原始 SMPL-X body 数据的时间戳
- `joint_pos`
- `body_pos_w`
- `body_quat_w`
- 以及兼容用的 `qpos/root_pos/root_quat/dof_pos`

实时 buffer 的核心逻辑是：

1. 按 `smplx_t_ns` 建立时间轴
   - 不是按“subscriber 收到包的时刻”建轴

2. 维护一个有冗余的时间缓存
   - 默认 delay 逻辑是：
   - `max(future_steps) * motion_dt_s + motion_tolerance_s`
   - 对 50 Hz、未来 16 step、默认容错 40 ms 来说，就是 `16 * 20ms + 40ms = 360ms`

3. 当当前时间是 `t` 时
   - 先取 `t - delay` 作为参考的 step 0
   - 再根据 observation 里的 `future_steps`，例如 `[-4, -2, 0, 1, 2, 3, 4, 8, 16]`
   - 推出真正要采样的一组目标时间戳

4. 对这些目标时间戳做插值
   - `pos` 用 lerp
   - `quat` 用 slerp

5. 删除不可能再被用到的旧帧
   - 也就是时间戳早于 `t - delay - abs(min(future_steps)) * dt` 的数据

所以从策略视角看：

- policy 在控制时刻 `t` 看到的参考 motion，本质上是“以 `t - delay` 为中心对齐后”的未来片段
- 这正好对应你说的“为了拿到未来 16 step，实际用的是更早一段时间已经积累好的 motion buffer”

### 4.4 `track.py` 里的 observation 是参考动作编码器

`sim2real/rl_policy/observations/track.py` 中的 `_motion_obs` 是所有 tracking reference observation 的基类。它会：

- 在初始化时向 `StateProcessor` 注册 motion 需求
- 在 `reset()` / `update()` 时从缓存中取对应 slice
- 把 joint/body/root/anchor 的多帧参考数据挂到自身字段上

当前 `policy-tstz89tc-final.yaml` 里，tracking 主要用了三类参考量：

1. `ref_body_pos_future_local`
   - 把未来 body 位置变换到参考 anchor 的局部坐标系
   - 当前 anchor 是 `torso_link`

2. `ref_joint_pos_future`
   - 直接提供未来多帧参考关节角

3. `ref_root_ori_future_b`
   - 提供未来 root 姿态，但会先在 reset 时根据机器人当前朝向和参考轨迹初始朝向求一个 yaw 对齐偏移
   - 这样策略看到的是“对齐到当前机器人朝向后”的未来 root 朝向

完整 yaw 职责划分见 [Yaw Alignment](/reference/yaw-alignment)。

这部分本质上是在做“参考轨迹编码”，把 motion 数据变成策略能消费的 command 类输入。

### 4.5 VR teleop 模式怎么接进来

现在 tracking observation 可以在原有 `motion_path` 方案之外，显式改成 `zmq` backend。这个切换不需要改 policy yaml，直接在启动 policy 时传参数覆盖即可，例如：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-backend zmq
```

其中：

- `motion_backend=zmq`
  - 告诉 `StateProcessor` 不再去读离线 any4hdmi qpos `.npz`
- `motion_zmq_connect`
  - 只有 publisher 和 policy 不在同一台机器时才需要显式指定
- `motion_dt_s`
  - tracking policy 的参考 motion 时间步长，当前通常是 `0.02`
- `motion_tolerance_s`
  - 给 retarget / 网络抖动预留的 buffer 冗余，默认 `0.04`
  - 由 `RealtimeMotionBuffer` 内部自动推导，不再显式传参

因此，VR teleop 链路变成：

```text
PICO / XRoboToolkit
    -> XRobotStreamer
    -> GMR retarget
    -> MuJoCo forward
    -> ZMQ motion stream
    -> StateProcessor / RealtimeMotionBuffer
    -> tracking observations
    -> policy
```

## 5. observation 系统的组织方式

当前 observation 是“注册表 + 配置驱动”的结构。

### 5.1 注册机制

所有 observation 类继承自 `Observation`。子类定义后会自动注册到 `Observation.registry`。

因此，`BasePolicy.setup_observations()` 只需要按配置里的 observation 名称实例化即可。

### 5.2 group 的作用

配置里 observation 被分成多个 group，例如当前 tracking 配置中的：

- `policy`
- `command`

每个 group 会被 `ObsGroup.compute()` 拼成一个 numpy 向量，最后形成：

```python
{
  "policy": ...,
  "command": ...,
}
```

再作为 ONNX 输入。

### 5.3 当前 tracking 配置实际包含的观测

`policy-tstz89tc-final.yaml` 里可以分成两类：

1. 机器人历史状态
   - `root_ang_vel_history`
   - `projected_gravity_history`
   - `joint_pos_history`
   - `joint_vel_history`
   - `prev_actions`

2. 参考轨迹
   - `ref_body_pos_future_local`
   - `ref_joint_pos_future`
   - `ref_root_ori_future_b`

也就是说，tracking policy 的输入可以理解成：

```text
当前机器人状态历史 + 上几步动作历史 + 未来参考轨迹片段
```

### 5.4 一个容易忽略的实现细节

当前推理代码真正依赖的是 observation 条目的“键名”，例如 `root_ang_vel_history`、`ref_joint_pos_future`。配置里的 `_target_`、`noise_std` 更像是训练/导出时留下的元信息，推理侧并不会按 `_target_` 动态 import，也不会应用 `noise_std`。

换句话说，推理端 observation 构建逻辑是：

- 用 YAML 键名在 `Observation.registry` 中查类
- 把其余字段作为构造参数传进去

## 6. 命令发送和下游 bridge

### 6.1 `CommandSender`

`CommandSender` 只做一件事：把目标关节状态发布到 `low_cmd` ZMQ 通道。

消息内容包括：

- `q_target`
- `dq_target`
- `tau_ff`
- `kp`
- `kd`

其中 `kp/kd/default_joint_pos` 都来自 `policy_config`。

这意味着 tracking policy 输出的并不是 torque，而是“关节位置目标 + 固定 PD 增益”的控制命令。

### 6.2 仿真侧 `SimulationBridge`

`sim2real/sim_env/utils/bridge.py` 把统一的 `low_cmd` 落到 MuJoCo 上：

1. 订阅 `low_cmd`
2. 对每个共享关节做

```text
tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
```

3. 再按 `joint_effort_limit` 做裁剪
4. 写入 `mj_data.ctrl`

同时它还会从 MuJoCo 中反向抽取：

- root quaternion
- root angular velocity
- joint positions
- joint velocities
- actuator force

并封装成 `LowStateMessage` 发布到 `low_state`。

所以仿真中 ZMQ 两端的对应关系是：

- policy 进程发布 `low_cmd`
- sim 进程订阅 `low_cmd`
- sim 进程发布 `low_state`
- policy 进程订阅 `low_state`

## 7. sim2sim 和 sim2real 为什么能共用同一套 tracking 代码

因为策略侧的 observation / inference 逻辑不依赖底层执行器。sim2sim 默认走 ZMQ；sim2real 可以走 inline robot I/O，也可以保留 ZMQ bridge。

### sim2sim

下游是：

```text
BaseSimulator + SimulationBridge
```

### sim2real

真机下游由 `--robot-io` 决定：

- `inline`：`BasePolicy` 直接持有 robot object，直接读状态、写命令。
- `zmq`：`BasePolicy` 继续走 `low_state` / `low_cmd` ZMQ，总线另一端可以是 `scripts/real_bridge.py` 或 `scripts/real_bridge_cpp.py`。

三种部署方式的命令集中维护在 [Robot I/O](/reference/robot-io)。

这也是当前仓库 tracking 框架最重要的抽象边界。

## 8. 你给的两个命令在整体框架中的位置

### 命令 1：启动 tracking policy

```bash
uv run sim2real/rl_policy/tracking.py --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml
```

它负责：

- 读取策略配置
- 建 observation
- 读取 `low_state`
- 从 any4hdmi qpos `.npz` 或 live ZMQ 加载参考动作
- 跑 ONNX
- 输出 `low_cmd`

### 命令 2：启动 MuJoCo 仿真

```bash
uv run sim2real/sim_env/base_sim.py
```

它负责：

- 加载机器人场景
- 提供可视化
- 读取 `low_cmd`
- 在 MuJoCo 中执行 PD 控制
- 回传 `low_state`

因此两者必须同时运行，tracking 才是闭环。

### 命令 3：启动 VR teleop motion publisher

如果 tracking 的参考动作不再来自离线 any4hdmi qpos `.npz`，而是来自 VR teleop，那么还需要额外启动：

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py
```

它负责：

- 读取 XR body stream
- 做 SMPL-X -> G1 retarget
- 用 G1 MuJoCo 模型 forward 出 body world pose
  - 把 `joint_pos/body_pos_w/body_quat_w/smplx_t_ns` 按固定 joint/body 顺序发到 ZMQ

这时 tracking 闭环会变成“三条进程链路同时存在”：

- `pico_g1_zmq_publisher.py`
  - 提供参考 motion
- `sim2real/rl_policy/tracking.py`
  - 同时消费 `low_state` 和 live motion，输出 `low_cmd`
- `sim2real/sim_env/base_sim.py` 或 [Robot I/O](/reference/robot-io) 里的真机执行路径
  - 执行 `low_cmd` 并回传 `low_state`

## 9. 当前 tracking 框架的几个实现特点

### 9.1 优点

- 策略层和执行层解耦得比较干净
- observation 系统是配置驱动的，便于加新观测
- sim2sim / sim2real 共用同一套推理逻辑
- motion 数据集中缓存，避免每个 observation 重复加载

### 9.2 当前约束

- 推理侧 motion 目前只支持单条参考序列
- live motion 默认假设 publisher 和 policy 进程的 `time_ns` 基本对齐；如果跨机器或时钟漂移明显，还需要额外做 time offset / sync
- 轨迹播放结束后会自动暂停，而不是自动循环持续 tracking
- joint limit 裁剪代码已经写好，但在 `_rl_step_scheduled()` 里被注释掉了
- tracking 任务目前默认是 position target + PD，不是更底层的 torque policy

## 10. 一句话总结

当前 sim2real 的 tracking 框架，本质上是一个 policy runtime 加可替换 robot I/O 的控制系统：

- `BasePolicy/Tracking` 负责“看当前状态 + 看未来参考轨迹 + 输出关节目标”
- `SimulationBridge` 或 `robot_io` 负责“把控制命令落到仿真或真机”

也就是说，tracking 的任务特性主要体现在 observation 和 motion reference 上，而不是体现在底层执行接口上。
