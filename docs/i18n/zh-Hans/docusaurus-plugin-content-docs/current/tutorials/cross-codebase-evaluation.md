# 跨代码库动作跟踪评测

排行榜使用冻结的 MotionDecode 子集和统一 MuJoCo sim2sim。策略使用 CPU ONNX Runtime，保存完整轨迹后离线计算指标。

## 安装与策略接入

```bash
uv sync --extra inference-cpu
```

新策略可使用仓库的 [adapt-policy-to-sim2real skill](https://github.com/EGalahad/sim2real/blob/main/.agents/skills/adapt-policy-to-sim2real/SKILL.md)，完成 ONNX 导出、观测历史与动作语义对齐、deploy YAML 和 sim2sim 验证。仅有 ONNX 不够，观测、关节顺序、控制增益和 action scale 都必须匹配训练。

## 冻结数据集

| 数据集 | 数量 | 用途 |
|---|---:|---|
| [Locomotion-80](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-locomotion) | 80 | 主要评测，含全局 root |
| [Manipulation-48](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-manipulation) | 48 | 主要评测，body/wrist 跟踪 |
| [Ground-60](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-ground) | 60 | 补充评测 |
| [Dance-40](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-motiondecode-dance) | 40 | 补充评测 |

原始数据来自 **ChingMu** 的 [CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode)，revision 为 `80b489e0378b60bb44d495d5437475ce0e084283`，遵循原始 ChingMu 使用条款。这些是小规模研究评测子集，不是完整数据集。G1 轨迹转换为 120 Hz NPZ，没有重新 retarget。

每组包含固定顺序的 `motion-list.txt`、原始选择清单、motion 哈希、`manifest.json` 和 `checksums.json`。不要重新抽样，也不要将 G1 关节直接解释成其他机器人关节。公开 revision 固定在 [motiondecode.json](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/manifests/motiondecode.json)，脚本下载指定版本并逐文件校验。

## 运行评测

在仓库根目录执行，把 YAML 替换为已接入策略的 deploy 配置：

```bash
uv run scripts/tracking_experiment/run_motiondecode_eval.py \
  --download \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_motiondecode \
  --max-workers 8
```

默认评测 Locomotion 和 Manipulation。四组全部评测：

```bash
uv run scripts/tracking_experiment/run_motiondecode_eval.py \
  --download --splits locomotion manipulation ground dance \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_motiondecode_all \
  --max-workers 8
```

重复 `--policy name=path` 可比较多个策略。默认数据目录为 `datasets/motiondecode`，可通过 `--datasets-root` 修改。`--dry-run` 下载、校验并打印命令；`--skip-existing` 复用可读轨迹并重算指标，仅在策略、仿真器和数据版本不变时使用。

每组输出完整轨迹、`runs.csv`、`tracking_metrics.csv/json` 和 `summary.json`；总目录的 `motiondecode_summary.json` 收集各组结果和数据版本。请同时保留 Git commit、依赖 lock、策略文件和原始轨迹。

## 统一口径

- seed 0，50 Hz 策略帧，初始暂停为 0，完整 motion，无运行时长上限。
- 单帧失败条件：pelvis 高度误差大于 0.3 m，或 projected gravity Z 误差大于 0.8。不启用 root position termination。这些是离线评分条件，与训练终止设置无关。
- Progress 和 Tracking Return 使用同一个失败帧，return 不含失败帧。Return 使用 BeyondMimic body pos/ori 的 0.3/0.4 尺度，除以完整参考长度，范围 0–2。
- Body/wrist position 使用米，orientation 使用四元数角距离（弧度）。Local 对齐移除当前 pelvis XY/yaw，保留绝对 Z。误差统计失败前帧；wrist 使用双侧 wrist-yaw link。Manipulation 不测物体任务成功率。
- Global Root 仅报告 Locomotion。`root_final_error_norm` 在固定初始 XY/yaw 对齐、保留绝对世界 Z 后，计算最后一个有效同时间帧的 XYZ 距离，排除终止 sentinel。XY 和绝对 Z 仅作为分量。必须同时报告 progress，因为提前失败的终点不是完整 motion 终点。
- 新结果标记 `initial_xy_yaw_aligned_endpoint_xyz_v1`。旧的全四元数对齐、轨迹均值 root 结果需要从原始轨迹重算，不能改标签后直接比较。
- 每组内 motion 等权，包括 return。主要综合值为 Locomotion 与 Manipulation 组均值的等权平均；Ground/Dance 分开报告，root 不混入 Manipulation。

## 历史评测

LAFAN-40、PHUMA-30、Root-90 是补充历史面板，保留在 `run_canonical_tracking_eval.py` 和 `scripts/tracking_experiment/manifests/`，不代替 MotionDecode。对应 HF 数据集为 `elijahgalahad/any4hdmi-g1-lafan`、`elijahgalahad/any4hdmi-g1-phuma30`、`elijahgalahad/any4hdmi-g1-root90`。历史绘图脚本使用冻结 CSV，重新画图不等于重新运行策略。
