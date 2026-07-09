---
name: adapt-policy-to-sim2real
description: Adapt a policy trained in any external codebase into the sim2real codebase for integrated sim2sim/deploy by tracing the training observation/action implementation, exporting one complete ONNX, implementing matching sim2real observations, writing deploy YAML, and validating tracking behavior.
---

# Adapt Policy To sim2real

Use this skill when bringing a policy checkpoint from any training repo into `/home/elijah/Documents/projects/simple-tracking/sim2real`.

## Core Workflow

1. Identify the source of truth in the training codebase:
   - the exact run/play/eval/export command,
   - checkpoint path and training YAML/config,
   - observation implementation,
   - action scaling, action order, default pose, PD gains,
   - robot asset, joint names, body names, and motion format.
2. Trace the actual runtime code path, not only config names. Read the training policy wrapper, observation manager/builder, history buffers, normalization/scaling, action postprocessing, and export code before writing sim2real code.
3. Export a single complete ONNX for sim2real. The ONNX should include all encoder/decoder wiring and tensor concatenation needed for inference. Do not require `sim2real/sim2real/rl_policy/base_policy.py` to load multiple model pieces unless the user explicitly changes that constraint. If the original training codebase only exports split encoder/decoder ONNX files, use the helper script in `scripts/merge_encoder_decoder_onnx.py` or patch the training exporter.
4. Verify ONNX I/O with `onnxruntime` before writing YAML:
   - input names must match `observation.<group>` names in deploy YAML,
   - input dimensions must match the concatenated sim2real obs implementation,
   - output action dimension and order must match `policy_joint_names`,
   - prefer unbatched inputs for the ordinary sim2real BasePolicy path.
5. Implement every training observation in `sim2real/sim2real/rl_policy/observations` or reuse an existing implementation only after proving it is semantically identical.
6. Match training observation semantics exactly:
   - component order and group order,
   - history order and stride,
   - local/world/body frame conventions,
   - quaternion order and scalar position,
   - future command indexing,
   - normalization, clipping, scaling, noise-disabled deploy behavior,
   - default joint offsets and previous-action history.
7. Write a deploy YAML from the training YAML, not from memory. Include observation groups, joint/body names, `policy_joint_names`, action scales, default joint positions, gains, and motion settings.
8. Validate with integrated sim2sim:
   - set robot to motion frame 0,
   - keep policy active during initial pause,
   - rely on the default final-frame hold behavior,
   - save root trajectory when evaluating tracking error,
   - compare robot final root displacement against motion final root displacement in each start frame.
9. Keep a small experiment log with exact commands, policy paths, motion paths, seeds, output files, and summary metrics.

## General Rules

- Prefer the training codebase's real helper APIs over reimplementing math from names alone.
- Never infer joint order from visual order or XML order when the training repo has an explicit action/joint order.
- Do not use `set()` or unordered dict iteration to build deploy input layouts.
- Treat correct shape as necessary but not sufficient. The most common failure is right dimension with wrong order, frame, history direction, or scale.
- Disable training-only noise/randomization in deploy observations.
- Preserve one complete ONNX per policy. If the training repo exports split encoder/decoder ONNX files, merge or re-export them into one graph whose inputs match sim2real observation groups.
- Assume the training codebase may be unmodified upstream code. Do not rely on export flags or helper code that were added in a previous local adaptation unless you verify they exist in that checkout.
- If source motions may need retargeting, inspect `joint_names`, `body_names`, root pose fields, and quaternion order first. If they already match the sim2real robot and qpos layout, skip retargeting.

## ONNX Runtime Compatibility

- G1 GPU deploy may use `onnxruntime-gpu==1.16.0`, whose local wheel supports ONNX IR up to 9. Newer exporters can produce IR 10 / opset 20 models that load on CPU ORT 1.23 but fail on G1 GPU with `Unsupported model IR version: 10, max supported IR version: 9`.
- Prefer re-exporting from the source codebase with `opset <= 19` and `ir_version <= 9` when targeting ORT 1.16.
- If re-export is not practical, convert the already-exported ONNX and compare outputs:

  ```bash
  uv run --with onnx --with onnxruntime --with numpy python skills/adapt-policy-to-sim2real/scripts/convert_onnx_compat.py \
    checkpoints/.../policy.onnx \
    --suffix -ort116 \
    --target-opset 19 \
    --target-ir 9 \
    --compare-runs 100
  ```

- The converter copies an adjacent YAML to the converted ONNX stem by default because sim2real derives the ONNX path from `--policy_config`.
- Validate the converted YAML on G1 with `uv run --no-sync scripts/test_policy_inference.py --policy_config <converted>.yaml --inference_backend onnx-gpu --single`. For non-interactive `g1-rp` SSH, run commands through `bash -lc` so `uv` is on PATH.
- Do not only edit `ir_version`. The conversion is acceptable only when ONNX checker passes and source-vs-converted output comparison matches within tolerance.
- CPU inference should not meaningfully slow down from this compatibility conversion, but benchmark both files if latency matters for the deployment decision.

## Validation Checklist

- ONNX input names and dimensions match deploy YAML observation groups.
- The concatenated obs vector from sim2real has the expected size for each ONNX input.
- Action output dimension equals `len(policy_joint_names)`.
- A one-motion headless integrated sim2sim smoke test runs to final-frame hold.
- Saved root trajectory `.npz` contains robot/motion start and end positions, relative final positions, and final error.
- For batch eval, scripts preserve motion order and validate result path alignment against the run manifest.

## GR00T / SONIC Experience

- Use `GR00T-WholeBodyControl/.venv/bin/python` for export commands.
- Start from `GR00T-WholeBodyControl/play.sh` unless the user gives another source of truth. Do not use release artifacts when the user points to a trained checkpoint.
- Assume the user's GR00T checkout may be the original codebase before local export modifications. In that case it may export split encoder/decoder ONNX files or miss a "single complete ONNX" flag. First inspect `gear_sonic/eval_agent_trl.py`, `active-adaptation/projects/hdmi/hdmi_learning/ppo.py`, and `active-adaptation/projects/hdmi/scripts/play.py` to see what export path is actually available.
- For universal-token SONIC policies, prefer exporting a selected encoder wired to its decoder when the codebase supports it, for example:

  ```bash
  .venv/bin/python gear_sonic/eval_agent_trl.py \
    checkpoint=<checkpoint.pt> \
    +headless=True +num_envs=1 +use_wandb=False \
    +run_eval_loop=False +export_onnx_only=True \
    +export_encoder_name=g1 +export_decoder_name=g1_dyn \
    +export_onnx_name=<policy-name>.onnx \
    +export_unbatched=True
  ```

- If only split selected ONNX files are available, merge them with the bundled helper script:

  ```bash
  uv run --with onnx --with numpy python /home/elijah/.codex/skills/adapt-policy-to-sim2real/scripts/merge_encoder_decoder_onnx.py \
    --encoder <selected_encoder.onnx> \
    --decoder <selected_decoder.onnx> \
    --output <merged_policy.onnx> \
    --input-name obs_dict \
    --output-name action
  ```

  The helper creates one ONNX whose input layout is:

  ```text
  encoder_input | decoder_extra | proprioception
  ```

  By default `decoder_extra=0` and `proprio_dim = decoder_input_dim - encoder_token_dim`. For GR00T g1 selected exports this should normally produce the same deploy input dimension as the sim2real observation group, e.g. `obs_dict[1570] -> action[29]`. If the decoder expects extra non-proprio tokenizer features, pass `--decoder-extra-dim`.

- Release checkpoints may need dummy eval overrides to satisfy Hydra/env initialization even when only exporting:

  ```bash
  .venv/bin/python gear_sonic/eval_agent_trl.py \
    checkpoint=sonic_release/model_step_041550.pt \
    +manager_env.commands.motion.motion_lib_cfg.motion_file=data/lafan_motion_lib/robot \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
    +headless=True +num_envs=1 +use_wandb=False \
    +run_eval_loop=False +export_onnx_only=True \
    +export_encoder_name=g1 +export_decoder_name=g1_dyn \
    +export_onnx_name=policy-sonic-release.onnx \
    +export_unbatched=True
  ```

- For 2026 SONIC g1 deploy, the single ONNX input is usually `obs_dict[1570] -> action[29]`, with layout:

  ```text
  command_multi_future_nonflat | motion_anchor_ori_b_mf_nonflat | actor_obs
  ```

- The observed actor obs order was:

  ```text
  base_ang_vel | joint_pos | joint_vel | actions | gravity_dir
  ```

- `command_multi_future_nonflat` is future joint position plus joint velocity, not body positions.
- IsaacLab observation history is flattened oldest-to-newest; newest-to-oldest can have the right shape but destabilize SONIC.
- Use IsaacLab/action order for `policy_joint_names`, `joint_names_simulation`, and motion joint observations when the exported action order comes from GR00T.
- GR00T export paths may ignore `+exported_policy_path` and still write under the checkpoint's `exported/` directory. Check the actual log line and copy the generated ONNX into `sim2real/checkpoints/...`.
- Splitting a universal encoder and decoder after export can produce an all-tokenizer input shape such as 2681 instead of the selected g1 deploy shape 1570. Use the selected `g1`/`g1_dyn` export for sim2real, or merge only matching selected encoder/decoder ONNX files. After merging, always verify with onnxruntime before writing YAML.
- Recent XRobot raw G1 dumps already include G1 `joint_names`, `body_names`, `root_pos`, `root_rot`, and `dof_pos`; retargeting can be skipped if order matches. Their `root_rot` is `xyzw`, so convert to `wxyz` for any4hdmi qpos.
