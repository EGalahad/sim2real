# MuJoCo Sim2Sim

## DeepMimic Policy (29 DoF)

```bash
# sim
python sim_env/base_sim.py --robot_config=./config/robot/g1.yaml --scene_config=./config/scene/g1_29dof-nohand.yaml

# policy
python rl_policy/deepmimic.py --policy_config=./config/policy/deepmimic_29dof.yaml --robot_config=./config/robot/g1.yaml  --model_path=./checkpoints/exports/G1Track/policy-ez7qrkow-1000.onnx

policy-vb92hcf5-1050.onnx
policy-gwymhkq8-750.onnx
policy-0aiymctb-1000.onnx
```

