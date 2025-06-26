import numpy as np
import argparse
import yaml
import sys
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


class DeepMimicPolicy(BasePolicy):
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "space" or keycode == "]":
            self.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--policy_config", type=str, default="config/policy/deepmimic_29dof.yaml", help="policy config file"
    )
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    policy_config["observation"]["command"]["ref_motion_phase"]["motion_duration_second"] = policy_config["motion_duration_second"]

    policy = DeepMimicPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
    )
    policy.run()
