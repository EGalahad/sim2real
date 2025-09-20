import numpy as np
import argparse
import yaml
from loguru import logger

import sys
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)

class Tracking(BasePolicy):
    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        
        if cur_key == "B":
            self.state_dict["paused"] = not self.state_dict.get("paused", False)
            logger.info(f"Paused state toggled to {self.state_dict['paused']}")
        
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        
        if keycode == "space":
            self.state_dict["paused"] = not self.state_dict.get("paused", False)
            logger.info(f"Paused state toggled to {self.state_dict['paused']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", help="policy config file"
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    policy = Tracking(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
    )
    policy.run()
