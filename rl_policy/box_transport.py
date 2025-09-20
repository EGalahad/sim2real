import numpy as np
import argparse
import yaml
from loguru import logger

import sys
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)

from observations.box_transport import *

class BoxTransport(BasePolicy):
    pass
    # def handle_joystick_button(self, cur_key):
    #     super().handle_joystick_button(cur_key)
        
    #     if cur_key == "B":
    #         self.state_dict["paused"] = not self.state_dict.get("paused", False)
    #         logger.info(f"Paused state toggled to {self.state_dict['paused']}")
        
    # def handle_keyboard_button(self, keycode):
    #     super().handle_keyboard_button(keycode)
        
    #     if keycode == "space":
    #         self.state_dict["paused"] = not self.state_dict.get("paused", False)
    #         logger.info(f"Paused state toggled to {self.state_dict['paused']}")


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

    # obs_cfg = policy_config["observation"]
    # obs_cfg["policy"]["object_pos_b"]["object_name"] = "box_small"
    # obs_cfg["policy"]["object_ori_b"]["object_name"] = "box_small"
    # obs_cfg["policy"]["object_pos_b"]["root_body_name"] = "pelvis"
    # obs_cfg["policy"]["object_ori_b"]["root_body_name"] = "pelvis"
    # obs_cfg["policy"]["ref_contact_pos_b"]["object_name"] = "box_small"
    # obs_cfg["policy"]["ref_contact_pos_b"]["root_body_name"] = "pelvis"
    # obs_cfg["policy"]["ref_contact_pos_b"]["contact_target_pos_offset"] = [[0.0, 0.1, 0.0], [0.0,-0.1, 0.0]]
    print(policy_config["observation"])

    policy = BoxTransport(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
    )
    policy.run()
