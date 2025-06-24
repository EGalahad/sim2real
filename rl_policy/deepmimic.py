import numpy as np
import rclpy
import time
import threading

import argparse
import yaml
import sys
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


class DeepMimicPolicy(BasePolicy):
    def __init__(
        self,
        robot_config,
        policy_config,
        node,
        model_path,
        rl_rate=50,
    ):
        super().__init__(
            robot_config, policy_config, node, model_path, rl_rate
        )
        self.start_time = time.time()
        self.ref_motion_phase = np.zeros(1)
        self.motion_duration_second = self.policy_config["motion_duration_second"]

    def prepare_obs_for_rl(self):
        t = time.time()
        self.ref_motion_phase[:] = (t - self.start_time) / self.motion_duration_second
        # self.ref_motion_phase[:] = np.clip(self.ref_motion_phase, 0, 1)
        self.ref_motion_phase %= 1.0
        return super().prepare_obs_for_rl()
    
    def _get_obs_ref_motion_phase(self):
        print("ref_motion_phase:", self.ref_motion_phase)
        return self.ref_motion_phase
    
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "space":
            self.start_time = time.time()
            self.ref_motion_phase[:] = 0
            self.node.get_logger().info("Resetting ref motion phase")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--policy_config", type=str, default="config/policy/deepmimic_29dof.yaml", help="policy config file"
    )
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    rclpy.init(args=None)
    node = rclpy.create_node("deepmimic_policy")

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    policy = DeepMimicPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=args.model_path,
        node=node,
        rl_rate=50,
    )
    policy.run()
