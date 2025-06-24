import numpy as np
import rclpy
import time
from scipy.spatial.transform import Rotation as R
import threading

import argparse
import yaml
import sys
from geometry_msgs.msg import Pose
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)

from utils.math import quat_rotate_inverse_numpy, yaw_from_quat, wrap_to_pi, yaw_quat

class PushDoorPolicy(BasePolicy):
    def __init__(
        self,
        robot_config,
        policy_config,
        node,
        model_path,
        rl_rate=50,
    ):
        super().__init__(robot_config, policy_config, node, model_path, rl_rate)
        self.start_time = time.time()
        self.ref_motion_phase = np.zeros(1)
        self.motion_duration_second = self.policy_config["motion_duration_second"]

        # TODO: this should all go inside the observation classes
        self.mocap_data = {}
        
        self.Wall_pose_subscriber = self.node.create_subscription(
            Pose,
            'pose/Wall',
            self.Wall_pose_callback,
            10
        )

        self.Door_pose_subscriber = self.node.create_subscription(
            Pose,
            'pose/Door',
            self.Door_pose_callback,
            10
        )
        
        self.pelvis_pose_subscriber = self.node.create_subscription(
            Pose,
            'pose/pelvis',
            self.pelvis_pose_callback,
            10
        )
        time.sleep(1)
        
    def Wall_pose_callback(self, msg: Pose):
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        quaternion = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.mocap_data["Wall_pos"] = position
        self.mocap_data["Wall_quat"] = quaternion

    def Door_pose_callback(self, msg: Pose):
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        quaternion = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.mocap_data["Door_pos"] = position
        self.mocap_data["Door_quat"] = quaternion

    def pelvis_pose_callback(self, msg: Pose):
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        quaternion = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.mocap_data["pelvis_pos"] = position
        self.mocap_data["pelvis_quat"] = quaternion

    def prepare_obs_for_rl(self):
        t = time.time()
        self.ref_motion_phase[:] = (t - self.start_time) / self.motion_duration_second
        # self.ref_motion_phase[:] = np.clip(self.ref_motion_phase, 0, 1)
        self.ref_motion_phase %= 1.0
        return super().prepare_obs_for_rl()
    
    def _get_obs_ref_motion_phase(self):
        print(f"ref_motion_phase: {self.ref_motion_phase}")
        return self.ref_motion_phase
    
    def _get_obs_door_pos_b(self):
        door_pos_w = self.mocap_data["Wall_pos"]
        root_pos_w = self.mocap_data["pelvis_pos"]
        root_quat_w = self.mocap_data["pelvis_quat"]
        root_quat_yaw_w = yaw_quat(root_quat_w)
        door_pos_b = quat_rotate_inverse_numpy(
            root_quat_yaw_w[None, :], (door_pos_w - root_pos_w)[None, :]
        ).squeeze(0)
        return door_pos_b
    
    def _get_obs_root_yaw(self):
        door_quat_w = self.mocap_data["Wall_quat"]
        root_quat_w = self.mocap_data["pelvis_quat"]
        root_yaw_w = yaw_from_quat(root_quat_w[None, :]).squeeze(0)
        door_yaw_w = yaw_from_quat(door_quat_w[None, :]).squeeze(0)
        root_yaw = wrap_to_pi(root_yaw_w - door_yaw_w - np.pi)
        return root_yaw

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "space":
            self.start_time = time.time()
            self.ref_motion_phase[:] = 0
            self.node.get_logger().info("Resetting ref motion phase")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, default="config/policy/push_door.yaml", help="policy config file"
    )
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--use_jit", action="store_true", default=False, help="use jit")
    parser.add_argument(
        "--use_mocap", action="store_true", default=False, help="use mocap"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    rclpy.init(args=None)
    node = rclpy.create_node("simple_node")

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    rate = node.create_rate(50)

    policy = PushDoorPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=args.model_path,
        node=node,
        rl_rate=50,
    )
    policy.run()
