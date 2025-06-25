import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import threading

import argparse
import yaml
import sys
import zmq
sys.path.append(".")

from rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)

from utils.math import quat_rotate_inverse_numpy, yaw_from_quat, wrap_to_pi, yaw_quat

class PushDoorPolicy(BasePolicy):
    def __init__(
        self,
        robot_config,
        policy_config,
        model_path,
        rl_rate=50,
    ):
        super().__init__(robot_config, policy_config, model_path, rl_rate)
        self.start_time = time.time()
        self.ref_motion_phase = np.zeros(1)
        self.motion_duration_second = self.policy_config["motion_duration_second"]

        # TODO: this should all go inside the observation classes
        self.mocap_data = {}
        
        # Initialize ZMQ context and subscribers
        self.zmq_context = zmq.Context()
        self.pose_subscribers = {}
        
        # Use same fixed ports as publisher
        from utils.common import PORTS
        object_names = ["Wall", "Door", "pelvis"]
        object_ports = {name: PORTS[name] for name in object_names}
        
        for obj_name, port in object_ports.items():
            socket = self.zmq_context.socket(zmq.SUB)
            socket.connect(f"tcp://localhost:{port}")
            socket.setsockopt(zmq.SUBSCRIBE, obj_name.encode('utf-8'))
            socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            self.pose_subscribers[obj_name] = socket
            print(f"Subscribing to {obj_name} poses on port {port}")
        
        # Start subscriber threads
        for obj_name in object_ports.keys():
            thread = threading.Thread(target=self._pose_subscriber_thread, args=(obj_name,), daemon=True)
            thread.start()
        
        time.sleep(2)  # Give more time for connections
        print("ZMQ subscriber initialization complete")
        
    def _pose_subscriber_thread(self, obj_name):
        """Thread function to continuously receive pose data for a specific object"""
        socket = self.pose_subscribers[obj_name]
        
        while True:
            try:
                # Receive multipart message [object_name, pose_data]
                message = socket.recv_multipart(zmq.NOBLOCK)
                if len(message) == 2:
                    received_obj_name = message[0].decode('utf-8')
                    pose_bytes = message[1]
                    
                    # Convert bytes back to numpy array with explicit dtype
                    pose_data = np.frombuffer(pose_bytes, dtype=np.float64)
                    
                    if len(pose_data) == 7:  # [x, y, z, qw, qx, qy, qz]
                        position = pose_data[:3].copy()
                        quaternion = pose_data[3:].copy()  # [qw, qx, qy, qz]
                        
                        # Store in mocap_data (keep same naming convention as original)
                        self.mocap_data[f"{obj_name}_pos"] = position
                        self.mocap_data[f"{obj_name}_quat"] = quaternion
                        
                    else:
                        print(f"Invalid pose data length for {obj_name}: {len(pose_data)}")
                        
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in {obj_name} subscriber thread: {str(e)}")
                time.sleep(0.01)

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
            print("Resetting ref motion phase")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, default="config/policy/push_door.yaml", help="policy config file"
    )
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)

    policy = PushDoorPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=args.model_path,
        rl_rate=50,
    )
    policy.run()
