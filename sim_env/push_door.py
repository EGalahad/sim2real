import sys

sys.path.append(".")
from sim_env.base_sim import BaseSimulator
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import rclpy
import numpy as np
from threading import Thread
import mujoco
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=3, suppress=True)


class PushDoor(BaseSimulator):
    def __init__(self, robot_config, scene_config, node):
        self.object_names = ["Wall", "Door", "pelvis"]
        super().__init__(robot_config, scene_config, node)

        self.door_friction = scene_config["door_friction"]
        self.door_damping = scene_config["door_damping"]
        self.door_stiffness = scene_config["door_stiffness"]

        door_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "door_joint")
        self.door_joint_qposadr = self.mj_model.jnt_qposadr[door_joint_id]
        self.door_joint_qveladr = self.mj_model.jnt_dofadr[door_joint_id]
        self.door_ctrl_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "door_joint")
        assert self.door_ctrl_id != -1, "Door joint actuator not found"

    def init_publisher(self):
        def find_sensor_id(name):
            return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        # Get sensor indices
        self.pos_sensor_adrs = {}
        self.quat_sensor_adrs = {}
        for obj_name in self.object_names:
            pos_sensor_id = find_sensor_id(f"{obj_name}_pos")
            quat_sensor_id = find_sensor_id(f"{obj_name}_quat")
            assert pos_sensor_id != -1, f"Sensor {obj_name}_pos not found"
            assert quat_sensor_id != -1, f"Sensor {obj_name}_quat not found"
            self.pos_sensor_adrs[obj_name] = self.mj_model.sensor_adr[pos_sensor_id]
            self.quat_sensor_adrs[obj_name] = self.mj_model.sensor_adr[quat_sensor_id]

        # Create additional publishers
        self.pose_pubs = {}
        for obj_name in self.object_names:
            self.pose_pubs[obj_name] = self.node.create_publisher(
                Pose, f"/pose/{obj_name}", 10
            )

        # Start state publishing thread
        self.publish_rate = 100  # Hz
        self.state_thread = Thread(target=self.state_publisher_thread, daemon=True)
        self.state_thread.start()

    def state_publisher_thread(self):
        rate = self.node.create_rate(self.publish_rate)
        print("Starting state publisher thread")

        while rclpy.ok():
            try:
                for obj_name in self.object_names:
                    pos = self.mj_data.sensordata[
                        self.pos_sensor_adrs[obj_name] : self.pos_sensor_adrs[obj_name] + 3
                    ]
                    quat = self.mj_data.sensordata[
                        self.quat_sensor_adrs[obj_name] : self.quat_sensor_adrs[obj_name] + 4
                    ]
                    msg = Pose()
                    msg.position.x = pos[0]
                    msg.position.y = pos[1]
                    msg.position.z = pos[2]
                    msg.orientation.w = quat[0]
                    msg.orientation.x = quat[1]
                    msg.orientation.y = quat[2]
                    msg.orientation.z = quat[3]
                    self.pose_pubs[obj_name].publish(msg)

                rate.sleep()
            except Exception as e:
                self.node.get_logger().error(
                    f"Error in state publisher thread: {str(e)}"
                )

    def sim_step(self):
        self.unitree_bridge.PublishLowState()
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.scene_config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
                    self.elastic_band.Advance(
                        self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                    )
                )
        self.unitree_bridge.compute_torques()
        self.mj_data.ctrl[:] = self.unitree_bridge.torques

        # door joint resistance
        door_joint_qvel = self.mj_data.qvel[self.door_joint_qveladr]
        door_joint_qpos = self.mj_data.qpos[self.door_joint_qposadr]
        door_ctrl = (
            - self.door_friction * np.sign(door_joint_qvel) * (np.abs(door_joint_qvel) > 0.01)
            + self.door_stiffness * (0.0 - door_joint_qpos)
            + self.door_damping * (0.0 - door_joint_qvel)
        )
        self.mj_data.ctrl[self.door_ctrl_id] = door_ctrl
        print(f"door_torque: {door_ctrl}")

        mujoco.mj_step(self.mj_model, self.mj_data)

if __name__ == "__main__":
    import argparse
    import yaml
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    import threading

    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--scene_config", type=str, default="config/scene/g1_29dof-eef_sphere-door.yaml", help="scene config file"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.scene_config) as file:
        scene_config = yaml.load(file, Loader=yaml.FullLoader)

    if robot_config.get("INTERFACE", None):
        ChannelFactoryInitialize(robot_config["DOMAIN_ID"], robot_config["INTERFACE"])
    else:
        ChannelFactoryInitialize(robot_config["DOMAIN_ID"])
    rclpy.init(args=None)
    node = rclpy.create_node("sim_mujoco")

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    simulation = PushDoor(robot_config, scene_config, node)
    simulation.sim_thread.start()
