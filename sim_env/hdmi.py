import mujoco
import time
from typing import Dict
from threading import Thread


import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
sys.path.append(".")
from sim_env.base_sim import BaseSimulator
from utils.common import ZMQPublisher, PORTS


class HDMI(BaseSimulator):
    def __init__(self, robot_config, scene_config):
        super().__init__(robot_config, scene_config)

        if object_joint_name := scene_config.get("object_joint_name", None):
            self.joint_friction = scene_config["joint_friction"]
            self.joint_damping = scene_config["joint_damping"]
            self.joint_stiffness = scene_config["joint_stiffness"]

            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, object_joint_name)
            self.object_joint_qposadr = self.mj_model.jnt_qposadr[joint_id]
            self.object_joint_qveladr = self.mj_model.jnt_dofadr[joint_id]
            self.object_ctrl_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, object_joint_name)
            assert self.object_ctrl_id != -1, "object joint actuator not found"
        else:
            self.object_ctrl_id = -1
    
    def init_publisher(self):
        super().init_publisher()

        self.object_names = self.scene_config["publish_object_names"]
        if len(self.object_names) == 0:
            return
        
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

        # Initialize ZMQ context and publishers with fixed ports
        self.pose_publishers: Dict[str, ZMQPublisher] = {}

        for obj_name in self.object_names:
            port = PORTS[f"{obj_name}_pose"]
            publisher = ZMQPublisher(port)
            self.pose_publishers[obj_name] = publisher
            print(f"Publishing {obj_name} poses on port {port}")

        # Give time for sockets to bind
        time.sleep(1)

        def state_publisher_thread():
            print("Starting state publisher thread")
            
            while True:
                try:
                    start_time = time.perf_counter()
                    for obj_name in self.object_names:
                        pos_sensor_adr = self.pos_sensor_adrs[obj_name]
                        quat_sensor_adr = self.quat_sensor_adrs[obj_name]
                        pos = self.mj_data.sensordata[pos_sensor_adr:pos_sensor_adr+3]
                        quat = self.mj_data.sensordata[quat_sensor_adr:quat_sensor_adr+4]
                        self.pose_publishers[obj_name].publish_pose(pos, quat)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    if elapsed_time < 1.0 / self.publish_rate:
                        time.sleep((1.0 / self.publish_rate) - elapsed_time)
                    else:
                        print(f"Warning: State publishing took too long: {elapsed_time:.6f} seconds")
                except Exception as e:
                    print(f"Error in state publisher thread: {str(e)}")
                    time.sleep(0.1)

        # Start state publishing thread
        self.publish_rate = 100  # Hz
        self.state_thread = Thread(target=state_publisher_thread, daemon=True)
        self.state_thread.start()

    def sim_step(self):
        self.sim_bridge.publish_low_state()
        if self.scene_config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                pos = self.mj_data.xpos[self.band_attached_link]
                lin_vel = self.mj_data.cvel[self.band_attached_link, 3:6]
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
                    self.elastic_band.Advance(pos, lin_vel)
                )
        self.sim_bridge.compute_torques()
        self.mj_data.ctrl[:] = self.sim_bridge.torques

        if self.object_ctrl_id != -1:
            # door joint resistance
            door_joint_qvel = self.mj_data.qvel[self.object_joint_qveladr]
            door_joint_qpos = self.mj_data.qpos[self.object_joint_qposadr]
            door_ctrl = (
                - self.joint_friction * np.sign(door_joint_qvel) * (np.abs(door_joint_qvel) > 0.01)
                + self.joint_stiffness * (0.0 - door_joint_qpos)
                + self.joint_damping * (0.0 - door_joint_qvel)
            )
            self.mj_data.ctrl[self.object_ctrl_id] = door_ctrl
            # print(f"door_torque: {door_ctrl}")

        mujoco.mj_step(self.mj_model, self.mj_data)

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--scene_config", type=str, default="config/scene/g1_29dof_nohand-door.yaml", help="scene config file"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.scene_config) as file:
        scene_config = yaml.load(file, Loader=yaml.FullLoader)

    simulation = HDMI(robot_config, scene_config)
    simulation.sim_thread.start()
