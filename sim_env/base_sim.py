import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
import threading
import numpy as np
import time
import argparse
import yaml
from threading import Thread

import sys

sys.path.append(".")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand


class BaseSimulator:
    def __init__(self, robot_config, scene_config, node):
        self.robot_config = robot_config
        self.scene_config = scene_config
        self.node: Node = node
        self.rate = self.node.create_rate(1 / self.scene_config["SIMULATE_DT"])
        self.sim_dt = self.scene_config["SIMULATE_DT"]
        self.viewer_dt = self.scene_config["VIEWER_DT"]

        self.init_scene()
        self.init_unitree_bridge()

        # for more scenes
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = Thread(target=self.SimulationThread)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_scene(self):
        robot_scene = self.scene_config["ROBOT_SCENE"]
        self.mj_model = mujoco.MjModel.from_xml_path(robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        # Enable the elastic band
        if self.scene_config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "h1" in self.robot_config["ROBOT_TYPE"] or "g1" in self.robot_config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model,
                self.mj_data,
                key_callback=self.elastic_band.MujuocoKeyCallback,
                show_left_ui=False,
                show_right_ui=False,
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(
            self.mj_model, self.mj_data, self.robot_config, self.scene_config
        )
        # if self.config["PRINT_SCENE_INFORMATION"]:
        #     self.unitree_bridge.PrintSceneInformation()
        if self.scene_config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.scene_config["JOYSTICK_DEVICE"],
                js_type=self.scene_config["JOYSTICK_TYPE"],
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
        mujoco.mj_step(self.mj_model, self.mj_data)

    def SimulationThread(
        self,
    ):
        sim_cnt = 0
        start_time = time.time()

        self.viewer.cam.azimuth = 30
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 4.0
        self.viewer.cam.lookat = [0.0, 0.0, 1.0]
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            # self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.node.get_logger().info(f"FPS: {100 / (end_time - start_time)}")
                start_time = end_time
            self.rate.sleep()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--scene_config", type=str, default="config/scene/g1_27dof-nohand.yaml", help="scene config file"
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

    simulation = BaseSimulator(robot_config, scene_config, node)
    simulation.sim_thread.start()
