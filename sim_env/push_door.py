import sys

sys.path.append(".")
from sim_env.base_sim import BaseSimulator
import numpy as np
import mujoco

np.set_printoptions(precision=3, suppress=True)


class PushDoor(BaseSimulator):
    def __init__(self, robot_config, scene_config):
        super().__init__(robot_config, scene_config)

        self.door_friction = scene_config["door_friction"]
        self.door_damping = scene_config["door_damping"]
        self.door_stiffness = scene_config["door_stiffness"]

        door_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "door_joint")
        self.door_joint_qposadr = self.mj_model.jnt_qposadr[door_joint_id]
        self.door_joint_qveladr = self.mj_model.jnt_dofadr[door_joint_id]
        self.door_ctrl_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "door_joint")
        assert self.door_ctrl_id != -1, "Door joint actuator not found"

    def sim_step(self):
        self.unitree_bridge.PublishLowState()
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
        # print(f"door_torque: {door_ctrl}")

        mujoco.mj_step(self.mj_model, self.mj_data)

if __name__ == "__main__":
    import argparse
    import yaml
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

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

    simulation = PushDoor(robot_config, scene_config)
    simulation.sim_thread.start()
