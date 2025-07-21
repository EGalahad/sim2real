import numpy as np
np.set_printoptions(precision=3, suppress=True)

import sys
sys.path.append(".")
from sim_env.base_sim import BaseSimulator

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--scene_config", type=str, default="config/scene/g1_29dof_eef_L-box.yaml", help="scene config file"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.scene_config) as file:
        scene_config = yaml.load(file, Loader=yaml.FullLoader)

    simulation = BaseSimulator(robot_config, scene_config)
    simulation.sim_thread.start()
