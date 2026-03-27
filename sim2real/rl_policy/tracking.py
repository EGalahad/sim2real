import numpy as np
import argparse
from copy import deepcopy
import yaml
from loguru import logger

from sim2real.rl_policy.base_policy import BasePolicy
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


def _apply_runtime_motion_config(
    policy_config,
    motion_backend: str,
    motion_zmq_connect: str,
    motion_zmq_topic: str,
    motion_zmq_hwm: int,
    motion_dt_s: float,
    motion_tolerance_s: float,
):
    policy_config = deepcopy(policy_config)
    motion_cfg = policy_config.setdefault("motion", {})
    motion_cfg["motion_backend"] = motion_backend
    if motion_backend == "zmq":
        motion_cfg["motion_zmq_connect"] = motion_zmq_connect
        motion_cfg["motion_zmq_topic"] = motion_zmq_topic
        motion_cfg["motion_zmq_hwm"] = motion_zmq_hwm
        motion_cfg["motion_dt_s"] = motion_dt_s
        motion_cfg["motion_tolerance_s"] = motion_tolerance_s

    return policy_config


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
    parser.add_argument(
        "--motion_backend",
        type=str,
        default="npz",
        choices=["npz", "zmq"],
        help="motion backend to use at runtime",
    )
    parser.add_argument(
        "--motion_zmq_connect",
        type=str,
        default="tcp://127.0.0.1:28701",
        help="ZMQ endpoint for live motion publisher",
    )
    parser.add_argument(
        "--motion_zmq_topic",
        type=str,
        default="g1",
        help="ZMQ topic for live motion publisher",
    )
    parser.add_argument(
        "--motion_zmq_hwm",
        type=int,
        default=1,
        help="ZMQ receive high-water mark",
    )
    parser.add_argument(
        "--motion_dt_s",
        type=float,
        default=0.02,
        help="motion timestep in seconds",
    )
    parser.add_argument(
        "--motion_tolerance_s",
        type=float,
        default=0.04,
        help="extra delay tolerance in seconds",
    )
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="onnxruntime execution provider",
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    policy_config = _apply_runtime_motion_config(
        policy_config=policy_config,
        motion_backend=args.motion_backend,
        motion_zmq_connect=args.motion_zmq_connect,
        motion_zmq_topic=args.motion_zmq_topic,
        motion_zmq_hwm=args.motion_zmq_hwm,
        motion_dt_s=args.motion_dt_s,
        motion_tolerance_s=args.motion_tolerance_s,
    )
    if args.motion_backend == "zmq":
        logger.info(
            "Using runtime motion_backend=zmq "
            f"connect={args.motion_zmq_connect} topic={args.motion_zmq_topic}"
        )

    policy = Tracking(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
        onnx_provider=args.onnx_provider,
    )
    policy.run()
