from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim2real.rl_policy.utils.command_sender import ActionManager


class DummyRobotCfg:
    joint_names = ("left_joint", "right_joint")


class FakeRobot:
    def __init__(self) -> None:
        self.commands = []

    def create_zero_command(self) -> SimpleNamespace:
        return SimpleNamespace()

    def write_low_command(self, cmd: SimpleNamespace) -> None:
        self.commands.append(cmd)


def _policy_config() -> dict:
    return {
        "joint_kp": {
            "left_joint": 1.0,
            "right_joint": 2.0,
        },
        "joint_kd": {
            "left_joint": 0.1,
            "right_joint": 0.2,
        },
        "default_joint_pos": {
            "left_joint": -0.3,
            "right_joint": 0.4,
        },
    }


def test_inline_command_uses_policy_gains() -> None:
    robot = FakeRobot()
    manager = ActionManager(
        DummyRobotCfg(),
        _policy_config(),
        robot_io="inline",
        robot=robot,
    )

    q = np.asarray([0.1, -0.2], dtype=np.float32)
    dq = np.zeros(2, dtype=np.float32)
    tau = np.zeros(2, dtype=np.float32)

    manager.send_command(q, dq, tau)
    np.testing.assert_allclose(robot.commands[-1].kp, [1.0, 2.0])
    np.testing.assert_allclose(robot.commands[-1].kd, [0.1, 0.2])
