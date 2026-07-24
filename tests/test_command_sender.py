from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim2real.rl_policy.utils.command_sender import ActionManager


class DummyRobotCfg:
    joint_names = ("left_joint", "right_joint")


class FakeRobotIO:
    def __init__(self) -> None:
        self.commands = []

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        self.commands.append(
            SimpleNamespace(
                q_target=np.asarray(q_target).copy(),
                dq_target=np.asarray(dq_target).copy(),
                tau_ff=np.asarray(tau_ff).copy(),
                kp=np.asarray(kp).copy(),
                kd=np.asarray(kd).copy(),
            )
        )


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


def test_action_manager_passes_resolved_policy_gains_to_robot_io() -> None:
    robot_io = FakeRobotIO()
    manager = ActionManager(
        DummyRobotCfg(),
        _policy_config(),
        robot_io=robot_io,
    )

    q = np.asarray([0.1, -0.2], dtype=np.float32)
    dq = np.zeros(2, dtype=np.float32)
    tau = np.zeros(2, dtype=np.float32)

    manager.send_command(q, dq, tau)
    np.testing.assert_allclose(robot_io.commands[-1].kp, [1.0, 2.0])
    np.testing.assert_allclose(robot_io.commands[-1].kd, [0.1, 0.2])
