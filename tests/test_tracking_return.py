import numpy as np

from scripts.tracking_experiment.compute_tracking_metrics import (
    RETURN_ANCHOR_BODY_NAME,
    RETURN_BODY_NAMES,
    _normalized_tracking_return,
)


def _perfect_trajectory() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(RETURN_BODY_NAMES)
    pos = np.zeros((4, len(names), 3), dtype=np.float32)
    quat = np.zeros((4, len(names), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    motion_t = np.arange(1, 5, dtype=np.int32)
    return names, pos, quat, motion_t


def test_normalized_tracking_return_counts_only_pretermination_reward() -> None:
    names, motion_pos, motion_quat, motion_t = _perfect_trajectory()
    robot_pos = motion_pos.copy()
    robot_quat = motion_quat.copy()

    score, mean_reward, terminated, reason, _ = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 2.0
    assert mean_reward == 2.0
    assert not terminated
    assert reason == "motion_end"

    robot_pos[2:, names.index(RETURN_ANCHOR_BODY_NAME), 2] = 1.0
    robot_pos[3] = 100.0
    score, mean_reward, terminated, reason, termination_motion_t = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 1.0
    assert score < mean_reward < 2.0
    assert terminated
    assert reason == "anchor_pos_z"
    assert termination_motion_t == 3

    robot_pos = motion_pos.copy()
    robot_quat = motion_quat.copy()
    robot_quat[1, names.index(RETURN_ANCHOR_BODY_NAME)] = (0.0, 1.0, 0.0, 0.0)
    score, mean_reward, terminated, reason, termination_motion_t = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 0.5
    assert score < mean_reward < 2.0
    assert terminated
    assert reason == "anchor_ori"
    assert termination_motion_t == 2
