from pathlib import Path

import numpy as np

from scripts.tracking_experiment.compute_tracking_metrics import (
    RETURN_BODY_NAMES,
    TERMINATION_ANCHOR_BODY_NAME,
    _compute_one,
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

    score, mean_reward, terminated, reason, _, termination_idx = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 2.0
    assert mean_reward == 2.0
    assert not terminated
    assert reason == "motion_end"
    assert termination_idx == 3

    robot_pos[2:, names.index(TERMINATION_ANCHOR_BODY_NAME), 2] = 1.0
    robot_pos[3] = 100.0
    (
        score,
        mean_reward,
        terminated,
        reason,
        termination_motion_t,
        termination_idx,
    ) = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 1.0
    assert score < mean_reward < 2.0
    assert terminated
    assert reason == "anchor_pos_z"
    assert termination_motion_t == 3
    assert termination_idx == 2

    robot_pos = motion_pos.copy()
    robot_quat = motion_quat.copy()
    robot_quat[1, names.index(TERMINATION_ANCHOR_BODY_NAME)] = (0.0, 1.0, 0.0, 0.0)
    (
        score,
        mean_reward,
        terminated,
        reason,
        termination_motion_t,
        termination_idx,
    ) = _normalized_tracking_return(
        robot_pos, robot_quat, motion_pos, motion_quat, names, motion_t, 4
    )
    assert score == 0.5
    assert score < mean_reward < 2.0
    assert terminated
    assert reason == "anchor_ori"
    assert termination_motion_t == 2
    assert termination_idx == 1


def test_progress_and_return_share_pelvis_termination(tmp_path: Path) -> None:
    names, motion_pos, motion_quat, motion_t = _perfect_trajectory()
    robot_pos = motion_pos.copy()
    robot_quat = motion_quat.copy()
    robot_pos[2:, names.index(TERMINATION_ANCHOR_BODY_NAME), 2] = 1.0
    root_pos = np.zeros((4, 3), dtype=np.float32)
    root_quat = np.zeros((4, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        body_names=np.asarray(names),
        robot_body_pos_w=robot_pos,
        robot_body_quat_w=robot_quat,
        motion_body_pos_w=motion_pos,
        motion_body_quat_w=motion_quat,
        robot_root_pos_w=root_pos,
        robot_root_quat_w=root_quat,
        motion_root_pos_w=root_pos,
        motion_root_quat_w=root_quat,
        motion_t=motion_t,
        motion_length=np.asarray(5),
        policy_config=np.asarray("policy.yaml"),
        motion_path=np.asarray("motion.npz"),
        seed=np.asarray(0),
    )

    row = _compute_one(path)

    assert row["termination_idx"] == 2
    assert row["termination_motion_t"] == row["return_termination_motion_t"] == 3
    assert (
        row["termination_reason"]
        == row["return_termination_reason"]
        == "anchor_pos_z"
    )
    assert row["progress"] == 0.75
    assert row["normalized_tracking_return"] == 1.0
