import numpy as np
from sim2real.utils.root_metrics import root_final_error, root_endpoint_index


def test_initial_xy_yaw_alignment_preserves_absolute_z():
    robot = np.array([[10., 20., 1.2], [10., 22., 1.4]])
    motion = np.array([[1., 2., 1.], [3., 2., 1.]])
    robot_q = np.tile([np.sqrt(.5), 0., 0., np.sqrt(.5)], (2, 1))
    motion_q = np.tile([1., 0., 0., 0.], (2, 1))
    np.testing.assert_allclose(root_final_error(robot, robot_q, motion, motion_q), [0., 0., .4], atol=1e-12)


def test_endpoint_excludes_full_motion_sentinel_but_keeps_early_failure():
    assert root_endpoint_index(np.array([1, 2, 3, 4]), 5) == 2
    assert root_endpoint_index(np.array([1, 2]), 5) == 1


def test_repeated_terminal_sentinels_are_all_excluded():
    assert root_endpoint_index(np.array([1, 1, 2, 2, 3, 3, 4, 4]), 5) == 5
