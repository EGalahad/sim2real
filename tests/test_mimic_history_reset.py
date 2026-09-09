from types import SimpleNamespace
import numpy as np
from sim2real.rl_policy.observations.common import (
    root_ang_vel_history, projected_gravity_history, joint_pos_history,
    joint_vel_history, prev_actions,
)


def test_reset_matches_training_history_fill():
    state = SimpleNamespace(root_ang_vel_b=np.array([1., 2., 3.]),
                            root_quat_w=np.array([1., 0., 0., 0.]),
                            joint_names=["a", "b"], joint_pos=np.array([.2, .3]),
                            joint_vel=np.array([.4, .5]))
    env = SimpleNamespace(state_processor=state, joint_names_simulation=["a", "b"], num_actions=2)
    for cls, value in [(root_ang_vel_history, state.root_ang_vel_b),
                       (projected_gravity_history, np.array([0., 0., -1.])),
                       (joint_pos_history, state.joint_pos), (joint_vel_history, state.joint_vel)]:
        obs = cls(env=env, history_steps=[0, 1, 4])
        obs.reset()
        np.testing.assert_allclose(obs.compute(), np.tile(value, 3))
        obs.update({})
        np.testing.assert_allclose(obs.compute(), np.tile(value, 3))
    obs = prev_actions(env=env, steps=3)
    obs.update({"action": np.ones(2)})
    obs.reset()
    np.testing.assert_array_equal(obs.compute(), np.zeros(6))
