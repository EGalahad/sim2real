import mujoco
import numpy as np

from sim2real.config.robots import H2_CFG, get_robot_cfg
from sim2real.config.robots.base import resolve_mjcf_joint_names
from sim2real.sim_env.utils.mjcf import load_sim_model


def test_h2_packaged_model_contract():
    assert get_robot_cfg("h2") is get_robot_cfg("unitree_h2") is H2_CFG
    model = load_sim_model(H2_CFG)
    assert (model.nq, model.nv, model.nu) == (38, 37, 31)
    assert resolve_mjcf_joint_names(model) == H2_CFG.joint_names
    assert len(H2_CFG.default_qpos) == model.nq
    assert set(H2_CFG.joint_names) == set(H2_CFG.joint_effort_limit)
    data = mujoco.MjData(model)
    data.qpos[:] = H2_CFG.default_qpos
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
