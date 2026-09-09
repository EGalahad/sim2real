import mujoco
import numpy as np

from sim2real.config.robots import H2_CFG, get_robot_cfg
from sim2real.config.robots.base import resolve_mjcf_joint_names
from sim2real.sim_env.utils.mjcf import load_sim_model


def test_h2_shared_model_contract():
    assert get_robot_cfg("h2") is get_robot_cfg("unitree_h2") is H2_CFG
    model = load_sim_model(H2_CFG)
    assert (model.nq, model.nv, model.nu) == (38, 37, 31)
    assert resolve_mjcf_joint_names(model) == H2_CFG.joint_names
    assert len(H2_CFG.default_qpos) == model.nq
    assert set(H2_CFG.joint_names) == set(H2_CFG.joint_effort_limit)
    feet = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.endswith("_collision"):
            assert model.geom_contype[index] == model.geom_conaffinity[index] == 1
            if "_foot" in name:
                feet.append(index)
                assert model.geom_condim[index] == 3
                assert model.geom_priority[index] == 1
                assert model.geom_friction[index, 0] == 0.6
            else:
                assert model.geom_condim[index] == 1
                assert model.geom_priority[index] == 0
    assert len(feet) == 14
    data = mujoco.MjData(model)
    data.qpos[:] = H2_CFG.default_qpos
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
