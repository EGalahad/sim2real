import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from general_motion_retargeting import GeneralMotionRetargeting
from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import resolve_mjcf_joint_names
from sim2real.teleop.pico_retarget_pub import _register_gmr_robot, _resolve_gmr_target_robot


def test_h2_xrobot_upright_retarget():
    cfg = get_robot_cfg("h2")
    target = _resolve_gmr_target_robot(cfg)
    assert target == "unitree_h2"
    _register_gmr_robot(cfg, target)
    retarget = GeneralMotionRetargeting("xrobot", target, actual_human_height=1.8, verbose=False)
    model = retarget.model
    assert model.nq == 38
    assert resolve_mjcf_joint_names(model) == cfg.joint_names
    data = mujoco.MjData(model)
    data.qpos[:] = cfg.default_qpos
    mujoco.mj_forward(model, data)
    foot_ids = [model.body(name).id for name in ("left_ankle_pitch_link", "right_ankle_pitch_link")]
    data.qpos[2] -= min(data.xpos[i,2] for i in foot_ids) - 0.08
    mujoco.mj_forward(model, data)
    human = {}
    for body, (source, _, _, pos_offset, rot_offset) in retarget.ik_match_table1.items():
        robot_rotation = Rotation.from_quat(data.xquat[model.body(body).id], scalar_first=True)
        rotation = robot_rotation * Rotation.from_quat(rot_offset, scalar_first=True).inv()
        human[source] = [data.xpos[model.body(body).id] - robot_rotation.apply(pos_offset), rotation.as_quat(scalar_first=True)]
    for _ in range(30):
        qpos = retarget.retarget(copy.deepcopy(human))
        assert qpos.shape == (38,) and np.isfinite(qpos).all()
    assert 0.8 < qpos[2] < 1.4
    up = Rotation.from_quat(qpos[3:7], scalar_first=True).apply([0.,0.,1.])
    assert up[2] > 0.95
    print("H2 synthetic retarget: qpos38, joints31, pelvis height",qpos[2],"up_z",up[2])
