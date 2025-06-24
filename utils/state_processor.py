import numpy as np

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.core.channel import ChannelSubscriber

from utils.strings import unitree_joint_names

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_type, dest_joint_names, source_joint_names=unitree_joint_names):
        # Initialize channel subscriber
        if robot_type == "h1" or robot_type == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 1)
        elif robot_type == "g1_29dof" or robot_type == "h1-2_27dof" or robot_type == "h1-2_21dof":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [source_joint_names.index(name) for name in dest_joint_names]

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        # self.tau_est = np.zeros(self.num_dof)
        # self.temp_first = np.zeros(self.num_dof)
        # self.temp_second = np.zeros(self.num_dof)
        self.robot_low_state = None

    def _prepare_low_state(self):
        if not self.robot_low_state:
            return False

        # imu sensor
        imu_state = self.robot_low_state.imu_state
        self.root_quat_b[:] = imu_state.quaternion # w, x, y, z
        self.root_ang_vel_b[:] = imu_state.gyroscope

        # joint encoder
        source_joint_state = self.robot_low_state.motor_state
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = source_joint_state[src_idx].q
            self.joint_vel[dst_idx] = source_joint_state[src_idx].dq
        
        self.robot_state_data = np.concatenate([self.qpos, self.qvel]).reshape(1, -1)
        return True

    def LowStateHandler_go(self, msg: LowState_go):
        self.robot_low_state = msg
    
    def LowStateHandler_hg(self, msg: LowState_hg):
        self.robot_low_state = msg
