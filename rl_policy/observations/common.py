from .base import Observation

import numpy as np
from typing import Any, Dict
from utils.math import quat_rotate_inverse_numpy


class root_angvel_b(Observation):
    def compute(self) -> np.ndarray:
        base_ang_vel = self.state_processor.root_ang_vel_b
        return base_ang_vel

class root_ang_vel_b(Observation):
    def compute(self) -> np.ndarray:
        base_ang_vel = self.state_processor.root_ang_vel_b
        return base_ang_vel

class root_ang_vel_history(Observation):
    def __init__(self, history_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.root_ang_vel_history = np.zeros((buffer_size, 3))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.root_ang_vel_history = np.roll(self.root_ang_vel_history, 1, axis=0)
        self.root_ang_vel_history[0, :] = self.state_processor.root_ang_vel_b

    def compute(self) -> np.ndarray:
        return self.root_ang_vel_history[self.history_steps].reshape(-1)

class projected_gravity_b(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = np.array([0, 0, -1])

    def compute(self) -> np.ndarray:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        return projected_gravity
    
class projected_gravity_history(Observation):
    def __init__(self, history_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.projected_gravity_history = np.zeros((buffer_size, 3))
        self.v = np.array([0, 0, -1])
    
    def update(self, data: Dict[str, Any]) -> None:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        self.projected_gravity_history = np.roll(self.projected_gravity_history, 1, axis=0)
        self.projected_gravity_history[0, :] = projected_gravity

    def compute(self) -> np.ndarray:
        return self.projected_gravity_history[self.history_steps].reshape(-1)

class joint_pos_multistep(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.joint_pos_multistep = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.joint_pos_multistep = np.roll(self.joint_pos_multistep, 1, axis=0)
        self.joint_pos_multistep[0, :] = self.state_processor.joint_pos

    def compute(self) -> np.ndarray:
        return self.joint_pos_multistep.reshape(-1)

class joint_pos_history(Observation):
    def __init__(self, history_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.joint_pos_multistep = np.zeros((buffer_size, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.joint_pos_multistep = np.roll(self.joint_pos_multistep, 1, axis=0)
        self.joint_pos_multistep[0, :] = self.state_processor.joint_pos

    def compute(self) -> np.ndarray:
        return self.joint_pos_multistep[self.history_steps].reshape(-1)

class joint_vel_multistep(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.joint_vel_multistep = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.joint_vel_multistep = np.roll(self.joint_vel_multistep, 1, axis=0)
        self.joint_vel_multistep[0, :] = self.state_processor.joint_vel

    def compute(self) -> np.ndarray:
        return self.joint_vel_multistep.reshape(-1)

class prev_actions(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.prev_actions = np.zeros((self.env.num_actions, self.steps))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.prev_actions = np.roll(self.prev_actions, 1, axis=1)
        self.prev_actions[:, 0] = data["action"]

    def compute(self) -> np.ndarray:
        return self.prev_actions.reshape(-1)
