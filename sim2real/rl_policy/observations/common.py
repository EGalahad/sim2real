from .base import Observation

import numpy as np
from typing import Any, Dict, List, Sequence
from sim2real.utils.math import quat_rotate_inverse_numpy
from sim2real.utils.strings import resolve_matching_names


def sort_names_by_preferred_order(
    matched_names: Sequence[str],
    preferred_names: Sequence[str],
) -> List[str]:
    matched_names = list(matched_names)
    preferred_names = list(preferred_names)
    ordered_names = [name for name in preferred_names if name in matched_names]
    if len(ordered_names) != len(matched_names):
        missing_names = [name for name in matched_names if name not in preferred_names]
        raise ValueError(
            f"Failed to resolve names {missing_names} in preferred order."
        )
    return ordered_names


def _get_simulation_joint_selection(env, joint_names: str | List[str]):
    _, matched_joint_names = resolve_matching_names(
        joint_names,
        env.state_processor.joint_names,
        preserve_order=True,
    )
    ordered_joint_names = sort_names_by_preferred_order(
        matched_joint_names,
        env.joint_names_simulation,
    )

    joint_ids = [
        env.state_processor.joint_names.index(joint_name)
        for joint_name in ordered_joint_names
    ]
    return joint_ids, ordered_joint_names


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

class projected_gravity_history(Observation):
    def __init__(self, history_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.projected_gravity_history = np.zeros((buffer_size, 3))
        self.v = np.array([0, 0, -1])
    
    def update(self, data: Dict[str, Any]) -> None:
        base_quat = self.state_processor.root_quat_w
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        self.projected_gravity_history = np.roll(self.projected_gravity_history, 1, axis=0)
        self.projected_gravity_history[0, :] = projected_gravity

    def compute(self) -> np.ndarray:
        return self.projected_gravity_history[self.history_steps].reshape(-1)

class joint_pos_history(Observation):
    def __init__(self, history_steps: int, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1

        self.joint_ids, self.joint_names = _get_simulation_joint_selection(
            self.env,
            joint_names,
        )
        self.joint_pos_multistep = np.zeros((buffer_size, len(self.joint_ids)))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.joint_pos_multistep = np.roll(self.joint_pos_multistep, 1, axis=0)
        self.joint_pos_multistep[0, :] = self.state_processor.joint_pos[self.joint_ids]

    def compute(self) -> np.ndarray:
        return self.joint_pos_multistep[self.history_steps].reshape(-1)

class joint_vel_history(Observation):
    def __init__(self, history_steps: int, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1

        self.joint_ids, self.joint_names = _get_simulation_joint_selection(
            self.env,
            joint_names,
        )
        self.joint_vel_multistep = np.zeros((buffer_size, len(self.joint_ids)))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.joint_vel_multistep = np.roll(self.joint_vel_multistep, 1, axis=0)
        self.joint_vel_multistep[0, :] = self.state_processor.joint_vel[self.joint_ids]

    def compute(self) -> np.ndarray:
        return self.joint_vel_multistep[self.history_steps].reshape(-1)

class prev_actions(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.prev_actions = np.zeros((self.env.num_actions, self.steps))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.prev_actions = np.roll(self.prev_actions, 1, axis=1)
        self.prev_actions[:, 0] = data["action"]

    def compute(self) -> np.ndarray:
        # Match training flatten order from [steps, action_dim].
        return self.prev_actions.T.reshape(-1)
