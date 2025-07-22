from .base import Observation

import numpy as np
from typing import Any, Dict

class command_lin_vel_b(Observation):
    def __init__(self, env: Any, **kwargs: Dict[str, Any]):
        super().__init__(env, **kwargs)
        self.command_lin_vel_b = np.zeros(2, dtype=np.float32)
        self.alpha = 0.04

    def reset(self):
        self.command_lin_vel_b[:] = 0.0
    
    def update(self, state_dict: Dict[str, Any]):
        if self.env.use_joystick:
            try:
                lxy = self.env.wc_msg.left_stick
            except:
                lxy = [0, 0]
            lin_vel_x = lxy[1]
            lin_vel_y = -lxy[0]
        else:
            lin_vel_x, lin_vel_y = 0, 0
            if "w" in self.env.key_pressed:
                lin_vel_x += 1.0
            if "s" in self.env.key_pressed:
                lin_vel_x -= 1.0
            if "a" in self.env.key_pressed:
                lin_vel_y += 1.0
            if "d" in self.env.key_pressed:
                lin_vel_y -= 1.0

        self.command_lin_vel_b[0] += self.alpha * (lin_vel_x - self.command_lin_vel_b[0])
        self.command_lin_vel_b[1] += self.alpha * (lin_vel_y - self.command_lin_vel_b[1])
        print(f"command_lin_vel_b: {self.command_lin_vel_b}")
    
    def compute(self):
        return self.command_lin_vel_b

class command_ang_vel_b(Observation):
    def __init__(self, env: Any, **kwargs: Dict[str, Any]):
        super().__init__(env, **kwargs)
        self.command_ang_vel_b = np.zeros(1, dtype=np.float32)
        self.alpha = 0.04

    def reset(self):
        self.command_ang_vel_b[:] = 0.0
    
    def update(self, state_dict: Dict[str, Any]):
        if self.env.use_joystick:
            try:
                rxy = self.env.wc_msg.right_stick
            except:
                rxy = [0, 0]
            ang_vel_z = rxy[0]
        else:
            ang_vel_z = 0
            if "q" in self.env.key_pressed:
                ang_vel_z += 1.0
            if "e" in self.env.key_pressed:
                ang_vel_z -= 1.0

        self.command_ang_vel_b[0] += self.alpha * (ang_vel_z - self.command_ang_vel_b[0])
        print(f"command_ang_vel_b: {self.command_ang_vel_b}")
    
    def compute(self):
        return self.command_ang_vel_b

