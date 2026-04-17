from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import threading
import time

from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.control_mode import resolve_unitree_joystick_control_mode
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.inference import Timer


@dataclass(frozen=True)
class UnitreeJoystickState:
    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False
    L1: bool = False
    L2: bool = False
    R1: bool = False
    R2: bool = False
    left_stick: tuple[float, float] = (0.0, 0.0)
    right_stick: tuple[float, float] = (0.0, 0.0)


class UnitreeJoystickController(ControllerBase):
    name = "unitree_joystick"

    def __init__(self, robot_cfg: RobotCfg, perf_dict: dict[str, float]) -> None:
        self.robot_cfg = robot_cfg
        self.perf_dict = perf_dict
        self._wc_lock = threading.Lock()
        self._joystick_thread_stop = threading.Event()
        self._pending_extra_keys: list[str] = []
        self._wc_msg: UnitreeJoystickState | None = None
        self._last_wc_msg = UnitreeJoystickState()

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

        if self.robot_cfg.interface:
            ChannelFactoryInitialize(self.robot_cfg.domain_id, self.robot_cfg.interface)
        else:
            ChannelFactoryInitialize(self.robot_cfg.domain_id)

        self.wireless_controller_sub = ChannelSubscriber(
            "rt/wirelesscontroller",
            WirelessController_,
        )
        self.wireless_controller_sub.Init(None, 0)

        self._joystick_thread = threading.Thread(
            target=self._poll_wireless_controller,
            daemon=True,
        )
        self._joystick_thread.start()

    @property
    def state(self) -> UnitreeJoystickState | None:
        with self._wc_lock:
            if self._wc_msg is None:
                return None
            return deepcopy(self._wc_msg)

    def _decode_wireless_controller(self, msg) -> UnitreeJoystickState:
        key_bits = {
            "R1": 0,
            "L1": 1,
            "R2": 4,
            "L2": 5,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
        }
        keys = getattr(msg, "keys", 0)
        return UnitreeJoystickState(
            A=bool(keys & (1 << key_bits["A"])),
            B=bool(keys & (1 << key_bits["B"])),
            X=bool(keys & (1 << key_bits["X"])),
            Y=bool(keys & (1 << key_bits["Y"])),
            L1=bool(keys & (1 << key_bits["L1"])),
            L2=bool(keys & (1 << key_bits["L2"])),
            R1=bool(keys & (1 << key_bits["R1"])),
            R2=bool(keys & (1 << key_bits["R2"])),
            left_stick=(getattr(msg, "lx", 0.0), getattr(msg, "ly", 0.0)),
            right_stick=(getattr(msg, "rx", 0.0), getattr(msg, "ry", 0.0)),
        )

    def _poll_wireless_controller(self) -> None:
        poll_interval = 0.2
        while not self._joystick_thread_stop.is_set():
            try:
                with Timer(self.perf_dict, "read_wireless_controller"):
                    raw_msg = self.wireless_controller_sub.Read()
                if raw_msg is not None:
                    with Timer(self.perf_dict, "decode_wireless_controller"):
                        decoded = self._decode_wireless_controller(raw_msg)
                    with self._wc_lock:
                        self._wc_msg = decoded
            except Exception as exc:
                logger.debug(f"Joystick poll error: {exc}")
            finally:
                time.sleep(poll_interval)

    def get_control_mode(self):
        wc_local = self.state
        if wc_local is None:
            return None

        mode = None
        if wc_local.A and not self._last_wc_msg.A:
            mode = resolve_unitree_joystick_control_mode("A")
        if wc_local.B and not self._last_wc_msg.B:
            self._pending_extra_keys.append("B")
        if wc_local.X and not self._last_wc_msg.X:
            self._pending_extra_keys.append("X")
        if wc_local.Y and not self._last_wc_msg.Y:
            self._pending_extra_keys.append("Y")
        if wc_local.L1 and not self._last_wc_msg.L1:
            self._pending_extra_keys.append("L1")
        if wc_local.L2 and not self._last_wc_msg.L2:
            self._pending_extra_keys.append("L2")
        if wc_local.R1 and not self._last_wc_msg.R1:
            mode = resolve_unitree_joystick_control_mode("R1")
        if wc_local.R2 and not self._last_wc_msg.R2:
            mode = resolve_unitree_joystick_control_mode("R2")

        self._last_wc_msg = wc_local
        return mode

    def get_extra_keys(self) -> list[str]:
        extra_keys = list(self._pending_extra_keys)
        self._pending_extra_keys.clear()
        return extra_keys

    def close(self) -> None:
        try:
            self._joystick_thread_stop.set()
            if self._joystick_thread.is_alive():
                self._joystick_thread.join(timeout=1.0)
        except Exception as exc:
            logger.debug(f"Failed to stop joystick listener cleanly: {exc}")
