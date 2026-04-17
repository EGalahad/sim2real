from __future__ import annotations

from copy import deepcopy

import zmq
from loguru import logger

from sim2real.rl_policy.control_mode import PicoButtonState, resolve_pico_control_mode
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.utils.common import PORTS, PicoControllerStateMessage


class PicoController(ControllerBase):
    name = "pico"

    def __init__(
        self,
        connect: str = f"tcp://127.0.0.1:{PORTS['pico_controller']}",
        hwm: int = 1,
    ) -> None:
        self._pico_msg = PicoButtonState()
        self._last_pico_msg = PicoButtonState()
        self._available = True
        self._connect = connect

        self._zmq_context = zmq.Context.instance()
        self._socket = self._zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, int(hwm))
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            self._socket.connect(connect)
        except Exception as exc:
            self._available = False
            self._socket.close(0)
            logger.warning(f"PICO controller ZMQ subscriber unavailable, continuing without it: {exc}")
        else:
            logger.info("PICO controller ZMQ subscriber connected to {}", connect)

    def _receive_pico_controller(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            try:
                decoded = PicoControllerStateMessage.from_bytes(raw)
            except Exception as exc:
                logger.debug(f"PICO controller ZMQ decode error: {exc}")
                continue

            self._pico_msg = PicoButtonState(
                A=decoded.A,
                B=decoded.B,
            )

    def get_control_mode(self):
        if not self._available:
            return None

        try:
            self._receive_pico_controller()
        except Exception as exc:
            logger.debug(f"PICO controller ZMQ receive error from {self._connect}: {exc}")
            return None

        pico_local = deepcopy(self._pico_msg)

        mode = resolve_pico_control_mode(pico_local, self._last_pico_msg)
        self._last_pico_msg = pico_local
        return mode

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug(f"Failed to stop PICO listener cleanly: {exc}")
