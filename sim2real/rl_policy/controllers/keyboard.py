from __future__ import annotations

import subprocess
import threading

from loguru import logger

from sim2real.rl_policy.control_mode import resolve_keyboard_control_mode
from sim2real.rl_policy.controllers.base import ControllerBase


class KeyboardController(ControllerBase):
    name = "keyboard"

    def __init__(self) -> None:
        self.key_pressed: set[str] = set()
        self._pending_control_mode = None
        self._pending_extra_keys: list[str] = []
        self._lock = threading.Lock()
        self._listener_thread = threading.Thread(
            target=self._start_key_listener,
            daemon=True,
        )
        self._listener_thread.start()

    def _start_key_listener(self) -> None:
        def on_press(keycode):
            try:
                if keycode in self.key_pressed:
                    return
                self.key_pressed.add(keycode)
                with self._lock:
                    mode = resolve_keyboard_control_mode(keycode)
                    if mode is not None:
                        self._pending_control_mode = mode
                    else:
                        self._pending_extra_keys.append(keycode)
            except AttributeError as exc:
                logger.warning(f"Keyboard key {keycode}. Error: {exc}")

        def on_release(keycode):
            try:
                self.key_pressed.discard(keycode)
            except AttributeError as exc:
                logger.warning(f"Keyboard key {keycode}. Error: {exc}")

        from sshkeyboard import listen_keyboard

        try:
            listen_keyboard(on_press=on_press, on_release=on_release)
        except Exception as exc:
            logger.warning(f"Keyboard listener stopped unexpectedly: {exc}")

    def get_control_mode(self):
        with self._lock:
            mode = self._pending_control_mode
            self._pending_control_mode = None
        return mode

    def get_extra_keys(self) -> list[str]:
        with self._lock:
            extra_keys = list(self._pending_extra_keys)
            self._pending_extra_keys.clear()
        return extra_keys

    def close(self) -> None:
        try:
            from sshkeyboard import stop_listening

            stop_listening()
        except Exception as exc:
            logger.debug(f"Failed to stop keyboard listener cleanly: {exc}")
        finally:
            try:
                subprocess.run(["stty", "sane"], check=False)
            except Exception as exc:
                logger.debug(f"Failed to run stty sane: {exc}")
