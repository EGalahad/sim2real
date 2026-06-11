from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import mujoco
import numpy as np


class _ShutdownTolerantExecutor:
    """Drop late viser callbacks after its executor has already shut down."""

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def submit(self, *args: Any, **kwargs: Any) -> Future:
        try:
            return self._executor.submit(*args, **kwargs)
        except RuntimeError as exc:
            if "shutdown" not in str(exc):
                raise
            future: Future = Future()
            future.set_result(None)
            return future

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


class MjviserMujocoViewer:
    """Small adapter exposing the viewer methods used by sim2real runtimes."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        label: str,
        port: int | None = None,
        tracked_body_id: int | None = None,
        camera_distance: float = 3.0,
        camera_azimuth: float = 120.0,
        camera_elevation: float = 20.0,
        create_gui: bool = True,
    ) -> None:
        try:
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise ImportError(
                "Interactive sim2real visualization now uses mjviser. "
                "Install dependencies with `uv sync` or run through `uv run`."
            ) from exc

        if port is None:
            self.server = viser.ViserServer(label=label)
        else:
            self.server = viser.ViserServer(port=int(port), label=label)
        self._install_shutdown_tolerant_executor()
        self.scene = ViserMujocoScene(self.server, model, num_envs=1)
        self.model = model
        self.data = data
        self._running = True

        if tracked_body_id is not None:
            self.scene._tracked_body_id = int(tracked_body_id)
            self.scene.camera_tracking_enabled = True

        if create_gui:
            self.scene.create_visualization_gui(
                camera_distance=float(camera_distance),
                camera_azimuth=float(camera_azimuth),
                camera_elevation=float(camera_elevation),
            )

        self.sync()
        print(f"[{label}] mjviser server: http://localhost:{self.server.get_port()}")

    def _install_shutdown_tolerant_executor(self) -> None:
        executor = _ShutdownTolerantExecutor(self.server._thread_executor)
        self.server._thread_executor = executor
        self.server.scene._thread_executor = executor
        self.server.gui._thread_executor = executor

    def is_running(self) -> bool:
        return self._running

    def has_clients(self) -> bool:
        return bool(self.server.get_clients())

    def sync(self) -> None:
        body_xpos = self.data.xpos[None, ...]
        body_xmat = self.data.xmat.reshape(1, -1, 3, 3)
        if self.model.nmocap > 0:
            mocap_pos = self.data.mocap_pos[None, ...]
            mocap_quat = self.data.mocap_quat[None, ...]
        else:
            mocap_pos = np.zeros((1, 0, 3), dtype=np.float64)
            mocap_quat = np.zeros((1, 0, 4), dtype=np.float64)

        ctrl = self.data.ctrl[None, ...] if self.model.nu > 0 else None
        self.scene.update_from_arrays(
            body_xpos=body_xpos,
            body_xmat=body_xmat,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qpos=self.data.qpos[None, ...],
            qvel=self.data.qvel[None, ...],
            ctrl=ctrl,
        )

    def close(self) -> None:
        self._running = False
        self.server.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scene, name)
