import mujoco
import time
from dataclasses import dataclass
from threading import Event, Thread
import sched
import os

import tyro
from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.sim_env.utils.mjcf import load_sim_model
from sim2real.sim_env.utils.bridge import SimulationBridge
from sim2real.sim_env.utils.elastic_band import ElasticBand
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer


class BaseSimulator:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        sim_dt: float = 0.005,
        decimation: int = 4,
        enable_elastic_band: bool = True,
    ):
        self.robot_cfg = robot_cfg
        self.sim_dt = float(sim_dt)
        self.decimation = int(decimation)
        self.enable_elastic_band = bool(enable_elastic_band)
        self._stop_event = Event()
        self._reset_event = Event()
        self._reset_button_timestamp = 0.0
        self._reset_button = None
        self._elastic_enabled = None
        self._elastic_length = None

        self.init_scene()
        # for more scenes
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = Thread(target=self.SimulationThread)

        try:
            if os.name == 'posix':
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                # set real-time scheduling policy
                SCHED_FIFO = 1
                class sched_param(ctypes.Structure):
                    _fields_ = [("sched_priority", ctypes.c_int)]
                
                param = sched_param()
                param.sched_priority = 50
                try:
                    libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
                    print("Set real-time scheduling priority")
                except Exception:
                    print("Could not set real-time priority (try running with sudo)")
        except Exception:
            pass

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass
    
    def init_scene(self):
        self.mj_model = load_sim_model(self.robot_cfg)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        # Enable the elastic band
        if self.enable_elastic_band:
            self.elastic_band = ElasticBand()
            self.band_attached_link = self._resolve_body_id(
                self.robot_cfg.elastic_band_attach_body_names
            )

        self.pelvis_body_id = self._resolve_body_id(self.robot_cfg.viewer_track_body_names)
        self.viewer = MjviserMujocoViewer(
            self.mj_model,
            self.mj_data,
            label="sim2real-base-sim",
            tracked_body_id=self.pelvis_body_id,
        )
        self._create_sim_gui()
        if self.enable_elastic_band:
            self._create_elastic_band_gui()

        self.sim_bridge = SimulationBridge(
            self.mj_model, self.mj_data, self.robot_cfg
        )

    def _resolve_body_id(self, body_names: tuple[str, ...]) -> int:
        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                return int(body_id)
        names = ", ".join(body_names)
        raise ValueError(f"Failed to resolve body from candidates: {names}")

    def _create_sim_gui(self) -> None:
        self._reset_button = self.viewer.server.gui.add_button("Reset sim")
        self._reset_button_timestamp = float(self._reset_button.update_timestamp)

    def _create_elastic_band_gui(self) -> None:
        if self.viewer is None:
            return

        with self.viewer.server.gui.add_folder("Elastic Band"):
            self._elastic_enabled = self.viewer.server.gui.add_checkbox(
                "Enabled",
                initial_value=bool(self.elastic_band.enable),
            )
            self._elastic_length = self.viewer.server.gui.add_slider(
                "Length",
                min=0.0,
                max=2.0,
                step=0.05,
                initial_value=float(self.elastic_band.length),
            )

    def _poll_viewer_gui(self) -> None:
        if self._reset_button is not None:
            timestamp = float(self._reset_button.update_timestamp)
            if timestamp != self._reset_button_timestamp:
                self._reset_button_timestamp = timestamp
                self._reset_event.set()

        if self.enable_elastic_band:
            if self._elastic_enabled is not None:
                self.elastic_band.enable = bool(self._elastic_enabled.value)
            if self._elastic_length is not None:
                self.elastic_band.length = float(self._elastic_length.value)
            if not self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :] = 0.0

    def reset_sim(self) -> None:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.xfrc_applied[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.sim_bridge.torques[:] = 0.0
        self.sim_bridge.cmd_q[:] = 0.0
        self.sim_bridge.cmd_dq[:] = 0.0
        self.sim_bridge.cmd_tau[:] = 0.0
        self.sim_bridge.cmd_kp[:] = 0.0
        self.sim_bridge.cmd_kd[:] = 0.0
        self.sim_bridge.has_received_command = False
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def sim_step(self):
        self._poll_viewer_gui()
        if self._reset_event.is_set():
            self._reset_event.clear()
            self.reset_sim()

        self.sim_bridge.publish_low_state()
        if self.enable_elastic_band:
            self.mj_data.xfrc_applied[self.band_attached_link, :] = 0.0
            if self.elastic_band.enable:
                pos = self.mj_data.xpos[self.band_attached_link]
                lin_vel = self.mj_data.cvel[self.band_attached_link, 3:6]
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = (
                    self.elastic_band.Advance(pos, lin_vel)
                )
        self.sim_bridge.compute_torques()
        self.mj_data.ctrl[:] = self.sim_bridge.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

    def SimulationThread(self):
        sim_cnt = 0
        start_time = time.time()
        
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()
        
        while self.is_running():
            scheduler.enterabs(next_run_time, 1, self._sim_step_scheduled, ())
            scheduler.run()
            
            next_run_time += self.sim_dt
            sim_cnt += 1

            if self.viewer is not None and sim_cnt % self.decimation == 0:
                self.sync_viewer()
        
            # Get FPS
            if sim_cnt % 100 == 0:
                current_time = time.time()
                print(f"FPS: {100 / (current_time - start_time)}")
                start_time = current_time

    def _sim_step_scheduled(self):
        loop_start = time.perf_counter()
        self.sim_step()
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.sim_dt:
            print(f"Sim step took {elapsed:.6f} seconds, expected {self.sim_dt}")

    def is_running(self) -> bool:
        if self._stop_event.is_set():
            return False
        if self.viewer is None:
            return True
        return bool(self.viewer.is_running())

    def sync_viewer(self) -> None:
        if self.viewer is None or not self.viewer.has_clients():
            return
        self.viewer.sync()

    def stop(self) -> None:
        self._stop_event.set()
        if self.viewer is not None:
            self.viewer.close()


@dataclass
class Args:
    """Robot."""

    robot: str = "g1"
    sim_dt: float = 0.005
    decimation: int = 4
    enable_elastic_band: bool = True


if __name__ == "__main__":
    args = tyro.cli(Args)

    simulation = BaseSimulator(
        get_robot_cfg(args.robot),
        sim_dt=args.sim_dt,
        decimation=args.decimation,
        enable_elastic_band=args.enable_elastic_band,
    )
    simulation.sim_thread.start()
