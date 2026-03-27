import mujoco
import mujoco.viewer
import time
from threading import Thread
import sched
import os

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.sim_env.utils.bridge import SimulationBridge
from sim2real.sim_env.utils.elastic_band import ElasticBand
from sim2real.teleop.mujoco_viewer_utils import temp_mjcf_with_floor


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class BaseSimulator:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        sim_dt: float = 0.005,
        enable_elastic_band: bool = True,
    ):
        self.robot_cfg = robot_cfg
        self.sim_dt = float(sim_dt)
        self.enable_elastic_band = bool(enable_elastic_band)

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
        robot_scene = self.robot_cfg.sim_mjcf_path
        if robot_scene is None:
            raise ValueError(f"Robot '{self.robot_cfg.name}' does not define sim_mjcf_path")
        with temp_mjcf_with_floor(robot_scene) as viewer_mjcf_path:
            self.mj_model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        # Enable the elastic band
        if self.enable_elastic_band:
            self.elastic_band = ElasticBand()
            self.band_attached_link = self._resolve_body_id(
                self.robot_cfg.elastic_band_attach_body_names
            )
            key_callback = self.elastic_band.MujocoKeyCallback
        else:
            key_callback = None

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.pelvis_body_id = self._resolve_body_id(self.robot_cfg.viewer_track_body_names)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.pelvis_body_id

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

    def sim_step(self):
        self.sim_bridge.publish_low_state()
        if self.enable_elastic_band:
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
        
        while self.viewer.is_running():
            scheduler.enterabs(next_run_time, 1, self._sim_step_scheduled, ())
            scheduler.run()
            
            next_run_time += self.sim_dt
            sim_cnt += 1

            self.viewer.sync()
        
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot", type=str, default="g1", help="robot name"
    )
    parser.add_argument(
        "--sim_dt", type=float, default=0.005, help="simulation timestep in seconds"
    )
    parser.add_argument(
        "--enable_elastic_band",
        type=_parse_bool_arg,
        default=True,
        help="enable the elastic band in simulation (true/false)",
    )
    args = parser.parse_args()

    simulation = BaseSimulator(
        get_robot_cfg(args.robot),
        sim_dt=args.sim_dt,
        enable_elastic_band=args.enable_elastic_band,
    )
    simulation.sim_thread.start()
