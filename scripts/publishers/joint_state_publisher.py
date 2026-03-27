"""This script listens to the low state of the robot and publishes the joint positions via ZMQ"""

import numpy as np
import time
import threading
import argparse
import sched

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.utils.common import ZMQPublisher, PORTS

class JointStatePublisher:
    """
    Receives joint state from Unitree SDK and publishes via ZMQ as numpy array
    """
    def __init__(self, robot_cfg: RobotCfg, dest_joint_names, publish_freq=50):
        self.robot_cfg = robot_cfg
        # initialize robot related processes
        if self.robot_cfg.interface:
            ChannelFactoryInitialize(self.robot_cfg.domain_id, self.robot_cfg.interface)
        else:
            ChannelFactoryInitialize(self.robot_cfg.domain_id)

        # Initialize channel subscriber
        robot_type = self.robot_cfg.robot_type
        if robot_type == "h1" or robot_type == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 1)
        elif robot_type == "g1_29dof" or robot_type == "h1-2_27dof" or robot_type == "h1-2_21dof" or robot_type == "g1_real":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [self.robot_cfg.joint_names.index(name) for name in dest_joint_names]
        
        # Initialize joint state arrays
        self.joint_pos = np.zeros(self.num_dof)
        self.joint_vel = np.zeros(self.num_dof)
        
        # Initialize robot state
        self.robot_low_state = None
        
        # Initialize ZMQ publisher using common.py
        zmq_port = PORTS['joint_pos']
        self.publisher = ZMQPublisher(zmq_port)
        print(f"ZMQ publisher bound to port {zmq_port}")
        
        self.joint_names = list(self.robot_cfg.joint_names)
        self.joint_names_publisher = ZMQPublisher(PORTS['joint_names'])

        # Publishing frequency
        self.publish_freq = publish_freq
        self.publish_interval = 1.0 / publish_freq
        
        # Start publishing thread
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()
        
    def _publish_loop(self):
        """Publishing loop that runs in a separate thread with precise timing"""
        publish_cnt = 0
        start_time = time.time()
        
        # use scheduler for precise timing
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()
        
        while True:
            try:
                scheduler.enterabs(next_run_time, 1, self._publish_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.publish_interval
                publish_cnt += 1
                
                # Print FPS every 100 iterations
                if publish_cnt % 100 == 0:
                    current_time = time.time()
                    actual_freq = 100 / (current_time - start_time)
                    print(f"Publishing frequency: {actual_freq:.1f} Hz (target: {self.publish_freq} Hz)")
                    start_time = current_time
                    
            except KeyboardInterrupt:
                print("Publishing loop interrupted")
                break
            except Exception as e:
                print(f"Error in publishing loop: {e}")
                time.sleep(0.01)
    
    def _publish_step_scheduled(self):
        """Execute one publishing step with timing measurement"""
        if not self.robot_low_state:
            return
            
        loop_start = time.perf_counter()

        self.joint_names_publisher.publish_names(self.joint_names)

        # Extract joint data from robot state
        source_joint_state = self.robot_low_state.motor_state
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = source_joint_state[src_idx].q
            self.joint_vel[dst_idx] = source_joint_state[src_idx].dq
        
        self.publisher.publish_joint_state(self.joint_pos, self.joint_vel)
        
        # Measure execution time
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.publish_interval:
            print(f"Publish step took {elapsed:.6f} seconds, expected {self.publish_interval:.6f}")

    def LowStateHandler_go(self, msg: LowState_go):
        self.robot_low_state = msg
    
    def LowStateHandler_hg(self, msg: LowState_hg):
        self.robot_low_state = msg

def main():
    parser = argparse.ArgumentParser(description="Joint State ZMQ Publisher")
    parser.add_argument("--robot", type=str, default="g1", help="Robot name")
    parser.add_argument("--freq", type=int, default=50, help="Publishing frequency")
    
    args = parser.parse_args()
    robot_cfg = get_robot_cfg(args.robot)
    dest_joint_names = list(robot_cfg.joint_names)
    
    publisher = JointStatePublisher(
        robot_cfg=robot_cfg,
        dest_joint_names=dest_joint_names,
        publish_freq=args.freq
    )

    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        publisher.publisher.close()

if __name__ == "__main__":
    main() 
