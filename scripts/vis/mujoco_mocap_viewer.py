import numpy as np
import time
import threading

import mujoco
import mujoco.viewer

import sys
sys.path.append(".")
from utils.common import ZMQSubscriber, PORTS
from typing import List

scene = "data/robots/g1/g1_29dof_rubberhand-stool.xml"
scene = "data/robots/g1/g1_29dof_rubberhand-stool-low.xml"
scene = "data/robots/g1/g1_29dof_rubberhand-foam.xml"
scene = "data/robots/g1/g1_29dof_rubberhand-door.xml"
scene = "data/robots/g1/g1_29dof_rubberhand-stair.xml"
# scene = "data/robots/g1/g1_29dof_rubberhand-suitcase.xml"
# scene = "data/robots/g1/g1_29dof_rubberhand-bread_box.xml"

# for motion data publisher
# JOINT_STATE_PUBLISHER_IP = "localhost"
# BODY_POSE_PUBLISHER_IP = "localhost"

# for deployment
IP = "172.26.27.152"
JOINT_STATE_PUBLISHER_IP = IP
BODY_POSE_PUBLISHER_IP = IP

# BODY_POSE_PUBLISHER_IP = "localhost"

class MuJoCoMocapViewer:
    def __init__(
        self,
        frequency: int = 50,
        mujoco_model_path: str = scene
    ):
        print("Initializing MuJoCo Mocap Viewer...")
        
        self.freq = frequency
        
        # Initialize MuJoCo model and viewer
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

        # Get all joint names from MuJoCo model (excluding free joints)
        mujoco_joint_names = [self.model.joint(i).name for i in range(self.model.njnt) if self.model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
        
        # Wait for publisher to send joint names and create mapping
        print("Waiting for publisher joint names...")
        joint_names_subscriber = ZMQSubscriber(PORTS['joint_names'], ip=JOINT_STATE_PUBLISHER_IP)
        while True:
            publisher_joint_names = joint_names_subscriber.receive_names()
            if publisher_joint_names is not None:
                break
        joint_names_subscriber.close()
        
        print(f"Received publisher joint names: {publisher_joint_names}")
        print(f"MuJoCo joint names: {mujoco_joint_names}")
        
        # Create mapping from publisher joints to MuJoCo joints
        shared_joint_names = list(sorted(set(mujoco_joint_names) & set(publisher_joint_names)))
        publisher_joint_indices = [publisher_joint_names.index(name) for name in shared_joint_names]
        mujoco_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in shared_joint_names]
        mujoco_qpos_adrs = [self.model.jnt_qposadr[joint_id] for joint_id in mujoco_joint_ids]
        self.publisher_joint_indices = np.array(publisher_joint_indices)
        self.mujoco_joint_qpos_adrs = np.array(mujoco_qpos_adrs)
        
        # Handle root joints (free joints)
        root_joint_ids = [i for i in range(self.model.njnt) if self.model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
        self.root_joint_names = [self.model.joint(i).name.replace('_root', '') for i in root_joint_ids]
        self.root_joint_qpos_adrs = self.model.jnt_qposadr[root_joint_ids]
        self.root_joint_subscribers: List[ZMQSubscriber] = []

        # Initialize ZMQ subscribers
        self.joint_subscriber = ZMQSubscriber(PORTS['joint_pos'], ip=JOINT_STATE_PUBLISHER_IP)
        for root_joint_name in self.root_joint_names:
            subscriber = ZMQSubscriber(PORTS[f"{root_joint_name}_pose"], ip=BODY_POSE_PUBLISHER_IP)
            self.root_joint_subscribers.append(subscriber)

        self.running = True
        self.comm_thread = threading.Thread(target=self.zmq_communication_loop)
        self.comm_thread.daemon = True
        self.comm_thread.start()

        print("MuJoCo Mocap Viewer initialized, waiting for data...")

    def zmq_communication_loop(self):
        """Handle ZMQ communication in a separate thread"""
        while self.running:
            joint_msg = self.joint_subscriber.receive_joint_state()
            if joint_msg and len(self.mujoco_joint_qpos_adrs):
                # Map from publisher joint order to MuJoCo joint order
                self.data.qpos[self.mujoco_joint_qpos_adrs] = joint_msg.positions[self.publisher_joint_indices]
            
            for qpos_adr, subscriber in zip(self.root_joint_qpos_adrs, self.root_joint_subscribers):
                pose_msg = subscriber.receive_pose()
                if pose_msg:
                    pose = np.concatenate([pose_msg.position, pose_msg.quaternion])
                    self.data.qpos[qpos_adr:qpos_adr + 7] = pose
            

    def mujoco_update(self):
        """Update MuJoCo simulation at 50 Hz"""
        while self.running:
            if self.viewer is None:
                time.sleep(0.5)
                print("Waiting for MuJoCo model to be loaded...")
                continue
            
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(1.0 / self.freq)

    def run(self):
        """Main loop"""
        try:
            self.mujoco_update()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'comm_thread'):
            self.comm_thread.join()
        
        self.joint_subscriber.close()
        for subscriber in self.root_joint_subscribers:
            subscriber.close()
        
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    viewer = MuJoCoMocapViewer()
    viewer.run()