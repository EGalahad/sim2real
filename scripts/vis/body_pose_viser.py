"""
This script subscribes to ZMQ pose data published by the Vicon script
and visualizes the object frames in real-time using Viser.
"""

import sys
import time
import numpy as np
import viser
import threading
from typing import List, Dict

# Add the project root to the Python path to find the 'utils' module
sys.path.append(".")
from utils.common import ZMQSubscriber, PORTS

JOINT_STATE_PUBLISHER_IP = "172.26.52.156"
BODY_POSE_PUBLISHER_IP = "172.26.52.156"

class BodyPoseViser:
    def __init__(self, subscribe_names: List[str]):
        self.subscribe_names = subscribe_names
        
        # --- Viser Setup ---
        self.server = viser.ViserServer()
        # Add a grid to the scene for better orientation
        # CHANGED: Called .scene.add_grid() instead of .add_grid()
        self.server.scene.add_grid(
            "/world",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
        )
        
        # --- Data and Threading Setup ---
        # A dictionary to hold the latest pose for each object
        self.latest_poses: Dict[str, tuple] = {}
        # A lock to ensure thread-safe access to the latest_poses dictionary
        self.pose_lock = threading.Lock()

        # --- ZMQ and Viser Frame Initialization ---
        self.subscribers: Dict[str, ZMQSubscriber] = {}
        self.viser_frames: Dict[str, viser.SceneNodeHandle] = {} # Type hint changed for clarity

        for name in self.subscribe_names:
            # Initialize pose with a default value (origin)
            self.latest_poses[name] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) # (pos, wxyz_quat)

            # Create a ZMQ subscriber for each object
            port = PORTS[f"{name}_pose"]
            self.subscribers[name] = ZMQSubscriber(port, ip=BODY_POSE_PUBLISHER_IP)

            # Create a Viser frame for each object to visualize its pose
            # CHANGED: Called .scene.add_frame() and used 'axes_radius'
            self.viser_frames[name] = self.server.scene.add_frame(
                f"/{name}",
                axes_length=0.2,
                axes_radius=0.01, # Renamed from axes_width
                # show_name=True
            )

    def _receive_data_loop(self, object_name: str):
        """A dedicated thread loop to receive data for a single object."""
        subscriber = self.subscribers[object_name]
        print(f"Starting receiver thread for '{object_name}'...")
        while True:
            # Blocking receive call
            pose = subscriber.receive_pose()
            if pose is not None:
                position, quaternion = pose.position, pose.quaternion
                # Use a lock to safely update the shared pose dictionary
                with self.pose_lock:
                    self.latest_poses[object_name] = (position, quaternion)

    def start_receivers(self):
        """Starts a separate thread for each ZMQ subscriber."""
        for name in self.subscribe_names:
            thread = threading.Thread(
                target=self._receive_data_loop,
                args=(name,),
                daemon=True  # Daemon threads exit when the main program exits
            )
            thread.start()
        print("\nAll receiver threads started.")

    def run_visualization(self):
        """The main loop to update the Viser scene."""
        print("Starting visualization loop. Check your browser at the URL below.")
        
        while True:
            # Safely read the latest poses
            with self.pose_lock:
                for name, (position, quaternion) in self.latest_poses.items():
                    frame_handle = self.viser_frames[name]
                    # Update the position and orientation of the Viser frame
                    frame_handle.position = position
                    frame_handle.wxyz = quaternion # Viser uses [w, x, y, z] format

            # Sleep for a short duration to control the update rate (e.g., 60 FPS)
            time.sleep(1 / 60.0)

if __name__ == "__main__":
    # IMPORTANT: This list must match the 'publish_names' in your publisher script
    # For example, if you run `publish_vicon.py` with ["suitcase", "pelvis"], use the same here.
    names_to_visualize = ["suitcase", "pelvis"]
    # names_to_visualize = ["box", "pelvis"]
    names_to_visualize = ["stool", "pelvis"]

    # Create and run the visualizer
    visualizer = BodyPoseViser(subscribe_names=names_to_visualize)
    visualizer.start_receivers()
    visualizer.run_visualization()