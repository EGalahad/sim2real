"""Simple script to load and replay recorded qpos data from npz files

This script loads recorded qpos data, performs smoothing, calculates qvel,
and provides a MuJoCo viewer for replaying the motion with basic controls.

Features:
- Loads qpos data from npz files
- Smooths qpos data with special handling for quaternions  
- Calculates qvel from smoothed qpos
- Automatic replay with console controls
"""

import numpy as np
import time
import argparse
import os
import json
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer

import sys
sys.path.append(".")

# Default scene path
scene = "./data/robots/g1/g1_29dof_rev_1_0.xml"


class SimpleReplay:
    def __init__(self, npz_file_path, mujoco_model_path=scene):
        """Initialize the replay system
        
        Args:
            npz_file_path: Path to the recorded npz file
            mujoco_model_path: Path to MuJoCo model file
            output_path: Optional path to save motion data
        """
        self.npz_file_path = npz_file_path
        
        # Load recorded data
        print(f"Loading recorded data from: {npz_file_path}")
        self.recorded_data = np.load(npz_file_path)
        self.qpos_data = self.recorded_data['qpos']
        self.qvel_data = self.recorded_data.get('qvel')
        self.timestamps = self.recorded_data['timestamps']
        self.frequency = float(self.recorded_data['frequency'])
        self.nq = int(self.recorded_data['nq'])

        self.qpos_data = self.qpos_data[10:]
        self.qvel_data = self.qvel_data[10:]
        self.timestamps = self.timestamps[10:]
        self.qpos_data[:, 2] += 0.04

        print(f"Loaded {len(self.qpos_data)} frames")
        print(f"Recording frequency: {self.frequency} Hz")
        print(f"Model DOF: {self.nq}")
        print(f"Duration: {(self.timestamps[-1] - self.timestamps[0]):.2f} seconds")
        
        # Initialize MuJoCo
        print(f"Loading MuJoCo model: {mujoco_model_path}")
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Verify model compatibility
        if self.model.nq != self.nq:
            print(f"Warning: Model nq ({self.model.nq}) != recorded nq ({self.nq})")
        
    def _key_callback(self, keycode):
        """Handle keyboard input from MuJoCo viewer"""
        key = chr(keycode) if keycode < 256 else None
        
        if key == ' ':  # Spacebar - toggle pause
            self.toggle_pause()
        elif key == 'r' or key == 'R':  # R - reset
            self.reset_replay()
        elif key == 'q' or key == 'Q':  # Q - quit
            self.running = False
        elif key == '+' or key == '=':  # + - increase speed
            self.set_playback_speed(self.playback_speed + 0.1)
        elif key == '-':  # - - decrease speed
            self.set_playback_speed(self.playback_speed - 0.1)
        elif keycode == 262:  # Right arrow key
            if self.is_paused:
                self.step_frame(50)
        elif keycode == 263:  # Left arrow key
            if self.is_paused:
                self.step_frame(-50)
    
    def step_frame(self, direction):
        """Step frame forward or backward when paused"""
        self.current_frame += direction
        self.current_frame %= len(self.qpos_data)  # Wrap around
        
        # Update simulation with new frame
        self.update_simulation()
        print(f"Frame: {self.current_frame}/{len(self.qpos_data)}")
    
    def _input_handler(self):
        """Handle console input in separate thread"""
        while self.running:
            try:
                cmd = input().strip().lower()
                if cmd == 'p':
                    self.toggle_pause()
                elif cmd == 'r':
                    self.reset_replay()
                elif cmd == 's':
                    try:
                        speed = float(input("Enter new speed (0.1-5.0): "))
                        self.set_playback_speed(speed)
                    except ValueError:
                        print("Invalid speed value")
                elif cmd == 'q':
                    self.running = False
                    break
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        print(f"Replay {'paused' if self.is_paused else 'resumed'}")
    
    def reset_replay(self):
        """Reset replay to beginning"""
        self.current_frame = 0
        print("Replay reset to beginning")
    
    def set_playback_speed(self, speed):
        """Set playback speed multiplier"""
        self.playback_speed = max(0.1, min(5.0, speed))  # Clamp between 0.1x and 5.0x
        print(f"Playback speed: {self.playback_speed:.1f}x")
    
    def update_simulation(self):
        """Update MuJoCo simulation with current frame data"""
        # Set qpos and qvel
        frame_qpos = self.qpos_data[self.current_frame]
        frame_qvel = self.qvel_data[self.current_frame]
        
        # Copy data to MuJoCo
        self.data.qpos[:] = frame_qpos[:self.model.nq]
        self.data.qvel[:] = frame_qvel[:self.model.nv]
        
        # Forward simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Sync viewer
        self.viewer.sync()
        
        return True
    
    def export_motion_data(self, output_path):
        """Export motion data in the format compatible, no key callback"""
        os.makedirs(output_path, exist_ok=True)

        # Get body and joint names from model
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        joint_qpos_adr = [self.model.jnt_qposadr[i] for i in range(self.model.njnt)]
        joint_qvel_adr = [self.model.jnt_dofadr[i] for i in range(self.model.njnt)]

        body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        def get_sensor_ids(sensor_suffix: str):
            sensor_ids = [
                mujoco.mj_name2id(
                    self.model, 
                    mujoco.mjtObj.mjOBJ_SENSOR, 
                    f"{body_name}_{sensor_suffix}") 
                for body_name in body_names
            ]
            return sensor_ids
        
        body_pos_sensor_ids = get_sensor_ids("pos")
        body_quat_sensor_ids = get_sensor_ids("quat")
        body_lin_vel_sensor_ids = get_sensor_ids("linvel")
        body_ang_vel_sensor_ids = get_sensor_ids("angvel")

        valid_body_ids = [i for i, body_pos_sensor_id in enumerate(body_pos_sensor_ids) if body_pos_sensor_id != -1]
        body_names = [body_names[i] for i in valid_body_ids]
        
        body_pos_sensor_ids = [body_pos_sensor_ids[i] for i in valid_body_ids]
        body_quat_sensor_ids = [body_quat_sensor_ids[i] for i in valid_body_ids]
        body_lin_vel_sensor_ids = [body_lin_vel_sensor_ids[i] for i in valid_body_ids]
        body_ang_vel_sensor_ids = [body_ang_vel_sensor_ids[i] for i in valid_body_ids]

        if any(sensor_id == -1 for sensor_id in body_pos_sensor_ids):
            raise ValueError("Sensor pos not found for some body.")
        if any(sensor_id == -1 for sensor_id in body_quat_sensor_ids):
            raise ValueError("Sensor quat not found for some body.")
        if any(sensor_id == -1 for sensor_id in body_lin_vel_sensor_ids):
            raise ValueError("Sensor linvel not found for some body.")
        if any(sensor_id == -1 for sensor_id in body_ang_vel_sensor_ids):
            raise ValueError("Sensor angvel not found for some body.")

        body_pos_sensor_adrs = self.model.sensor_adr[body_pos_sensor_ids]
        body_quat_sensor_adrs = self.model.sensor_adr[body_quat_sensor_ids]
        body_lin_vel_sensor_adrs = self.model.sensor_adr[body_lin_vel_sensor_ids]
        body_ang_vel_sensor_adrs = self.model.sensor_adr[body_ang_vel_sensor_ids]

        # Initialize arrays for motion data
        n_frames = len(self.qpos_data)
        n_bodies = len(body_names)
        n_joints = len(joint_names)
        
        body_pos_w = np.zeros((n_frames, n_bodies, 3))
        body_quat_w = np.zeros((n_frames, n_bodies, 4))
        body_lin_vel_w = np.zeros((n_frames, n_bodies, 3))
        body_ang_vel_w = np.zeros((n_frames, n_bodies, 3))
        joint_pos = np.zeros((n_frames, n_joints))
        joint_vel = np.zeros((n_frames, n_joints))
        
        print("Computing motion data for all frames...")
        
        # Process each frame
        for frame_idx in range(n_frames):
            # Set qpos and qvel for this frame
            self.data.qpos[:] = self.qpos_data[frame_idx][:self.model.nq]
            self.data.qvel[:] = self.qvel_data[frame_idx][:self.model.nv]
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            joint_pos[frame_idx, :] = self.data.qpos[joint_qpos_adr]
            joint_vel[frame_idx, :] = self.data.qvel[joint_qvel_adr]
            for body_idx, body_name in enumerate(body_names):
                body_pos_w[frame_idx, body_idx] = self.data.sensordata[body_pos_sensor_adrs[body_idx]:body_pos_sensor_adrs[body_idx] + 3]
                body_quat_w[frame_idx, body_idx] = self.data.sensordata[body_quat_sensor_adrs[body_idx]:body_quat_sensor_adrs[body_idx] + 4]
                body_lin_vel_w[frame_idx, body_idx] = self.data.sensordata[body_lin_vel_sensor_adrs[body_idx]:body_lin_vel_sensor_adrs[body_idx] + 3]
                body_ang_vel_w[frame_idx, body_idx] = self.data.sensordata[body_ang_vel_sensor_adrs[body_idx]:body_ang_vel_sensor_adrs[body_idx] + 3]
            
            if frame_idx % max(1, n_frames // 10) == 0:
                progress = (frame_idx / n_frames) * 100
                print(f"  Progress: {progress:.1f}%")
        
        # Save motion data
        motion_data = {
            'body_pos_w': body_pos_w,
            'body_quat_w': body_quat_w,
            'body_lin_vel_w': body_lin_vel_w,
            'body_ang_vel_w': body_ang_vel_w,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
        }
        
        motion_file = os.path.join(output_path, 'motion.npz')
        np.savez_compressed(motion_file, **motion_data)
        print(f"Saved motion data to: {motion_file}")
        
        # Save metadata
        meta_data = {
            'body_names': body_names,
            'joint_names': joint_names,
            'fps': float(self.frequency),
        }

        meta_file = os.path.join(output_path, 'meta.json')
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)
        print(f"Saved metadata to: {meta_file}")
        
        print("Motion data export completed!")
    
    def replay_loop(self):
        # Launch viewer with key callback
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, 
            show_left_ui=False, show_right_ui=False,
            key_callback=self._key_callback
        )
        
        # Set camera to track pelvis
        model_body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        if "pelvis" in model_body_names:
            pelvis_body_id = model_body_names.index("pelvis")
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = pelvis_body_id
        
        # Replay control variables
        self.current_frame = 0
        self.is_paused = False
        self.playback_speed = 1.0
        self.running = True
        
        print("\nReplay Controls:")
        print("  Spacebar: Pause/Resume")
        print("  Left/Right Arrow: Previous/Next frame (when paused)")
        print("  R: Reset to beginning")
        print("  +/-: Increase/Decrease playback speed")
        print("  Q: Quit")
    
        """Main replay loop"""
        print(f"\nStarting replay... (Frame rate: {self.frequency} Hz)")
        
        frame_dt = 1.0 / self.frequency
        last_frame_time = time.time()
        
        try:
            while self.running and self.viewer.is_running():
                current_time = time.time()
                
                if not self.is_paused:
                    # Check if it's time for the next frame
                    if current_time - last_frame_time >= frame_dt / self.playback_speed:
                        if not self.update_simulation():
                            break
                        
                        self.current_frame += 1
                        self.current_frame %= len(self.qpos_data)  # Wrap around
                        last_frame_time = current_time
                        
                        # Print progress
                        if self.current_frame % int(self.frequency) == 0:
                            progress = (self.current_frame / len(self.qpos_data)) * 100
                            print(f"Progress: {progress:.1f}% (Frame {self.current_frame}/{len(self.qpos_data)})")
                else:
                    # When paused, still sync viewer but don't advance frame
                    self.viewer.sync()
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)
            
        except KeyboardInterrupt:
            print("\nReplay interrupted by user")
        
        finally:
            self.running = False
            if self.viewer:
                self.viewer.close()
            print("Replay finished")


def main():
    parser = argparse.ArgumentParser(description="Replay recorded qpos data in MuJoCo")
    parser.add_argument("npz_file", type=str, help="Path to recorded npz file")
    parser.add_argument("--model", type=str, default=scene, help="Path to MuJoCo model file")
    parser.add_argument("--output", type=str, help="Output directory to save motion data (optional)")
    
    args = parser.parse_args()
    
    # Check if npz file exists
    if not os.path.exists(args.npz_file):
        print(f"Error: NPZ file not found: {args.npz_file}")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Create and run replay
    replay = SimpleReplay(
        npz_file_path=args.npz_file,
        mujoco_model_path=args.model,
    )

    if args.output:
        replay.export_motion_data(args.output)
    else:
        replay.replay_loop()


if __name__ == "__main__":
    main()
