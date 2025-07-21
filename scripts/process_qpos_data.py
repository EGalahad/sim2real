"""Script to process recorded qpos data with smoothing and qvel calculation

This script loads qpos data from npz files, applies smoothing with separate
parameters for positions, quaternions, and joints, then calculates velocities
and saves the processed data to a new npz file.

Features:
- Separate smoothing parameters for pelvis position, quaternion, and joints
- Proper quaternion smoothing using weighted averaging
- Qvel calculation from smoothed qpos
- Saves processed data with metadata
"""

import numpy as np
import argparse
import os
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d


def weighted_quaternion_mean(quats, weights):
    """Compute weighted mean of quaternions using the method from 
    "Averaging Quaternions" by F. Landis Markley et al.
    
    Args:
        quats: (n, 4) array of quaternions [w, x, y, z]
        weights: (n,) array of weights
        
    Returns:
        mean_quat: (4,) weighted mean quaternion [w, x, y, z]
    """
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Convert to [x, y, z, w] format for scipy
    quats_scipy = quats[:, [1, 2, 3, 0]]
    
    # Method 1: Weighted sum approach (works for small angular distances)
    # Check if all quaternions are close to each other
    max_dot = -1
    for i in range(len(quats_scipy)):
        for j in range(i+1, len(quats_scipy)):
            dot = np.abs(np.dot(quats_scipy[i], quats_scipy[j]))
            max_dot = max(max_dot, dot)
    
    if max_dot > 0.9:  # Quaternions are close, use simple weighted average
        weighted_sum = np.sum(weights[:, np.newaxis] * quats_scipy, axis=0)
        weighted_sum = weighted_sum / np.linalg.norm(weighted_sum)
    else:
        # Method 2: Matrix method for more general case
        # Build the symmetric 4x4 matrix
        M = np.zeros((4, 4))
        for i, (q, w) in enumerate(zip(quats_scipy, weights)):
            M += w * np.outer(q, q)
        
        # Find the eigenvector with largest eigenvalue
        eigenvals, eigenvecs = np.linalg.eigh(M)
        max_idx = np.argmax(eigenvals)
        weighted_sum = eigenvecs[:, max_idx]
        
        # Ensure positive w component (standard quaternion convention)
        if weighted_sum[3] < 0:
            weighted_sum = -weighted_sum
    
    # Convert back to [w, x, y, z] format
    return np.array([weighted_sum[3], weighted_sum[0], weighted_sum[1], weighted_sum[2]])


def smooth_quaternions(quats, sigma=1.0):
    """Smooth quaternions using proper weighted averaging
    
    Args:
        quats: (n_frames, 4) array of quaternions [w, x, y, z]
        sigma: smoothing parameter for Gaussian kernel
        
    Returns:
        smoothed_quats: (n_frames, 4) array of smoothed quaternions
    """
    n_frames = len(quats)
    if n_frames < 3:
        return quats.copy()
    
    # Create smoothed quaternions
    smoothed_quats = np.zeros_like(quats)
    
    # Use Gaussian weights for smoothing
    kernel_size = min(n_frames, int(6 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    half_kernel = kernel_size // 2
    x = np.arange(-half_kernel, half_kernel + 1)
    weights = np.exp(-0.5 * (x / sigma) ** 2)
    weights /= weights.sum()
    
    for i in range(n_frames):
        # Determine the range of frames to consider for smoothing
        start_idx = max(0, i - half_kernel)
        end_idx = min(n_frames, i + half_kernel + 1)
        
        # Get corresponding weights
        weight_start = max(0, half_kernel - i)
        weight_end = weight_start + (end_idx - start_idx)
        frame_weights = weights[weight_start:weight_end]
        frame_weights /= frame_weights.sum()
        
        # Get quaternions in the window
        window_quats = quats[start_idx:end_idx]
        
        # Handle sign ambiguity: ensure all quaternions are in the same hemisphere
        reference_quat = window_quats[0]
        for j in range(1, len(window_quats)):
            if np.dot(reference_quat, window_quats[j]) < 0:
                window_quats[j] = -window_quats[j]
        
        # Compute weighted average
        if len(window_quats) == 1:
            smoothed_quats[i] = window_quats[0]
        else:
            smoothed_quats[i] = weighted_quaternion_mean(window_quats, frame_weights)
    
    return smoothed_quats


def process_qpos_data(qpos_data, timestamps, pos_sigma=1.0, quat_sigma=1.0, joint_sigma=1.0):
    """Process qpos data with smoothing and qvel calculation
    
    Args:
        qpos_data: (n_frames, nq) array of joint positions
        timestamps: (n_frames,) array of timestamps
        pos_sigma: smoothing parameter for pelvis position (0:3)
        quat_sigma: smoothing parameter for pelvis quaternion (3:7)
        joint_sigma: smoothing parameter for joint angles (7:nq)
        
    Returns:
        smoothed_qpos: (n_frames, nq) smoothed positions
        qvel: (n_frames, nv) calculated velocities
    """
    n_frames, nq = qpos_data.shape
    smoothed_qpos = np.zeros_like(qpos_data)
    
    print(f"Processing {n_frames} frames with {nq} DOF")
    print(f"Smoothing parameters - Position: {pos_sigma:.3f}, Quaternion: {quat_sigma:.3f}, Joints: {joint_sigma:.3f}")
    
    # Smooth pelvis position (0:3)
    if pos_sigma > 0:
        smoothed_qpos[:, 0:3] = gaussian_filter1d(qpos_data[:, 0:3], sigma=pos_sigma, axis=0)
        print("✓ Smoothed pelvis position")
    else:
        smoothed_qpos[:, 0:3] = qpos_data[:, 0:3]
        print("✓ Copied pelvis position (no smoothing)")
    
    # Smooth pelvis quaternion (3:7)
    if quat_sigma > 0:
        smoothed_qpos[:, 3:7] = smooth_quaternions(qpos_data[:, 3:7], sigma=quat_sigma)
        print("✓ Smoothed pelvis quaternion")
    else:
        smoothed_qpos[:, 3:7] = qpos_data[:, 3:7]
        print("✓ Copied pelvis quaternion (no smoothing)")
    
    # Smooth joint positions (7:nq)
    if nq > 7:
        if joint_sigma > 0:
            smoothed_qpos[:, 7:] = gaussian_filter1d(qpos_data[:, 7:], sigma=joint_sigma, axis=0)
            print(f"✓ Smoothed {nq-7} joint angles")
        else:
            smoothed_qpos[:, 7:] = qpos_data[:, 7:]
            print(f"✓ Copied {nq-7} joint angles (no smoothing)")
    
    # Calculate qvel from smoothed qpos
    print("Calculating velocities...")
    dt = np.diff(timestamps)
    dt = np.append(dt, dt[-1])  # Extend dt to match qpos length
    
    # Initialize qvel (nv = nq - 1 for floating base + joints)
    nv = nq - 1
    qvel = np.zeros((n_frames, nv))
    
    # Calculate linear velocity for pelvis (0:3)
    qvel[:-1, 0:3] = np.diff(smoothed_qpos[:, 0:3], axis=0) / dt[:-1, np.newaxis]
    qvel[-1, 0:3] = qvel[-2, 0:3]  # Extend last velocity
    
    # Calculate angular velocity for pelvis quaternion (3:6 in qvel)
    for i in range(n_frames - 1):
        q1 = smoothed_qpos[i, 3:7]      # [w, x, y, z]
        q2 = smoothed_qpos[i+1, 3:7]    # [w, x, y, z]
        
        # Convert to scipy format [x, y, z, w]
        rot1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
        rot2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        
        # Calculate angular velocity
        rel_rot = rot1.inv() * rot2
        angle_axis = rel_rot.as_rotvec()
        angular_vel = angle_axis / dt[i]
        
        qvel[i, 3:6] = angular_vel
    
    qvel[-1, 3:6] = qvel[-2, 3:6]  # Extend last angular velocity
    
    # Calculate joint velocities (6:nv)
    if nv > 6:
        qvel[:-1, 6:] = np.diff(smoothed_qpos[:, 7:], axis=0) / dt[:-1, np.newaxis]
        qvel[-1, 6:] = qvel[-2, 6:]  # Extend last velocity
    
    print("✓ Calculated linear, angular, and joint velocities")
    
    return smoothed_qpos, qvel


def main():
    parser = argparse.ArgumentParser(description="Process qpos data with smoothing and velocity calculation")
    parser.add_argument("input_npz", type=str, help="Path to input npz file containing qpos data")
    parser.add_argument("--output", "-o", type=str, default=None, 
                       help="Output npz file path (default: adds '_processed' to input filename)")
    parser.add_argument("--pos_sigma", type=float, default=0.2,
                       help="Smoothing sigma for pelvis position (0:3) (default: 0.2)")
    parser.add_argument("--quat_sigma", type=float, default=0.2,
                       help="Smoothing sigma for pelvis quaternion (3:7) (default: 0.2)")
    parser.add_argument("--joint_sigma", type=float, default=0.1,
                       help="Smoothing sigma for joint angles (7:nq) (default: 0.1)")

    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_npz):
        print(f"Error: Input file not found: {args.input_npz}")
        return 1
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input_npz)[0]
        args.output = f"{base_name}_processed.npz"
    
    print(f"Loading data from: {args.input_npz}")
    
    # Load input data
    try:
        input_data = np.load(args.input_npz)
        qpos_data = input_data['qpos']
        timestamps = input_data['timestamps']
        
        # Get metadata
        frequency = float(input_data.get('frequency', 50.0))
        nq = int(input_data.get('nq', qpos_data.shape[1]))
        
        # Get other metadata if available
        joint_names = input_data.get('joint_names', None)
        
    except Exception as e:
        print(f"Error loading input file: {e}")
        return 1
    
    print(f"Loaded {len(qpos_data)} frames at {frequency} Hz")
    print(f"Data shape: {qpos_data.shape}")
    print(f"Duration: {(timestamps[-1] - timestamps[0]):.2f} seconds")
    
    # Scale sigma values by frequency if requested
    pos_sigma = args.pos_sigma * frequency
    quat_sigma = args.quat_sigma * frequency
    joint_sigma = args.joint_sigma * frequency

    # Process the data
    print("\nProcessing data...")
    smoothed_qpos, qvel = process_qpos_data(
        qpos_data, timestamps, pos_sigma, quat_sigma, joint_sigma
    )
    
    # Prepare output data
    output_data = {
        'qpos': smoothed_qpos,
        'qvel': qvel,
        'timestamps': timestamps,
        'frequency': frequency,
        'nq': nq,
        'nv': qvel.shape[1],
        'pos_sigma': pos_sigma,
        'quat_sigma': quat_sigma,
        'joint_sigma': joint_sigma,
        'processing_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'original_file': os.path.basename(args.input_npz)
    }
    
    # Add joint names if available
    if joint_names is not None:
        output_data['joint_names'] = joint_names
    
    # Save processed data
    print(f"\nSaving processed data to: {args.output}")
    try:
        np.savez_compressed(args.output, **output_data)
        
        # Calculate file sizes
        input_size_mb = os.path.getsize(args.input_npz) / (1024 * 1024)
        output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        
        print(f"✓ Processing complete!")
        print(f"  Input file:  {input_size_mb:.2f} MB")
        print(f"  Output file: {output_size_mb:.2f} MB")
        print(f"  Contains: smoothed qpos ({smoothed_qpos.shape}) and qvel ({qvel.shape})")
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
