import numpy as np
from scipy.spatial.transform import Rotation as R

def yaw_from_quat(quat):
    return R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[:, 2:3]



def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c

def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def quat_multiply_numpy(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.concatenate([
        (w1*w2 - x1*x2 - y1*y2 - z1*z2)[:, None],
        (w1*x2 + x1*w2 + y1*z2 - z1*y2)[:, None],
        (w1*y2 - x1*z2 + y1*w2 + z1*x2)[:, None],
        (w1*z2 + x1*y2 - y1*x2 + z1*w2)[:, None]
    ], axis=1)

def quat_conjugate_numpy(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.concatenate([
        w[:, None],
        -x[:, None],
        -y[:, None],
        -z[:, None]
    ], axis=1)
