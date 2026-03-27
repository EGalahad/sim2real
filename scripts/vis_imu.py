"""Visualize IMU quaternion as a root frame in Viser."""

import argparse
import time

import numpy as np
import viser
import zmq

from sim2real.utils.common import LowStateMessage, PORTS


def quat_wxyz_to_rpy_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = [float(v) for v in quat_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def make_arrow_segments(
    vec: np.ndarray,
    head_length_ratio: float = 0.25,
    head_width_ratio: float = 0.18,
) -> np.ndarray:
    """Build 3 line segments for an arrow from origin to vec (shape (3, 2, 3))."""
    v = np.asarray(vec, dtype=np.float32)
    length = float(np.linalg.norm(v))
    if length < 1e-8:
        return np.zeros((3, 2, 3), dtype=np.float32)

    direction = v / length
    tip = v
    head_len = max(1e-4, length * head_length_ratio)
    head_width = max(1e-4, length * head_width_ratio)

    # Build an orthonormal basis around direction.
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(ref, direction))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side = np.cross(direction, ref)
    side_norm = float(np.linalg.norm(side))
    if side_norm < 1e-8:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / side_norm

    tail = np.zeros(3, dtype=np.float32)
    neck = tip - direction * head_len
    wing_a = neck + side * head_width
    wing_b = neck - side * head_width

    segments = np.stack(
        [
            np.stack([tail, tip], axis=0),    # shaft
            np.stack([tip, wing_a], axis=0),  # head side A
            np.stack([tip, wing_b], axis=0),  # head side B
        ],
        axis=0,
    )
    return segments.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize root IMU frame from low_state quaternion.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP of low_state publisher")
    parser.add_argument("--port", type=int, default=PORTS.get("low_state", 5590), help="ZMQ low_state port")
    parser.add_argument("--axes-length", type=float, default=0.25, help="Frame axis length")
    parser.add_argument("--axes-radius", type=float, default=0.01, help="Frame axis radius")
    parser.add_argument("--angvel-scale", type=float, default=0.2, help="Scale factor for angvel vector visualization")
    parser.add_argument("--rate-hz", type=float, default=60.0, help="Visualization update rate")
    parser.add_argument("--print-rpy", action="store_true", help="Print roll/pitch/yaw in radians")
    parser.add_argument("--print-angvel", action="store_true", help="Print angular velocity in rad/s")
    args = parser.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"

    ctx = zmq.Context.instance()
    low_state_socket: zmq.Socket = ctx.socket(zmq.SUB)
    low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    low_state_socket.setsockopt(zmq.CONFLATE, 1)
    low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
    low_state_socket.connect(endpoint)

    server = viser.ViserServer()
    server.scene.add_grid("/world", width=4.0, height=4.0, width_segments=8, height_segments=8)
    root_frame = server.scene.add_frame(
        "/root",
        axes_length=args.axes_length,
        axes_radius=args.axes_radius,
        show_axes=True,
    )
    angvel_arrow = server.scene.add_line_segments(
        "/root/angvel_arrow",
        points=np.zeros((3, 2, 3), dtype=np.float32),
        colors=np.array(
            [
                [[255, 200, 0], [255, 200, 0]],
                [[255, 120, 0], [255, 120, 0]],
                [[255, 120, 0], [255, 120, 0]],
            ],
            dtype=np.uint8,
        ),
        line_width=3.0,
    )

    print(f"Subscribed low_state: {endpoint}")
    print("Viser started. Open the URL shown by viser in your browser.")

    dt = 1.0 / max(args.rate_hz, 1e-3)
    last_print_t = 0.0

    while True:
        latest_state = None
        while True:
            try:
                data = low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                latest_state = LowStateMessage.from_bytes(data)
            except Exception:
                latest_state = None

        if latest_state is not None:
            quat_wxyz = np.asarray(latest_state.quaternion, dtype=np.float32)
            norm = float(np.linalg.norm(quat_wxyz))
            if norm > 1e-8:
                quat_wxyz = quat_wxyz / norm

            root_frame.wxyz = quat_wxyz

            angvel = np.asarray(latest_state.gyroscope, dtype=np.float32)
            arrow_segments = make_arrow_segments(angvel * args.angvel_scale)
            try:
                angvel_arrow.points = arrow_segments
            except Exception:
                # Fallback for viser versions where in-place point updates are unsupported.
                angvel_arrow.remove()
                angvel_arrow = server.scene.add_line_segments(
                    "/root/angvel_arrow",
                    points=arrow_segments,
                    colors=np.array(
                        [
                            [[255, 200, 0], [255, 200, 0]],
                            [[255, 120, 0], [255, 120, 0]],
                            [[255, 120, 0], [255, 120, 0]],
                        ],
                        dtype=np.uint8,
                    ),
                    line_width=3.0,
                )

            if args.print_rpy:
                now = time.time()
                if now - last_print_t > 0.2:
                    rpy = quat_wxyz_to_rpy_xyz(quat_wxyz)
                    print(f"rpy(rad): {rpy[0]: .3f}, {rpy[1]: .3f}, {rpy[2]: .3f}")
                    if args.print_angvel:
                        print(f"angvel(rad/s): {angvel[0]: .3f}, {angvel[1]: .3f}, {angvel[2]: .3f}")
                    last_print_t = now

        time.sleep(dt)


if __name__ == "__main__":
    main()
