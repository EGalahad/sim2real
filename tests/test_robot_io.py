from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.robot_io import factory as robot_io_factory
from sim2real.rl_policy.robot_io.factory import create_robot_io
from sim2real.rl_policy.robot_io import g1 as g1_module
from sim2real.rl_policy.robot_io.g1 import G1RobotIO
from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO
from sim2real.rl_policy.utils.state_processor import StateProcessor
from sim2real.utils.common import LowCmdMessage, LowStateMessage


class DummyRobotCfg:
    name = "g1"
    joint_names = ("joint_0", "joint_1")
    domain_id = 0
    mocap_ip = "127.0.0.1"
    low_state_host = "127.0.0.1"
    low_state_port = 15591
    low_cmd_bind_addr = "*"
    low_cmd_port = 15592


class FakeSocket:
    def __init__(self) -> None:
        self.options: list[tuple[int, object]] = []
        self.received: list[bytes] = []
        self.sent: list[tuple[bytes, int]] = []
        self.connected: str | None = None
        self.bound: str | None = None
        self.closed = False

    def setsockopt(self, option: int, value: object) -> None:
        self.options.append((option, value))

    def connect(self, endpoint: str) -> None:
        self.connected = endpoint

    def bind(self, endpoint: str) -> None:
        self.bound = endpoint

    def recv(self, *, flags: int = 0) -> bytes:
        if not self.received:
            raise zmq.Again()
        return self.received.pop(0)

    def send(self, data: bytes, *, flags: int = 0) -> None:
        self.sent.append((data, flags))

    def close(self, *, linger: int = 0) -> None:
        self.closed = True


class FakeContext:
    def __init__(self) -> None:
        self.state_socket = FakeSocket()
        self.command_socket = FakeSocket()

    def socket(self, socket_type: int) -> FakeSocket:
        if socket_type == zmq.SUB:
            return self.state_socket
        assert socket_type == zmq.PUB
        return self.command_socket


def _low_state_bytes(*, tick: int = 7) -> bytes:
    return LowStateMessage(
        quaternion=np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        gyroscope=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        joint_positions=np.array([0.4, 0.5], dtype=np.float32),
        joint_velocities=np.array([0.6, 0.7], dtype=np.float32),
        joint_torques=np.array([0.8, 0.9], dtype=np.float32),
        tick=tick,
    ).to_bytes()


def test_zmq_robot_io_preserves_socket_options_and_state_layout() -> None:
    context = FakeContext()
    backend = ZMQRobotIO(DummyRobotCfg(), context=context)  # type: ignore[arg-type]
    context.state_socket.received.append(_low_state_bytes())

    state = backend.read_state()

    assert state is not None
    np.testing.assert_allclose(state.qpos[:3], 0.0)
    np.testing.assert_allclose(state.qpos[3:7], [0.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(state.qpos[7:], [0.4, 0.5])
    np.testing.assert_allclose(state.qvel[:3], 0.0)
    np.testing.assert_allclose(state.qvel[3:6], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.qvel[6:], [0.6, 0.7])
    np.testing.assert_allclose(state.joint_torque, [0.8, 0.9])
    assert state.tick == 7
    assert (zmq.SUBSCRIBE, b"") in context.state_socket.options
    assert (zmq.CONFLATE, 1) in context.state_socket.options
    assert (zmq.RCVTIMEO, 10) in context.state_socket.options
    assert (zmq.SNDHWM, 1) in context.command_socket.options
    assert (zmq.LINGER, 0) in context.command_socket.options


def test_zmq_robot_io_ignores_bad_message_and_keeps_latest_state() -> None:
    context = FakeContext()
    backend = ZMQRobotIO(DummyRobotCfg(), context=context)  # type: ignore[arg-type]
    context.state_socket.received.extend([_low_state_bytes(tick=11), b"broken"])

    state = backend.read_state()

    assert state is not None
    assert state.tick == 11
    assert backend.read_state() is state


def test_zmq_robot_io_writes_existing_low_command_abi_and_closes() -> None:
    context = FakeContext()
    backend = ZMQRobotIO(DummyRobotCfg(), context=context)  # type: ignore[arg-type]
    arrays = [np.array([index, index + 0.5], dtype=np.float32) for index in range(5)]

    backend.write_command(*arrays)

    data, flags = context.command_socket.sent[-1]
    command = LowCmdMessage.from_bytes(data)
    for actual, expected in zip(
        (command.q_target, command.dq_target, command.tau_ff, command.kp, command.kd),
        arrays,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert command.source_time_ns is not None
    assert command.source_time_ns > 0
    assert command.sequence == 1
    assert flags == zmq.DONTWAIT

    backend.close()
    assert context.state_socket.closed
    assert context.command_socket.closed


def test_low_cmd_message_metadata_footer_roundtrip() -> None:
    message = LowCmdMessage(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0], dtype=np.float32),
        source_time_ns=123456789,
        sequence=42,
    )

    decoded = LowCmdMessage.from_bytes(message.to_bytes())

    np.testing.assert_array_equal(decoded.q_target, [1.0, 2.0])
    np.testing.assert_array_equal(decoded.dq_target, [3.0, 4.0])
    np.testing.assert_array_equal(decoded.tau_ff, [5.0, 6.0])
    np.testing.assert_array_equal(decoded.kp, [7.0, 8.0])
    np.testing.assert_array_equal(decoded.kd, [9.0, 10.0])
    assert decoded.source_time_ns == 123456789
    assert decoded.sequence == 42


def test_low_cmd_message_rejects_invalid_metadata_footer() -> None:
    message = LowCmdMessage(
        np.array([1.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32),
        np.array([5.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="invalid metadata footer"):
        LowCmdMessage.from_bytes(message.to_bytes() + b"12345678")


class FakeG1Robot:
    def __init__(self) -> None:
        self.read_count = 0
        self.control_modes: list[object] = []
        self.commands: list[SimpleNamespace] = []
        self.closed = False
        self.low_state = SimpleNamespace(
            imu=SimpleNamespace(quat=[0.0, 0.0, 0.0, 1.0], omega=[1.0, 2.0, 3.0]),
            motor=SimpleNamespace(
                q=[0.1, 0.2, 99.0],
                dq=[0.3, 0.4, 99.0],
                tau_est=[0.5, 0.6, 99.0],
            ),
        )

    def set_control_mode(self, mode: object) -> None:
        self.control_modes.append(mode)

    def read_low_state(self) -> SimpleNamespace:
        self.read_count += 1
        return self.low_state

    def create_zero_command(self) -> SimpleNamespace:
        return SimpleNamespace()

    def write_low_command(self, command: SimpleNamespace) -> None:
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True


class FakeMotionSwitcherClient:
    def __init__(
        self,
        modes: list[tuple[int, dict[str, str] | None]] | None = None,
        *,
        release_status: int = 0,
    ) -> None:
        self.modes = list(modes or [(0, {"name": "sport"}), (0, {"name": ""})])
        self.release_status = release_status
        self.timeouts: list[float] = []
        self.init_count = 0
        self.check_count = 0
        self.release_count = 0

    def SetTimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def Init(self) -> None:
        self.init_count += 1

    def CheckMode(self) -> tuple[int, dict[str, str] | None]:
        self.check_count += 1
        return self.modes.pop(0)

    def ReleaseMode(self) -> tuple[int, None]:
        self.release_count += 1
        return self.release_status, None


def _fake_g1_sdk(
    robot: FakeG1Robot,
    *,
    on_create: object | None = None,
) -> SimpleNamespace:
    sdk = SimpleNamespace(
        RobotType=SimpleNamespace(G1="g1"),
        MessageType=SimpleNamespace(HG="hg"),
        ControlMode=SimpleNamespace(PR="pr"),
    )
    def create_robot(interface: str, robot_type: object, message_type: object) -> object:
        if callable(on_create):
            on_create()
        return robot

    sdk.create_robot = create_robot
    return sdk


def test_g1_robot_io_primes_first_read_and_converts_sdk_state() -> None:
    robot = FakeG1Robot()
    motion_switcher = FakeMotionSwitcherClient()
    channel_initializations: list[tuple[int, str]] = []
    backend = G1RobotIO(
        DummyRobotCfg(),  # type: ignore[arg-type]
        interface="enP8p1s0",
        sdk_module=_fake_g1_sdk(
            robot,
            on_create=lambda: (
                motion_switcher.check_count == 2
                and motion_switcher.release_count == 1
            )
            or pytest.fail("G1 SDK was created before debug mode was confirmed"),
        ),  # type: ignore[arg-type]
        motion_switcher_factory=lambda: motion_switcher,
        channel_factory_initialize=lambda domain_id, interface: (
            channel_initializations.append((domain_id, interface))
        ),
        motion_release_delay_s=0.0,
    )

    assert channel_initializations == [(0, "enP8p1s0")]
    assert motion_switcher.timeouts == [5.0]
    assert motion_switcher.init_count == 1
    assert motion_switcher.check_count == 2
    assert motion_switcher.release_count == 1
    assert robot.control_modes == ["pr"]
    assert robot.read_count == 1

    state = backend.read_state()

    assert robot.read_count == 2
    np.testing.assert_allclose(state.qpos, [0, 0, 0, 0, 0, 0, 1, 0.1, 0.2])
    np.testing.assert_allclose(state.qvel, [0, 0, 0, 1, 2, 3, 0.3, 0.4])
    np.testing.assert_allclose(state.joint_torque, [0.5, 0.6])
    assert state.tick > 0


def test_g1_robot_io_runs_debug_helper_before_importing_inline_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = FakeG1Robot()
    sdk = _fake_g1_sdk(robot)
    events: list[str] = []

    def run_debug_helper(command: list[str], **kwargs: object) -> None:
        events.append("debug_helper")
        assert "sim2real.rl_policy.robot_io.g1_debug_mode" in command
        assert kwargs == {"check": True, "timeout": 30.0}

    def import_sdk(name: str) -> object:
        assert name == "unitree_interface"
        events.append("import_inline_sdk")
        return sdk

    monkeypatch.setattr(g1_module.subprocess, "run", run_debug_helper)
    monkeypatch.setattr(g1_module.importlib, "import_module", import_sdk)

    G1RobotIO(DummyRobotCfg(), interface="enP8p1s0")  # type: ignore[arg-type]

    assert events == ["debug_helper", "import_inline_sdk"]


def test_g1_robot_io_writes_pr_command_arrays() -> None:
    robot = FakeG1Robot()
    backend = G1RobotIO(
        DummyRobotCfg(),  # type: ignore[arg-type]
        interface="enP8p1s0",
        sdk_module=_fake_g1_sdk(robot),  # type: ignore[arg-type]
        motion_switcher_factory=lambda: FakeMotionSwitcherClient(
            [(0, {"name": ""})]
        ),
        channel_factory_initialize=lambda domain_id, interface: None,
        motion_release_delay_s=0.0,
    )
    arrays = [np.array([index, index + 0.5], dtype=np.float64) for index in range(5)]

    backend.write_command(*arrays)

    command = robot.commands[-1]
    np.testing.assert_array_equal(command.q_target, arrays[0].astype(np.float32))
    np.testing.assert_array_equal(command.dq_target, arrays[1].astype(np.float32))
    np.testing.assert_array_equal(command.tau_ff, arrays[2].astype(np.float32))
    assert command.kp == arrays[3].astype(np.float32).tolist()
    assert command.kd == arrays[4].astype(np.float32).tolist()

    backend.close()
    assert robot.closed


def test_g1_robot_io_aborts_when_debug_mode_cannot_be_confirmed() -> None:
    robot = FakeG1Robot()
    motion_switcher = FakeMotionSwitcherClient([(1, None)])
    create_count = 0

    def record_create() -> None:
        nonlocal create_count
        create_count += 1

    with pytest.raises(RuntimeError, match="Failed to query the G1 motion mode"):
        G1RobotIO(
            DummyRobotCfg(),  # type: ignore[arg-type]
            interface="enP8p1s0",
            sdk_module=_fake_g1_sdk(robot, on_create=record_create),  # type: ignore[arg-type]
            motion_switcher_factory=lambda: motion_switcher,
            channel_factory_initialize=lambda domain_id, interface: None,
            motion_release_delay_s=0.0,
        )

    assert create_count == 0
    assert not robot.closed
    assert robot.control_modes == []
    assert robot.read_count == 0


def test_factory_rejects_unknown_mode_and_inline_robot() -> None:
    with pytest.raises(ValueError, match="Unsupported robot_io"):
        create_robot_io(
            mode="invalid",  # type: ignore[arg-type]
            robot_name="g1",
            robot_cfg=DummyRobotCfg(),  # type: ignore[arg-type]
            interface="eth0",
        )

    with pytest.raises(NotImplementedError, match="not implemented"):
        create_robot_io(
            mode="inline",
            robot_name="new_robot",
            robot_cfg=DummyRobotCfg(),  # type: ignore[arg-type]
            interface="eth0",
        )


def test_factory_selects_zmq_and_g1_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    zmq_backend = object()
    g1_backend = object()
    monkeypatch.setattr(robot_io_factory, "ZMQRobotIO", lambda cfg: zmq_backend)
    monkeypatch.setattr(g1_module, "G1RobotIO", lambda cfg, interface: g1_backend)

    assert create_robot_io(
        mode="zmq",
        robot_name="anything",
        robot_cfg=DummyRobotCfg(),  # type: ignore[arg-type]
        interface="ignored",
    ) is zmq_backend
    assert create_robot_io(
        mode="inline",
        robot_name="G1",
        robot_cfg=DummyRobotCfg(),  # type: ignore[arg-type]
        interface="enP8p1s0",
    ) is g1_backend


def test_g1_robot_io_reports_missing_optional_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(g1_module.importlib, "import_module", missing_module)
    with pytest.raises(ImportError, match="robot-g1"):
        G1RobotIO(  # type: ignore[arg-type]
            DummyRobotCfg(),
            interface="eth0",
            motion_switcher_factory=lambda: FakeMotionSwitcherClient(
                [(0, {"name": ""})]
            ),
            channel_factory_initialize=lambda domain_id, interface: None,
            motion_release_delay_s=0.0,
        )


def test_robot_io_is_abstract() -> None:
    with pytest.raises(TypeError):
        RobotIO()


class FakeRobotIO(RobotIO):
    def __init__(self, state: RobotState | None) -> None:
        self.state = state

    def read_state(self) -> RobotState | None:
        return self.state

    def write_command(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        pass


def test_state_processor_keeps_stable_views_and_copies_backend_state() -> None:
    state = RobotState(
        qpos=np.arange(9, dtype=np.float32),
        qvel=np.arange(8, dtype=np.float32) + 10,
        joint_torque=np.array([20, 21], dtype=np.float32),
        tick=22,
    )
    processor = StateProcessor(
        DummyRobotCfg(),  # type: ignore[arg-type]
        robot_io=FakeRobotIO(state),
    )
    qpos_view = processor.qpos
    joint_pos_view = processor.joint_pos

    assert processor._prepare_low_state()
    assert processor.qpos is qpos_view
    assert processor.joint_pos is joint_pos_view
    np.testing.assert_array_equal(processor.qpos, state.qpos)
    np.testing.assert_array_equal(processor.qvel, state.qvel)
    assert processor.latest_state is not None
    np.testing.assert_array_equal(processor.latest_state.joint_torque, [20, 21])
    assert processor.latest_state.tick == 22

    state.qpos[:] = -1
    state.qvel[:] = -2
    state.joint_torque[:] = -3
    np.testing.assert_array_equal(processor.qpos, np.arange(9, dtype=np.float32))
    np.testing.assert_array_equal(
        processor.qvel,
        np.arange(8, dtype=np.float32) + 10,
    )
    np.testing.assert_array_equal(processor.latest_state.joint_torque, [20, 21])


def test_state_processor_handles_none_and_rejects_invalid_layout() -> None:
    processor = StateProcessor(
        DummyRobotCfg(),  # type: ignore[arg-type]
        robot_io=FakeRobotIO(None),
    )
    assert not processor._prepare_low_state()

    processor.robot_io = FakeRobotIO(
        RobotState(
            qpos=np.zeros(8, dtype=np.float32),
            qvel=np.zeros(8, dtype=np.float32),
            joint_torque=np.zeros(2, dtype=np.float32),
            tick=0,
        )
    )
    with pytest.raises(ValueError, match="qpos shape"):
        processor._prepare_low_state()
