import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from piper_vla_guard.camera_adapter import InvalidCameraFrame, PersistentCamera, validate_rgb_frame
from piper_vla_guard.calibration_calculator import calculate_calibration_samples
from piper_vla_guard.executor import PlanExecutor
from piper_vla_guard.config import load_config
from piper_vla_guard.hybrid_pick import HybridPickController, HybridPickError, detect_white_cylinder
from piper_vla_guard.pick_calibration import (
    CameraGeometry,
    PickCalibration,
    apply_calibration_to_safety,
)
from piper_vla_guard.piper_adapter import (
    ArmInitializationRefused,
    MockPiperAdapter,
    PiperSDKAdapter,
    ROSBridgeAdapter,
)
from piper_vla_guard.piper_test import (
    InitializationRefused,
    _prepare_can_hold_from_teaching,
    _recover_target_limit_with_joint_hold,
)
from piper_vla_guard.policy_adapter import OpenPIClientError, OpenPIPolicyClient
from piper_vla_guard.real_loop import (
    _clamp_pick_descent_to_floor,
    _joint_path_samples,
    parse_args,
    _require_pure_vla_gate,
    _require_ready_pose,
    _validate_ready_path_fk,
)
from piper_vla_guard.safety import SafetyChecker
from piper_vla_guard.types import EEPose, JointState, SafetyConfig, SafetyPlane


def test_ros_bridge_cartesian_payload_uses_ros_units_without_opening_can():
    class FakeTopic:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class FakeRoslibpy:
        @staticmethod
        def Message(payload):
            return payload

    adapter = ROSBridgeAdapter(SafetyConfig())
    adapter._roslibpy = FakeRoslibpy
    adapter._command_topic = FakeTopic()
    adapter._joint_state = JointState((0.0, 90.0, -45.0, 0.0, 30.0, 180.0))
    adapter.command_end_pose(EEPose(0.3, 0.0, 0.2, 90.0, 0.0, -180.0), speed_pct=2)

    payload = adapter._command_topic.messages[0]
    assert payload["sequence"] == 1
    assert payload["control_mode"] == 2
    assert payload["joints_rad"] == pytest.approx([0.0, math.pi / 2, -math.pi / 4, 0.0, math.pi / 6, math.pi])
    assert payload["target_pose_m_rad"] == pytest.approx([0.3, 0.0, 0.2, math.pi / 2, 0.0, -math.pi])
    assert payload["speed_pct"] == 2.0


def test_persistent_camera_reopens_failed_capture_before_retry():
    frame = np.indices((32, 32, 3)).sum(axis=0).astype(np.uint8) * 7

    class FakeCapture:
        def __init__(self, reads):
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return self.reads.pop(0)

        def release(self):
            self.released = True

    class FakeCV2:
        COLOR_BGR2RGB = 1

        @staticmethod
        def cvtColor(value, _code):
            return value[..., ::-1]

    failed = FakeCapture([(False, None)])
    recovered = FakeCapture([(True, frame)])
    camera = PersistentCamera(6, retries=2, retry_sleep_s=0.0, warmup_frames=0)
    camera._cap = failed
    camera._cv2 = FakeCV2

    def reopen():
        failed.release()
        camera._cap = recovered

    camera._reopen = reopen  # type: ignore[method-assign]

    rgb = camera.read_rgb()

    assert failed.released
    assert rgb is not None and rgb.shape == frame.shape


def test_prepare_can_hold_is_refused_without_sending_commands(monkeypatch):
    class FakeInterface:
        def __init__(self):
            self.ctrl_mode = 2
            self.mode_feed = 0
            self.enabled = [True, True, True, False, True, True]
            self.events = []

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    arm_status=0,
                    ctrl_mode=self.ctrl_mode,
                    mode_feed=self.mode_feed,
                )
            )

        def GetArmEnableStatus(self):
            return list(self.enabled)

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))
            if values == (0x00, 0x00, 0x02):
                self.ctrl_mode = 0

        def MotionCtrl_2(self, *values):
            self.events.append(("motion2", values))
            self.ctrl_mode = 1
            self.mode_feed = 1

        def EnableArm(self, *values):
            self.events.append(("enable", values))
            self.enabled = [True] * 6

    class FakeApi:
        def get_joint_states(self):
            return ((0.0, 1.0, -1.0, 0.0, 0.8, 0.0), 0.0, 200.0)

        def get_end_pose_euler(self):
            return ((0.23, 0.0, 0.26, 0.0, 0.0, 0.0), 0.0, 200.0)

    monkeypatch.setattr("piper_vla_guard.piper_test.time.sleep", lambda _: None)
    interface = FakeInterface()

    with pytest.raises(InitializationRefused, match="Teaching-to-CAN preparation is disabled"):
        _prepare_can_hold_from_teaching(interface, FakeApi(), 2)

    assert interface.events == []
    assert interface.ctrl_mode == 2
    assert interface.mode_feed == 0
    assert interface.enabled == [True, True, True, False, True, True]


def test_target_limit_recovery_replaces_cartesian_fault_with_measured_movej_hold(monkeypatch):
    class FakeInterface:
        def __init__(self):
            self.arm_status = 4
            self.mode_feed = 2
            self.events = []

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    arm_status=self.arm_status,
                    ctrl_mode=1,
                    mode_feed=self.mode_feed,
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def MotionCtrl_2(self, *values):
            self.events.append(("mode", values))
            self.mode_feed = values[1]
            self.arm_status = 0

        def EnableArm(self, *values):
            self.events.append(("enable", values))

    class FakePiper:
        def get_joint_states(self):
            return ((0.0, 1.75, -1.13, 0.0, 1.03, 0.0), 0.0, 200.0)

    monkeypatch.setattr("piper_vla_guard.piper_test.time.sleep", lambda _: None)
    interface = FakeInterface()

    joints = _recover_target_limit_with_joint_hold(interface, FakePiper(), 2, timeout_s=1.0)

    names = [name for name, _ in interface.events]
    assert names[:3] == ["joints"] * 3
    assert names[3:] == ["joints", "mode", "joints", "enable", "joints"]
    assert interface.arm_status == 0
    assert interface.mode_feed == 1
    assert joints == pytest.approx([0.0, 1.75, -1.13, 0.0, 1.03, 0.0])


def _geometry():
    return CameraGeometry(
        source_points_px=((0, 0), (223, 0), (223, 223), (0, 223)),
        base_points_m=((0, 0), (0.223, 0), (0.223, 0.223), (0, 0.223)),
    )


def _calibration():
    return PickCalibration(
        complete=True,
        overhead=_geometry(),
        wrist=None,
        table_z_m=0.10,
        table_margin_m=0.002,
        tool_points_m=((0.0, 0.0, -0.02),),
        finger_center_offset_m=(0.0, 0.0, -0.02),
        cylinder_diameter_m=0.030,
        cylinder_height_m=0.030,
        ready_joints_deg=(0.0, 100.73, -64.93, 0.0, 58.89, 0.0),
        ready_tolerance_deg=(5.0, 8.0, 8.0, 5.0, 10.0, 5.0),
        workspace_floor_corners_m=(
            (0.0, 0.0, 0.10),
            (0.223, 0.0, 0.10),
            (0.223, 0.223, 0.10),
            (0.0, 0.223, 0.10),
        ),
    )


def test_calibrated_floor_supersedes_stale_legacy_workspace_and_table_plane():
    cfg = SafetyConfig()
    cfg.workspace_x_m = (0.0, 0.35)
    cfg.workspace_y_m = (-0.15, 0.15)
    cfg.workspace_z_m = (0.139718, 0.35)
    cfg.min_z_m = 0.139718
    cfg.safety_planes = (
        SafetyPlane("table_clearance", (0.0, 0.0, 1.0), (0.0, 0.0, 0.139718)),
        SafetyPlane("front_fixture", (0.0, -1.0, 0.0), (0.0, 0.30, 0.0)),
    )

    apply_calibration_to_safety(_calibration(), cfg)

    assert cfg.workspace_x_m == pytest.approx((0.0, 0.223))
    assert cfg.workspace_y_m == pytest.approx((0.0, 0.223))
    assert cfg.workspace_z_m == pytest.approx((0.105, 0.35))
    assert cfg.min_z_m == pytest.approx(0.105)
    assert [plane.name for plane in cfg.safety_planes] == ["front_fixture"]


def test_offline_calibration_calculator_builds_four_corners_and_tool_envelope():
    payload = {
        "probe_offset_m": [0.0, 0.0, -0.10],
        "floor_samples": [
            {"name": "A", "ee_pose_m_deg": [0.2, -0.1, 0.2, 0, 0, 0], "pixel_xy": [10, 10]},
            {"name": "B", "ee_pose_m_deg": [0.4, -0.1, 0.2, 0, 0, 0], "pixel_xy": [210, 10]},
            {"name": "C", "ee_pose_m_deg": [0.4, 0.1, 0.2, 0, 0, 0], "pixel_xy": [210, 210]},
            {"name": "D", "ee_pose_m_deg": [0.2, 0.1, 0.2, 0, 0, 0], "pixel_xy": [10, 210]},
        ],
        "finger_center_samples": [
            {"ee_pose_m_deg": [0.2, 0.0, 0.2, 0, 0, 0], "target_point_m": [0.2, 0.0, 0.1]},
            {"ee_pose_m_deg": [0.3, 0.0, 0.2, 0, 0, 0], "target_point_m": [0.3, 0.0, 0.1]},
            {"ee_pose_m_deg": [0.4, 0.0, 0.2, 0, 0, 0], "target_point_m": [0.4, 0.0, 0.1]},
        ],
        "tool_bounds_m": {"x": [-0.05, 0.05], "y": [-0.06, 0.06], "z": [-0.12, 0.02]},
    }

    result = calculate_calibration_samples(payload)

    assert result["workspace_floor_corners_m"][0] == pytest.approx([0.2, -0.1, 0.1])
    assert result["finger_center_offset_m"] == pytest.approx([0.0, 0.0, -0.1])
    assert len(result["tool_points_m"]) == 8
    assert result["diagnostics"]["floor_max_fit_error_m"] < 1e-9


def test_solid_green_camera_frame_is_rejected():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = 180
    with pytest.raises(InvalidCameraFrame, match="dominant-colour"):
        validate_rgb_frame(frame)


def test_uniform_xyz_scaling_preserves_direction():
    cfg = SafetyConfig(reject_on_clip=False)
    cfg.max_step_xyz_m = (0.005, 0.005, 0.005)
    pose = EEPose(0.30, 0.0, 0.20, 0, 0, 0)
    plan = SafetyChecker(cfg).build_plan(
        pose,
        [[0.1, 0.0, -0.4, 0.0, 0.0, 0.0, -1.0]],
        "robosuite_osc_pose",
    )
    dx, _, dz = plan.steps[0].clipped_action[:3]
    assert dz == pytest.approx(-0.005)
    assert dx / dz == pytest.approx(-0.25)


def test_tool_point_table_clearance_rejects_target():
    cfg = SafetyConfig(reject_on_clip=False)
    cfg.workspace_z_m = (0.0, 1.0)
    cfg.min_z_m = 0.0
    cfg.table_z_m = 0.10
    cfg.table_margin_m = 0.005
    cfg.tool_points_m = ((0.0, 0.0, -0.10),)
    pose = EEPose(0.30, 0.0, 0.20, 0, 0, 0)
    plan = SafetyChecker(cfg).build_plan(pose, [[0, 0, 0, 0]], "delta_base_m_deg")
    assert not plan.approved_by_safety
    assert "tool point" in plan.steps[0].violations[0]


def test_checkpoint_identity_mismatch_is_blocked():
    wrapper = OpenPIPolicyClient.__new__(OpenPIPolicyClient)
    wrapper.server_metadata = lambda: {  # type: ignore[method-assign]
        "piper_policy_identity": {
            "dataset_repo_id": "local/wrong",
            "checkpoint_step": 30000,
            "task_prompt": "lift the white cylinder 10cm",
            "fps": 20,
            "norm_stats_sha256": "abc",
        }
    }
    with pytest.raises(OpenPIClientError, match="Wrong policy server"):
        wrapper.validate_identity(
            dataset_repo_id="local/piper_topdown_lift",
            checkpoint_step=30000,
            prompt="lift the white cylinder 10cm",
            fps=20,
        )


def test_ready_pose_gate_rejects_training_distribution_mismatch():
    robot = MockPiperAdapter()
    robot.connect()
    with pytest.raises(RuntimeError, match="outside the calibrated checkpoint ready pose"):
        _require_ready_pose(robot, SafetyConfig())


def test_xy_vla_mode_accepts_bounded_chunk_arguments():
    args = parse_args(
        [
            "--mode",
            "xy-vla",
            "--max-cycles",
            "3",
            "--chunk-size",
            "5",
            "--max-loop-translation-m",
            "0.03",
        ]
    )
    assert args.mode == "xy-vla"
    assert args.max_cycles == 3
    assert args.chunk_size == 5


def test_xyz_vla_mode_accepts_bounded_chunk_arguments():
    args = parse_args(
        [
            "--mode",
            "xyz-vla",
            "--max-cycles",
            "1",
            "--chunk-size",
            "3",
            "--max-loop-translation-m",
            "0.015",
            "--arm-test-max-xy-m",
            "0.003",
            "--xyz-vla-max-z-m",
            "0.002",
        ]
    )
    assert args.mode == "xyz-vla"
    assert args.max_cycles == 1
    assert args.chunk_size == 3
    assert args.arm_test_max_xy_m == pytest.approx(0.003)
    assert args.xyz_vla_max_z_m == pytest.approx(0.002)


def test_pick_vla_mode_accepts_full_feedback_run_arguments():
    args = parse_args(
        [
            "--mode",
            "pick-vla",
            "--max-cycles",
            "30",
            "--chunk-size",
            "10",
            "--max-loop-translation-m",
            "0.12",
            "--pick-vla-max-xyz-m",
            "0.05",
        ]
    )
    assert args.mode == "pick-vla"
    assert args.max_cycles == 30
    assert args.chunk_size == 10
    assert args.pick_vla_max_xyz_m == pytest.approx(0.05)
    assert args.pick_vla_tracking_tolerance_m == pytest.approx(0.002)
    assert args.pick_vla_approval_word == "PICK"


def test_pick_floor_guard_clamps_descent_but_retains_gripper_and_lift_actions():
    cfg = SafetyConfig(
        table_z_m=0.0689,
        table_margin_m=0.005,
        min_z_m=0.0739,
        workspace_z_m=(0.0739, 0.35),
        tool_points_m=((0.0, 0.0, 0.0385),),
        max_step_xyz_m=(0.05, 0.05, 0.05),
        max_step_rpy_deg=(0.0, 0.0, 0.0),
    )
    actions = np.asarray(
        [
            [0.0, 0.0, -0.22, 0.0, 0.0, 0.0, -0.12],
            [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, -0.11],
        ],
        dtype=np.float32,
    )

    guarded, plan = _clamp_pick_descent_to_floor(
        checker=SafetyChecker(cfg),
        current_pose=EEPose(0.30, 0.0, 0.123, 180.0, 0.0, 180.0),
        current_joints=None,
        action_chunk=actions,
        action_mode="robosuite_osc_pose",
    )

    assert plan.approved_by_safety
    assert len(plan.steps) == 2
    assert guarded[0, 2] > actions[0, 2]
    assert guarded[1, 2] == pytest.approx(actions[1, 2])
    assert guarded[:, 6] == pytest.approx(actions[:, 6])
    # The descent target retains 1 mm between the tool point and the hard
    # 0.0739 m floor so joint/FK tracking error does not trip the hard guard.
    assert plan.steps[0].target_pose.z - 0.0385 >= 0.07489


def test_pure_vla_gate_requires_strict_acceptance(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "dataset_repo_id": "local/piper_topdown_lift",
                "checkpoint_step": 30000,
                "image_sensitivity_pass": True,
                "dry_run_trials": 20,
                "hybrid_trials": 10,
                "hybrid_successes": 8,
            }
        )
    )
    _require_pure_vla_gate(str(path))


def test_white_cylinder_detection_and_manual_approval_gate():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(image, (100, 120), 15, (255, 255, 255), -1)
    detection = detect_white_cylinder(image, _geometry(), expected_diameter_m=0.030)
    assert detection.base_xy_m == pytest.approx((0.100, 0.120), abs=0.002)

    cfg = SafetyConfig()
    calibration = _calibration()
    cfg.table_z_m = calibration.table_z_m
    cfg.tool_points_m = calibration.tool_points_m
    controller = HybridPickController(cfg, calibration)
    controller.phase = controller.phase.WAIT_APPROVAL
    robot = MockPiperAdapter()
    robot.connect()
    with pytest.raises(HybridPickError, match="not manually approved"):
        controller.run_approved_grasp(
            robot,
            executor=None,  # type: ignore[arg-type]
            checker=None,  # type: ignore[arg-type]
            logger=None,  # type: ignore[arg-type]
            dry_run=False,
            approved=False,
        )


def test_executor_uses_pause_hold_on_tracking_error():
    class StuckRobot(MockPiperAdapter):
        def command_end_pose(self, pose, speed_pct=10, move_mode="L"):
            pass

    cfg = SafetyConfig()
    cfg.tracking_hold_error_m = 0.0001
    cfg.tracking_abort_error_m = 0.0002
    cfg.tracking_settle_timeout_s = 0.0
    robot = StuckRobot()
    robot.connect()
    plan = SafetyChecker(cfg).build_plan(
        robot.read_ee_pose(), [[0.001, 0.0, 0.0, float("nan")]], "delta_base_m_deg", robot.read_joint_state()
    )
    result = PlanExecutor(robot, cfg).execute(plan, human_approved=True, dry_run=False)
    assert not result["ok"]
    assert not robot.status["fault"]
    assert result["stop_kind"] == "pause_hold"
    assert "Tracking error" in result["messages"][-1]


def test_executor_uses_pause_hold_when_tracking_does_not_settle():
    class StuckRobot(MockPiperAdapter):
        def command_end_pose(self, pose, speed_pct=10, move_mode="L"):
            pass

    cfg = SafetyConfig()
    cfg.tracking_hold_error_m = 0.0001
    cfg.tracking_abort_error_m = 0.001
    cfg.tracking_settle_timeout_s = 0.0
    robot = StuckRobot()
    robot.connect()
    plan = SafetyChecker(cfg).build_plan(
        robot.read_ee_pose(), [[0.0005, 0.0, 0.0, float("nan")]], "delta_base_m_deg", robot.read_joint_state()
    )
    result = PlanExecutor(robot, cfg).execute(plan, human_approved=True, dry_run=False)
    assert not result["ok"]
    assert not robot.status["fault"]
    assert robot.status["ctrl_mode"] == 1
    assert result["stop_kind"] == "pause_hold"
    assert "Tracking did not settle" in result["messages"][-1]


def test_executor_refreshes_endpose_target_until_tracking_settles(monkeypatch):
    class SlowlyTrackingRobot(MockPiperAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.pose_commands = 0

        def command_end_pose(self, pose, speed_pct=10, move_mode="L"):
            self.pose_commands += 1
            fraction = min(1.0, self.pose_commands / 4.0)
            current = self.pose
            self.pose = EEPose(
                current.x + fraction * (pose.x - current.x),
                current.y + fraction * (pose.y - current.y),
                current.z + fraction * (pose.z - current.z),
                pose.rx,
                pose.ry,
                pose.rz,
            )

    monkeypatch.setattr("piper_vla_guard.executor.time.sleep", lambda _: None)
    cfg = SafetyConfig(
        cartesian_execution_mode="end_pose",
        settle_each_vla_step=True,
        tracking_hold_error_m=0.00005,
        tracking_abort_error_m=0.001,
        tracking_settle_timeout_s=1.0,
    )
    robot = SlowlyTrackingRobot(cfg)
    robot.connect()
    plan = SafetyChecker(cfg).build_plan(
        robot.read_ee_pose(),
        [[0.0005, 0.0, 0.0, float("nan")]],
        "delta_base_m_deg",
        robot.read_joint_state(),
    )

    result = PlanExecutor(robot, cfg).execute(plan, human_approved=True, dry_run=False)

    assert result["ok"]
    assert robot.pose_commands > 1
    assert result["max_tracking_error_m"] <= cfg.tracking_hold_error_m


def test_executor_uses_pause_hold_on_joint_speed_spike():
    class JumpingRobot(MockPiperAdapter):
        def command_end_pose(self, pose, speed_pct=10, move_mode="L"):
            super().command_end_pose(pose, speed_pct, move_mode)
            self.joints = JointState((20.0, *self.joints.values_deg[1:]))

    cfg = SafetyConfig(max_measured_joint_speed_deg_s=10.0)
    robot = JumpingRobot()
    robot.connect()
    plan = SafetyChecker(cfg).build_plan(
        robot.read_ee_pose(), [[0.001, 0.0, 0.0, float("nan")]], "delta_base_m_deg", robot.read_joint_state()
    )
    result = PlanExecutor(robot, cfg).execute(plan, human_approved=True, dry_run=False)
    assert not result["ok"]
    assert not robot.status["fault"]
    assert result["stop_kind"] == "pause_hold"
    assert "Measured joint speed" in result["messages"][-1]


def test_sdk_emergency_stop_sends_only_explicit_estop():
    class FakePiper:
        def __init__(self):
            self.estop_calls = []

        def EmergencyStop(self, value):
            self.estop_calls.append(value)

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True
    message = adapter.emergency_stop()
    assert adapter.piper.estop_calls == [0x01]
    assert "may descend" in message


def test_sdk_pause_hold_uses_jointctrl_without_can_0x150():
    class FakePiper:
        def __init__(self):
            self.motion_calls = []
            self.mode_calls = []
            self.joint_calls = []

        def MotionCtrl_1(self, emergency_stop, track_ctrl, teach_ctrl):
            self.motion_calls.append((emergency_stop, track_ctrl, teach_ctrl))

        def MotionCtrl_2(self, ctrl_mode, move_mode, speed_pct, mit_mode):
            self.mode_calls.append((ctrl_mode, move_mode, speed_pct, mit_mode))

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1,
                    arm_status=0,
                    mode_feed=1,
                    teach_status=0,
                    motion_status=0,
                    trajectory_num=0,
                    err_code=0,
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=0,
                    joint_2=100000,
                    joint_3=-65000,
                    joint_4=0,
                    joint_5=59000,
                    joint_6=0,
                )
            )

        def JointCtrl(self, *values):
            self.joint_calls.append(values)

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True
    message = adapter.pause_hold()
    assert adapter.piper.motion_calls == []
    assert adapter.piper.mode_calls == [(0x01, 0x01, 10, 0x00)]
    assert len(adapter.piper.joint_calls) == 8
    assert "no EmergencyStop" in message


def test_sdk_normal_shutdown_refuses_without_configured_safe_pose():
    class FakePiper:
        pass

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True
    adapter.read_joint_state = lambda: JointState((0.0, 100.0, -65.0, 0.0, 59.0, 0.0))
    with pytest.raises(RuntimeError, match="shutdown_joints_deg is not configured"):
        adapter.shutdown_at_safe_pose()


def test_sdk_normal_shutdown_holds_then_disables_only_at_configured_pose(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.enabled = [True] * 6
            self.disable_calls = 0

        def DisablePiper(self):
            self.disable_calls += 1
            self.enabled = [False] * 6

        def GetArmEnableStatus(self):
            return list(self.enabled)

    cfg = SafetyConfig(shutdown_joints_deg=(0.0, 100.0, -65.0, 0.0, 59.0, 0.0))
    adapter = PiperSDKAdapter(cfg)
    adapter.piper = FakePiper()
    adapter.connected = True
    adapter.read_joint_state = lambda: JointState((0.2, 100.1, -65.1, 0.0, 59.0, 0.0))
    events = []
    adapter.pause_hold = lambda: events.append("hold") or "held"
    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)

    message = adapter.shutdown_at_safe_pose()

    assert events == ["hold"]
    assert adapter.piper.disable_calls == 1
    assert "all motors disabled" in message


def test_sdk_enable_primes_measured_pose_before_enabling(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.events = []
            self.enabled = False
            self.ctrl_mode = 0

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=61004,
                    Y_axis=1640,
                    Z_axis=151946,
                    RX_axis=178799,
                    RY_axis=531,
                    RZ_axis=166048,
                )
            )

        def GetArmEnableStatus(self):
            return [self.enabled] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=self.ctrl_mode,
                    arm_status=0,
                    mode_feed=0,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))

        def MotionCtrl_2(self, *values):
            self.events.append(("motion2", values))
            self.ctrl_mode = values[0]

        def EndPoseCtrl(self, *values):
            self.events.append(("pose", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            self.enabled = True
            return True

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.enable(speed_pct=2, move_mode="L")

    event_names = [event[0] for event in adapter.piper.events]
    assert event_names == ["motion1"] * 2 + ["enable"] + ["pose"] * 3 + ["motion2"] + ["pose"] * 23
    assert adapter.piper.events[3][1][:3] == (61004, 1640, 151946)
    assert "primed hold XYZ=(0.061004, 0.001640, 0.151946)" in message


def test_sdk_joint_feedback_filters_one_transient_zero_frame(monkeypatch):
    expected = (2.807, 60.949, -33.420, -8.430, 39.277, 18.443)
    glitch = (2.807, 60.949, 0.0, -8.430, 0.0, 18.443)

    class FakePiper:
        def __init__(self):
            self.samples = [expected] * 5 + [glitch] + [expected] * 4

        def GetArmJointMsgs(self):
            values = self.samples.pop(0) if self.samples else expected
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    **{f"joint_{index + 1}": round(value * 1000) for index, value in enumerate(values)}
                )
            )

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    first = adapter.read_joint_state()
    second = adapter.read_joint_state()

    assert first is not None and first.values_deg == pytest.approx(expected)
    assert second is not None and second.values_deg == pytest.approx(expected)


def test_sdk_movej_enables_in_standby_before_selecting_can_and_sending_target(monkeypatch):
    joints_deg = (2.874, 52.270, -34.397, 2.866, 43.220, 18.397)

    class FakePiper:
        def __init__(self):
            self.events = []
            self.enabled = False
            self.ctrl_mode = 0

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=100000, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    **{f"joint_{index + 1}": round(value * 1000) for index, value in enumerate(joints_deg)}
                )
            )

        def GetArmEnableStatus(self):
            return [self.enabled] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=self.ctrl_mode,
                    arm_status=0,
                    mode_feed=1 if self.ctrl_mode == 1 else 0,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))

        def MotionCtrl_2(self, *values):
            self.events.append(("motion2", values))
            self.ctrl_mode = values[0]

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            self.enabled = True
            return True

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.enable(speed_pct=2, move_mode="J")

    event_names = [event[0] for event in adapter.piper.events]
    assert event_names == ["motion1"] * 2 + ["enable"] + ["joints"] * 3 + ["motion2"] + ["joints"] * 23
    assert adapter.piper.events[3][1] == tuple(round(value * 1000) for value in joints_deg)
    assert "Enabled in STANDBY then selected mode=J" in message


def test_sdk_enable_refuses_to_switch_mode_with_motors_already_enabled():
    class FakePiper:
        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=0, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=0,
                    arm_status=0,
                    mode_feed=0,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    with pytest.raises(ArmInitializationRefused, match="motors are already enabled"):
        adapter.enable(speed_pct=2, move_mode="L")


def test_sdk_movej_refuses_enabled_standby_even_with_joint_feedback(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.events = []

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=100000, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=1000,
                    joint_2=60000,
                    joint_3=-33000,
                    joint_4=-8000,
                    joint_5=39000,
                    joint_6=18000,
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=0,
                    arm_status=0,
                    mode_feed=0,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))

        def MotionCtrl_2(self, *values):
            self.events.append(("motion2", values))

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            return True

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    with pytest.raises(ArmInitializationRefused, match="motors are already enabled"):
        adapter.enable(speed_pct=2, move_mode="J")

    assert adapter.piper.events == []


def test_sdk_explicit_attach_reuses_enabled_vendor_can_movej_without_reset(monkeypatch):
    joints_deg = (1.0, 60.0, -33.0, -8.0, 39.0, 18.0)

    class FakePiper:
        def __init__(self):
            self.events = []

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=300000, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    **{f"joint_{index}": round(value * 1000) for index, value in enumerate(joints_deg, 1)}
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1,
                    arm_status=0,
                    mode_feed=1,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_2(self, *values):
            self.events.append(("mode", values))

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            return True

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))

        def EmergencyStop(self, *values):
            self.events.append(("estop", values))

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    cfg = SafetyConfig(sdk_attach_enabled_can=True)
    adapter = PiperSDKAdapter(cfg)
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.enable(speed_pct=2, move_mode="J")

    names = [name for name, _ in adapter.piper.events]
    assert names[:3] == ["joints"] * 3
    assert names[3:] == [item for _ in range(20) for item in ("mode", "joints")]
    assert not {"enable", "motion1", "estop"}.intersection(names)
    assert "no reset/resume/disable sent" in message


def test_sdk_explicit_attach_restores_missing_enable_while_holding(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.enabled = [True, True, True, True, False, True]
            self.events = []

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=300000, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=0, joint_2=60000, joint_3=-33000,
                    joint_4=0, joint_5=39000, joint_6=0,
                )
            )

        def GetArmEnableStatus(self):
            return list(self.enabled)

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1, arm_status=0, mode_feed=1, motion_status=0,
                    trajectory_num=0, err_status=0,
                )
            )

        def MotionCtrl_2(self, *values):
            self.events.append(("mode", values))

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            self.enabled = [True] * 6
            return True

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    cfg = SafetyConfig(sdk_attach_enabled_can=True)
    adapter = PiperSDKAdapter(cfg)
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.enable(speed_pct=2, move_mode="J")

    assert adapter.piper.enabled == [True] * 6
    assert any(name == "enable" for name, _ in adapter.piper.events)
    assert "restored_missing_enable=True" in message


def test_sdk_enabled_movej_attach_primes_pose_before_endpose_movel_transition(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.mode_feed = 1
            self.events = []

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=317285,
                    Y_axis=0,
                    Z_axis=193996,
                    RX_axis=180000,
                    RY_axis=515,
                    RZ_axis=180000,
                )
            )

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=0,
                    joint_2=100501,
                    joint_3=-64920,
                    joint_4=0,
                    joint_5=58902,
                    joint_6=0,
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1,
                    arm_status=0,
                    mode_feed=self.mode_feed,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_2(self, *values):
            self.events.append(("mode", values))
            self.mode_feed = values[1]

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EndPoseCtrl(self, *values):
            self.events.append(("pose", values))

        def EmergencyStop(self, *values):
            self.events.append(("estop", values))

        def DisablePiper(self):
            self.events.append(("disable", ()))

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    cfg = SafetyConfig(sdk_attach_enabled_can=True, cartesian_execution_mode="end_pose")
    adapter = PiperSDKAdapter(cfg)
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.enable(speed_pct=2, move_mode="L")

    names = [name for name, _ in adapter.piper.events]
    first_pose = names.index("pose")
    assert all(name in ("joints", "mode") for name in names[:first_pose])
    assert adapter.piper.mode_feed == 2
    assert not {"estop", "disable"}.intersection(names)
    assert "Cartesian L hold active" in message


def test_sdk_enable_refuses_to_auto_resume_latched_estop():
    class FakePiper:
        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=0, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmEnableStatus(self):
            return [False] * 6

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1,
                    arm_status=1,
                    mode_feed=2,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    with pytest.raises(ArmInitializationRefused, match="never auto-resumes an E-stop"):
        adapter.enable(speed_pct=2, move_mode="L")


def test_sdk_enable_refuses_teaching_mode_without_sending_control(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.ctrl_mode = 2
            self.enabled = [True, True, True, False, True, True]
            self.events = []

        def GetArmEndPoseMsgs(self):
            return SimpleNamespace(
                end_pose=SimpleNamespace(
                    X_axis=100000, Y_axis=0, Z_axis=200000, RX_axis=0, RY_axis=0, RZ_axis=0
                )
            )

        def GetArmEnableStatus(self):
            return list(self.enabled)

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=1000,
                    joint_2=60000,
                    joint_3=-33000,
                    joint_4=-8000,
                    joint_5=39000,
                    joint_6=18000,
                )
            )

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=self.ctrl_mode,
                    arm_status=0,
                    mode_feed=0,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def MotionCtrl_1(self, *values):
            self.events.append(("motion1", values))

        def MotionCtrl_2(self, *values):
            self.events.append(("motion2", values))
            self.ctrl_mode = values[0]

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

        def EnablePiper(self):
            self.events.append(("enable", ()))
            self.enabled = [True] * 6
            return True

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    status = adapter.read_arm_status()
    assert status["ctrl_mode_name"] == "TEACHING_MODE"
    assert status["arm_status_name"] == "NORMAL"
    with pytest.raises(ArmInitializationRefused, match="TEACHING_MODE"):
        adapter.enable(speed_pct=2, move_mode="J")

    assert adapter.piper.events == []


def test_sdk_vendor_teaching_bootstrap_matches_stop_disable_resume_enable_order(monkeypatch):
    class FakePiper:
        def __init__(self):
            self.ctrl_mode = 2
            self.arm_status = 0
            self.mode_feed = 0
            self.enabled = [False] * 6
            self.events = []

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=self.ctrl_mode,
                    arm_status=self.arm_status,
                    mode_feed=self.mode_feed,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def GetArmEnableStatus(self):
            return list(self.enabled)

        def GetArmJointMsgs(self):
            return SimpleNamespace(
                joint_state=SimpleNamespace(
                    joint_1=0,
                    joint_2=5000,
                    joint_3=-5000,
                    joint_4=0,
                    joint_5=20000,
                    joint_6=0,
                )
            )

        def EnablePiper(self):
            self.events.append(("enable", ()))
            self.enabled = [True] * 6
            return True

        def EmergencyStop(self, value):
            self.events.append(("estop", (value,)))
            self.arm_status = 1 if value == 0x01 else 0

        def DisablePiper(self):
            self.events.append(("disable", ()))
            self.enabled = [False] * 6
            return True

        def MotionCtrl_2(self, *values):
            self.events.append(("mode", values))
            self.ctrl_mode = values[0]
            self.mode_feed = values[1]

        def JointCtrl(self, *values):
            self.events.append(("joints", values))

    monkeypatch.setattr("piper_vla_guard.piper_adapter.time.sleep", lambda _: None)
    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    message = adapter.bootstrap_teaching_to_can(speed_pct=2, timeout_s=1.0)

    events = adapter.piper.events
    first_enable = next(i for i, event in enumerate(events) if event[0] == "enable")
    stop = events.index(("estop", (0x01,)))
    disable = events.index(("disable", ()))
    resume = events.index(("estop", (0x02,)))
    first_mode = next(i for i, event in enumerate(events) if event[0] == "mode")
    assert first_enable < stop < disable < resume < first_mode
    assert adapter.read_arm_status()["ctrl_mode_name"] == "CAN_CTRL"
    assert adapter.read_arm_status()["mode_feed_name"] == "MOVE_J"
    assert adapter.piper.enabled == [True] * 6
    assert "VENDOR TEACHING BOOTSTRAP COMPLETE" in message


def test_sdk_vendor_teaching_bootstrap_refuses_non_teaching_without_commands():
    class FakePiper:
        def __init__(self):
            self.events = []

        def GetArmStatus(self):
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=1,
                    arm_status=0,
                    mode_feed=1,
                    motion_status=0,
                    trajectory_num=0,
                    err_status=0,
                )
            )

        def GetArmEnableStatus(self):
            return [True] * 6

    adapter = PiperSDKAdapter(SafetyConfig())
    adapter.piper = FakePiper()
    adapter.connected = True

    with pytest.raises(ArmInitializationRefused, match="NORMAL TEACHING_MODE"):
        adapter.bootstrap_teaching_to_can(speed_pct=2, timeout_s=1.0)

    assert adapter.piper.events == []


def test_ready_path_fk_rejects_checkpoint_outside_workspace():
    cfg = SafetyConfig()
    cfg.workspace_x_m = (0.0, 0.25)
    cfg.workspace_y_m = (-0.15, 0.15)
    cfg.workspace_z_m = (0.10, 0.35)
    cfg.min_z_m = 0.10
    ready = (0.0, 100.73, -64.93, 0.0, 58.89, 0.0)
    measured = EEPose(0.3176198, 0.0, 0.1927714, 180.0, 0.31, 180.0)

    with pytest.raises(RuntimeError, match="outside ready-return workspace"):
        _validate_ready_path_fk(ready, [ready], measured, cfg)


def test_ready_path_uses_arbitrary_current_feedback_as_implicit_start():
    cfg = SafetyConfig()
    cfg.workspace_x_m = (0.0, 0.40)
    cfg.workspace_y_m = (-0.20, 0.20)
    cfg.workspace_z_m = (0.10, 0.35)
    cfg.min_z_m = 0.10
    cfg.safety_planes = ()
    current = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ready = (0.0, 100.73, -64.93, 0.0, 58.89, 0.0)
    measured = EEPose(0.0561275, 0.0, 0.2132663, 0.0, 85.0, 0.0)

    _validate_ready_path_fk(current, [ready], measured, cfg)


def test_real_ready_corridor_accepts_folded_bootstrap_to_ready_outside_table_polygon():
    cfg = load_config("configs/safety.example.yaml")
    calibration = PickCalibration(
        complete=False,
        table_z_m=0.0689,
        table_margin_m=0.005,
        cylinder_diameter_m=0.05,
        cylinder_height_m=0.05,
        workspace_floor_corners_m=(
            (0.112401, -0.277305, 0.0689),
            (0.111424, 0.274171, 0.0689),
            (0.516469, 0.292402, 0.0689),
            (0.512751, -0.303669, 0.0689),
        ),
        tool_points_m=((-0.00978, 0.00327, 0.03850),),
        finger_center_offset_m=(0.0, 0.0, 0.0),
        ready_joints_deg=(0.0, 100.73, -64.93, 0.0, 58.89, 0.0),
        ready_tolerance_deg=(5.0, 8.0, 8.0, 5.0, 10.0, 5.0),
        overhead=None,
        wrist=None,
    )
    apply_calibration_to_safety(calibration, cfg)
    folded = (3.612, 0.0, 0.0, -3.798, 23.091, 10.265)
    ready = calibration.ready_joints_deg
    measured = EEPose(0.050506, 0.0009, 0.1772, 161.4, 70.9, 162.5)

    assert measured.x < cfg.workspace_x_m[0]
    assert cfg.ready_return_enforce_workspace_floor_polygon is False
    _validate_ready_path_fk(folded, [ready], measured, cfg)


def test_ready_preflight_and_execution_share_proportional_joint_samples():
    start = (3.612, 0.0, 0.0, -3.798, 23.110, 10.265)
    ready = (0.0, 100.73, -64.93, 0.0, 58.89, 0.0)

    samples = _joint_path_samples(start, [ready], 0.5)

    assert samples[-1][3] == pytest.approx(ready)
    first = samples[0][3]
    fractions = [
        (first[i] - start[i]) / (ready[i] - start[i])
        for i in range(6)
        if ready[i] != start[i]
    ]
    assert max(fractions) - min(fractions) < 1e-12
    assert max(abs(first[i] - start[i]) for i in range(6)) <= 0.5


def test_ready_corridor_accepts_measured_mid_return_hold_and_descending_replan():
    cfg = load_config("configs/safety.example.yaml")
    current = (0.0, 56.95, -56.95, 0.0, 58.89, 0.0)
    ready = (0.0, 100.73, -64.93, 0.0, 58.89, 0.0)
    measured = EEPose(0.1816, 0.0, 0.3523, 180.0, 0.0, 180.0)

    _validate_ready_path_fk(current, [ready], measured, cfg)


def test_ready_path_requires_final_waypoint_to_match_checkpoint():
    calibration = _calibration()
    calibration = PickCalibration(
        **{
            **calibration.__dict__,
            "ready_path_joints_deg": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }
    )
    with pytest.raises(Exception, match="final .* waypoint must equal"):
        calibration.require_ready_path()


def test_empty_ready_path_resolves_to_direct_ready_move():
    calibration = _calibration()
    calibration = PickCalibration(
        **{
            **calibration.__dict__,
            "ready_path_joints_deg": (),
        }
    )

    calibration.require_ready_path()

    assert calibration.resolved_ready_path_joints_deg() == (calibration.ready_joints_deg,)
