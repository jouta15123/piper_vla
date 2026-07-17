from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EEPose:
    """End-effector pose in meters and degrees."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def rpy(self) -> Tuple[float, float, float]:
        return (self.rx, self.ry, self.rz)

    def moved_by(
        self,
        dxyz_m: Sequence[float],
        drpy_deg: Sequence[float],
    ) -> "EEPose":
        return EEPose(
            x=self.x + float(dxyz_m[0]),
            y=self.y + float(dxyz_m[1]),
            z=self.z + float(dxyz_m[2]),
            rx=self.rx + float(drpy_deg[0]),
            ry=self.ry + float(drpy_deg[1]),
            rz=self.rz + float(drpy_deg[2]),
        )


@dataclass(frozen=True)
class JointState:
    """Piper joint state in degrees."""

    values_deg: Tuple[float, float, float, float, float, float]

    def as_list(self) -> List[float]:
        return list(self.values_deg)


@dataclass(frozen=True)
class GripperState:
    """Piper gripper opening in meters.

    Piper SDK reports one gripper stroke value. OpenPI PiPER training used two
    robosuite gripper qpos values, so callers can duplicate this scalar when
    constructing an 8D policy state.
    """

    opening_m: float
    effort_n_m: Optional[float] = None
    status_code: Optional[float] = None

    def as_pair(self) -> List[float]:
        return [self.opening_m, self.opening_m]


@dataclass(frozen=True)
class SafetyPlane:
    """Half-space constraint in Piper/base coordinates.

    A target is allowed when dot(normal, target - point) >= margin_m.
    Use a unit normal when possible so margin_m has an intuitive distance.
    """

    name: str
    normal: Tuple[float, float, float]
    point: Tuple[float, float, float]
    margin_m: float = 0.0


@dataclass
class SafetyConfig:
    can_name: str = "can0"
    rosbridge_host: str = "127.0.0.1"
    rosbridge_port: int = 9090
    ros_command_topic: str = "/piper_vla/raw_command"
    ros_heartbeat_topic: str = "/piper_vla/heartbeat"
    ros_joint_topic: str = "/follower/current_joint_obs"
    ros_pose_topic: str = "/follower/current_end_pose"
    ros_status_topic: str = "/piper_vla/status"
    ros_arm_service: str = "/piper_vla/arm"
    judge_flag: bool = False
    can_auto_init: bool = True
    dh_is_offset: int = 1
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True
    # Explicitly allow takeover of an arm already held by the vendor Piper API
    # in CAN/MoveJ with all motors enabled. The default remains fail-closed.
    sdk_attach_enabled_can: bool = False

    dry_run: bool = True
    require_manual_approval: bool = True
    speed_pct: int = 10
    # The white-cylinder checkpoint was collected at 20 Hz.
    control_hz: float = 20.0
    step_sleep_s: float = 0.05
    require_status_available: bool = True
    max_start_pose_drift_m: float = 0.002
    max_start_rpy_drift_deg: float = 0.5
    max_start_joint_drift_deg: float = 1.0

    action_scale_xyz: float = 1.0
    action_scale_rpy: float = 1.0
    robosuite_osc_xyz_scale_m: float = 0.05
    robosuite_osc_rot_scale_rad: float = 0.05
    robosuite_gripper_open_action: float = -1.0
    robosuite_gripper_close_action: float = 0.0
    robosuite_gripper_qpos_max_m: float = 0.035
    robosuite_gripper_min_width_m: float = 0.020
    reject_on_clip: bool = True
    reject_on_warning: bool = False

    workspace_x_m: Tuple[float, float] = (0.20, 0.40)
    workspace_y_m: Tuple[float, float] = (-0.15, 0.15)
    workspace_z_m: Tuple[float, float] = (0.10, 0.30)
    min_z_m: float = 0.10
    safety_planes: Tuple[SafetyPlane, ...] = ()
    # Four ordered base-frame XYZ corners define the allowed XY polygon and a
    # fitted lower safety surface. Empty keeps the legacy box/planes only.
    workspace_floor_corners_m: Tuple[Tuple[float, float, float], ...] = ()
    workspace_floor_margin_m: float = 0.005
    workspace_floor_max_fit_error_m: float = 0.003

    max_step_xyz_m: Tuple[float, float, float] = (0.003, 0.003, 0.003)
    max_step_rpy_deg: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_total_translation_m: float = 0.050
    max_horizon: int = 25

    # Runtime Cartesian command shaping. The safety checker still validates the
    # complete requested target; the executor walks toward it with these limits.
    free_space_speed_m_s: float = 0.050
    near_table_speed_m_s: float = 0.015
    max_cartesian_accel_m_s2: float = 0.200
    max_rpy_speed_deg_s: float = 10.0
    max_rpy_accel_deg_s2: float = 50.0
    max_measured_joint_speed_deg_s: float = 30.0
    max_commanded_joint_speed_deg_s: float = 20.0
    joint_command_hz: float = 100.0
    near_table_distance_m: float = 0.050
    tracking_hold_error_m: float = 0.003
    tracking_abort_error_m: float = 0.005
    tracking_settle_timeout_s: float = 0.50
    settle_each_vla_step: bool = True

    # Convert Cartesian VLA targets to checked JointCtrl targets before real
    # execution. EndPoseCtrl remains an explicit compatibility mode.
    cartesian_execution_mode: str = "end_pose"
    ik_position_tolerance_m: float = 0.0005
    ik_rotation_tolerance_deg: float = 0.5
    ik_max_iterations: int = 60
    ik_damping: float = 0.02
    ik_jacobian_delta_rad: float = 0.001
    ik_max_update_deg: float = 1.0
    ik_path_sample_step_deg: float = 0.25

    # Calibrated geometry. tool_points_m are points expressed in the EE frame;
    # every configured point must remain above table_z_m + table_margin_m.
    calibration_complete: bool = False
    table_z_m: float = 0.0
    table_margin_m: float = 0.005
    tool_points_m: Tuple[Tuple[float, float, float], ...] = ()
    cylinder_diameter_m: float = 0.030
    cylinder_height_m: float = 0.030
    grasp_width_margin_m: float = 0.003
    hybrid_test_lift_m: float = 0.015
    hybrid_total_lift_m: float = 0.100
    expected_ready_joints_deg: Tuple[float, float, float, float, float, float] = (
        0.0,
        100.73,
        -64.93,
        0.0,
        58.89,
        0.0,
    )
    ready_joint_tolerance_deg: Tuple[float, float, float, float, float, float] = (
        5.0,
        8.0,
        8.0,
        5.0,
        10.0,
        5.0,
    )
    # Automatic ready return is additionally gated by calibrated joint
    # waypoints. These values only shape and monitor that measured path.
    ready_return_speed_pct: int = 2
    ready_return_max_step_deg: float = 0.5
    ready_return_tracking_tolerance_deg: float = 0.35
    ready_return_step_timeout_s: float = 4.0
    ready_return_total_timeout_s: float = 120.0
    ready_return_max_joint_speed_deg_s: float = 12.0
    ready_return_workspace_tolerance_m: float = 0.002
    # Optional base-to-ready corridor. When unset, the normal VLA workspace is
    # used. A calibrated tabletop polygon may start in front of the folded
    # arm, so real deployments can give ready return its own checked 3D box.
    ready_return_workspace_x_m: Optional[Tuple[float, float]] = None
    ready_return_workspace_y_m: Optional[Tuple[float, float]] = None
    ready_return_workspace_z_m: Optional[Tuple[float, float]] = None
    ready_return_enforce_workspace_floor_polygon: bool = True

    # Motor disable is allowed only after an operator-configured, mechanically
    # supported shutdown posture has been reached and verified. None keeps
    # normal shutdown disabled; the policy-ready pose is not assumed safe for
    # power removal.
    shutdown_joints_deg: Optional[Tuple[float, float, float, float, float, float]] = None
    shutdown_joint_tolerance_deg: Tuple[float, float, float, float, float, float] = (
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    )

    joint_limits_deg: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "j1": (-150.0, 150.0),
            "j2": (0.0, 180.0),
            "j3": (-170.0, 0.0),
            "j4": (-100.0, 100.0),
            "j5": (-70.0, 70.0),
            "j6": (-120.0, 120.0),
        }
    )
    joint_limit_margin_deg: float = 3.0
    max_joint_step_deg: Tuple[float, float, float, float, float, float] = (
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
    )

    gripper_open_m: float = 0.070
    gripper_closed_m: float = 0.000
    gripper_effort_n_m: float = 1.0

    log_dir: str = "logs"

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    index: int
    action_mode: str
    raw_action: List[float]
    scaled_action: List[float]
    clipped_action: List[float]
    start_pose: Optional[EEPose] = None
    target_pose: Optional[EEPose] = None
    start_joints: Optional[JointState] = None
    target_joints: Optional[JointState] = None
    gripper_m: Optional[float] = None
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action_mode": self.action_mode,
            "raw_action": self.raw_action,
            "scaled_action": self.scaled_action,
            "clipped_action": self.clipped_action,
            "start_pose": self.start_pose.as_list() if self.start_pose else None,
            "target_pose": self.target_pose.as_list() if self.target_pose else None,
            "start_joints": self.start_joints.as_list() if self.start_joints else None,
            "target_joints": self.target_joints.as_list() if self.target_joints else None,
            "gripper_m": self.gripper_m,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


@dataclass
class TrajectoryPlan:
    action_mode: str
    initial_pose: EEPose
    initial_joints: Optional[JointState]
    steps: List[TrajectoryStep]
    config_snapshot: Dict[str, Any]
    approved_by_safety: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_mode": self.action_mode,
            "initial_pose": self.initial_pose.as_list(),
            "initial_joints": self.initial_joints.as_list() if self.initial_joints else None,
            "steps": [step.to_dict() for step in self.steps],
            "config_snapshot": self.config_snapshot,
            "approved_by_safety": self.approved_by_safety,
            "summary": self.summary,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrajectoryPlan":
        pose_vals = data["initial_pose"]
        initial_pose = EEPose(*[float(x) for x in pose_vals])
        joint_vals = data.get("initial_joints")
        initial_joints = None
        if joint_vals is not None:
            initial_joints = JointState(tuple(float(x) for x in joint_vals))  # type: ignore[arg-type]
        steps: List[TrajectoryStep] = []
        for item in data.get("steps", []):
            start_pose = None
            target_pose = None
            start_joints = None
            target_joints = None
            if item.get("start_pose") is not None:
                start_pose = EEPose(*[float(x) for x in item["start_pose"]])
            if item.get("target_pose") is not None:
                target_pose = EEPose(*[float(x) for x in item["target_pose"]])
            if item.get("start_joints") is not None:
                start_joints = JointState(tuple(float(x) for x in item["start_joints"]))  # type: ignore[arg-type]
            if item.get("target_joints") is not None:
                target_joints = JointState(tuple(float(x) for x in item["target_joints"]))  # type: ignore[arg-type]
            steps.append(
                TrajectoryStep(
                    index=int(item["index"]),
                    action_mode=str(item["action_mode"]),
                    raw_action=[float(x) for x in item.get("raw_action", [])],
                    scaled_action=[float(x) for x in item.get("scaled_action", [])],
                    clipped_action=[float(x) for x in item.get("clipped_action", [])],
                    start_pose=start_pose,
                    target_pose=target_pose,
                    start_joints=start_joints,
                    target_joints=target_joints,
                    gripper_m=item.get("gripper_m"),
                    violations=list(item.get("violations", [])),
                    warnings=list(item.get("warnings", [])),
                )
            )
        return TrajectoryPlan(
            action_mode=data["action_mode"],
            initial_pose=initial_pose,
            initial_joints=initial_joints,
            steps=steps,
            config_snapshot=data.get("config_snapshot", {}),
            approved_by_safety=bool(data.get("approved_by_safety", False)),
            summary=str(data.get("summary", "")),
        )
