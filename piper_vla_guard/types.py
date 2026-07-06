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


@dataclass
class SafetyConfig:
    can_name: str = "can0"
    judge_flag: bool = False
    can_auto_init: bool = True
    dh_is_offset: int = 1
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True

    dry_run: bool = True
    require_manual_approval: bool = True
    speed_pct: int = 10
    step_sleep_s: float = 0.20
    require_status_available: bool = True
    max_start_pose_drift_m: float = 0.002
    max_start_rpy_drift_deg: float = 0.5
    max_start_joint_drift_deg: float = 1.0

    action_scale_xyz: float = 1.0
    action_scale_rpy: float = 1.0
    reject_on_clip: bool = True
    reject_on_warning: bool = False

    workspace_x_m: Tuple[float, float] = (0.20, 0.40)
    workspace_y_m: Tuple[float, float] = (-0.15, 0.15)
    workspace_z_m: Tuple[float, float] = (0.10, 0.30)
    min_z_m: float = 0.10

    max_step_xyz_m: Tuple[float, float, float] = (0.003, 0.003, 0.003)
    max_step_rpy_deg: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_total_translation_m: float = 0.050
    max_horizon: int = 25

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
