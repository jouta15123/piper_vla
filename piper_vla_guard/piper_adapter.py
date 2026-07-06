from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol

from .types import EEPose, JointState, SafetyConfig
from .utils import deg_to_sdk_angle, get_first_attr, m_to_sdk_pos, sdk_angle_to_deg, sdk_pos_to_m, unwrap_message


class RobotAdapter(Protocol):
    def connect(self) -> str: ...
    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str: ...
    def read_ee_pose(self) -> EEPose: ...
    def read_joint_state(self) -> Optional[JointState]: ...
    def read_arm_status(self) -> Dict[str, Any]: ...
    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None: ...
    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None: ...
    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None: ...
    def emergency_stop(self) -> str: ...
    def resume(self) -> str: ...
    def disable(self) -> str: ...


MOVE_MODE_TO_CODE = {
    "P": 0x00,
    "J": 0x01,
    "L": 0x02,
    "C": 0x03,
    "M": 0x04,
    "CPV": 0x05,
}


class PiperSDKAdapter:
    """Thin adapter over piper_sdk.C_PiperInterface_V2.

    The adapter keeps SDK-specific units and object field names out of the
    safety layer. It uses documented field candidates but still performs
    defensive introspection because piper_sdk message classes have changed over
    time.
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.piper: Any = None
        self.connected = False

    def connect(self) -> str:
        try:
            from piper_sdk import C_PiperInterface_V2  # type: ignore
            try:
                from piper_sdk import LogLevel  # type: ignore
            except Exception:  # pragma: no cover
                LogLevel = None  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "piper_sdk is not installed. Install with: pip install piper_sdk python-can"
            ) from exc

        kwargs = {
            "can_name": self.cfg.can_name,
            "judge_flag": self.cfg.judge_flag,
            "can_auto_init": self.cfg.can_auto_init,
            "dh_is_offset": self.cfg.dh_is_offset,
            "start_sdk_joint_limit": self.cfg.start_sdk_joint_limit,
            "start_sdk_gripper_limit": self.cfg.start_sdk_gripper_limit,
        }
        if LogLevel is not None:
            kwargs["logger_level"] = LogLevel.WARNING
        self.piper = C_PiperInterface_V2(**kwargs)
        self.piper.ConnectPort()
        time.sleep(0.05)
        self.connected = True
        return f"Connected to Piper on {self.cfg.can_name}"

    def _require(self) -> Any:
        if self.piper is None or not self.connected:
            raise RuntimeError("Piper is not connected")
        return self.piper

    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        p = self._require()
        code = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])
        if hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, code, int(speed_pct), 0x00)
            time.sleep(0.01)
        if hasattr(p, "EnableArm"):
            p.EnableArm(7, 0x02)
            time.sleep(0.05)
        return f"Enabled arm, mode={move_mode}, speed={speed_pct}%"

    def read_ee_pose(self) -> EEPose:
        p = self._require()
        msg = p.GetArmEndPoseMsgs()
        payload = unwrap_message(msg, ["end_pose", "pose", "arm_end_pose"])
        x = _required_number(payload, ["X_axis", "x_axis", "X", "x"])
        y = _required_number(payload, ["Y_axis", "y_axis", "Y", "y"])
        z = _required_number(payload, ["Z_axis", "z_axis", "Z", "z"])
        rx = _required_number(payload, ["RX_axis", "rx_axis", "RX", "rx", "roll"])
        ry = _required_number(payload, ["RY_axis", "ry_axis", "RY", "ry", "pitch"])
        rz = _required_number(payload, ["RZ_axis", "rz_axis", "RZ", "rz", "yaw"])
        return EEPose(
            x=sdk_pos_to_m(x),
            y=sdk_pos_to_m(y),
            z=sdk_pos_to_m(z),
            rx=sdk_angle_to_deg(rx),
            ry=sdk_angle_to_deg(ry),
            rz=sdk_angle_to_deg(rz),
        )

    def read_joint_state(self) -> Optional[JointState]:
        p = self._require()
        if not hasattr(p, "GetArmJointMsgs"):
            return None
        msg = p.GetArmJointMsgs()
        payload = unwrap_message(msg, ["joint_state", "joint_states", "arm_joint", "joints"])
        vals = []
        for i in range(1, 7):
            raw = get_first_attr(payload, [f"joint_{i}", f"joint{i}", f"j{i}"])
            if raw is None:
                return None
            vals.append(sdk_angle_to_deg(float(raw)))
        return JointState(tuple(vals))  # type: ignore[arg-type]

    def read_arm_status(self) -> Dict[str, Any]:
        p = self._require()
        if not hasattr(p, "GetArmStatus"):
            return {"available": False}
        msg = p.GetArmStatus()
        payload = unwrap_message(msg, ["arm_status", "status"])
        fields = {
            "available": True,
            "ctrl_mode": _optional_number(payload, ["ctrl_mode", "control_mode"]),
            "arm_status": _optional_number(payload, ["arm_status", "status", "arm_status_code"]),
            "mode_feed": _optional_number(payload, ["mode_feed", "mode"]),
            "motion_status": _optional_number(payload, ["motion_status"]),
            "trajectory_num": _optional_number(payload, ["trajectory_num"]),
        }
        err_obj = get_first_attr(payload, ["err_status", "error_status", "err"])
        fields["err_status"] = _err_status_to_dict(err_obj)
        code = fields["arm_status"]
        fields["fault"] = bool(code not in (None, 0, 0.0)) or _err_status_has_fault(err_obj)
        return fields

    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None:
        p = self._require()
        code = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])
        if hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, code, int(speed_pct), 0x00)
        p.EndPoseCtrl(
            m_to_sdk_pos(pose.x),
            m_to_sdk_pos(pose.y),
            m_to_sdk_pos(pose.z),
            deg_to_sdk_angle(pose.rx),
            deg_to_sdk_angle(pose.ry),
            deg_to_sdk_angle(pose.rz),
        )

    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None:
        p = self._require()
        if hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
        p.JointCtrl(*[deg_to_sdk_angle(v) for v in joints.values_deg])

    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None:
        p = self._require()
        p.GripperCtrl(
            m_to_sdk_pos(gripper_m),
            int(round(effort_n_m * 1000.0)),
            0x01,
            0x00,
        )

    def emergency_stop(self) -> str:
        p = self._require()
        p.EmergencyStop(0x01)
        return "Emergency stop sent"

    def resume(self) -> str:
        p = self._require()
        p.EmergencyStop(0x02)
        return "Emergency stop resume sent"

    def disable(self) -> str:
        p = self._require()
        if hasattr(p, "DisableArm"):
            p.DisableArm(7, 0x01)
        return "Disable arm sent"


class MockPiperAdapter:
    """A deterministic mock adapter for UI dry-run and tests."""

    def __init__(self) -> None:
        self.connected = False
        self.pose = EEPose(0.30, 0.00, 0.20, 0.0, 0.0, 0.0)
        self.joints = JointState((0.0, 60.0, -80.0, 0.0, 20.0, 0.0))
        self.status: Dict[str, Any] = {
            "available": True,
            "ctrl_mode": 1,
            "arm_status": 0,
            "mode_feed": 2,
            "motion_status": 0,
            "trajectory_num": 0,
            "err_status": None,
            "fault": False,
        }

    def connect(self) -> str:
        self.connected = True
        return "Connected to mock Piper"

    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        return f"Mock enabled, mode={move_mode}, speed={speed_pct}%"

    def read_ee_pose(self) -> EEPose:
        return self.pose

    def read_joint_state(self) -> Optional[JointState]:
        return self.joints

    def read_arm_status(self) -> Dict[str, Any]:
        return dict(self.status)

    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None:
        self.pose = pose
        self.status["trajectory_num"] = int(self.status.get("trajectory_num", 0)) + 1

    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None:
        self.joints = joints
        self.status["trajectory_num"] = int(self.status.get("trajectory_num", 0)) + 1

    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None:
        self.status["last_gripper_m"] = gripper_m
        self.status["last_gripper_effort_n_m"] = effort_n_m

    def emergency_stop(self) -> str:
        self.status["arm_status"] = 1
        self.status["fault"] = True
        return "Mock emergency stop"

    def resume(self) -> str:
        self.status["arm_status"] = 0
        self.status["fault"] = False
        return "Mock resumed"

    def disable(self) -> str:
        self.status["ctrl_mode"] = 0
        return "Mock disabled"


def _required_number(obj: Any, candidates: list[str]) -> float:
    value = get_first_attr(obj, candidates)
    if value is None:
        raise AttributeError(f"Could not find any of fields {candidates} in {obj!r}")
    return float(value)


def _optional_number(obj: Any, candidates: list[str]) -> Optional[float]:
    value = get_first_attr(obj, candidates)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _err_status_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, (int, float, bool)):
        return int(obj) if isinstance(obj, bool) else obj
    result: Dict[str, Any] = {}
    for name in (
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
        "communication_status_joint_1",
        "communication_status_joint_2",
        "communication_status_joint_3",
        "communication_status_joint_4",
        "communication_status_joint_5",
        "communication_status_joint_6",
    ):
        if hasattr(obj, name):
            result[name] = bool(getattr(obj, name))
    return result or str(obj)


def _err_status_has_fault(obj: Any) -> bool:
    normalized = _err_status_to_dict(obj)
    if normalized is None:
        return False
    if isinstance(normalized, dict):
        return any(bool(value) for value in normalized.values())
    if isinstance(normalized, (int, float, bool)):
        return bool(normalized)
    return str(normalized).strip() not in ("", "0", "None")
