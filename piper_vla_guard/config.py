from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .types import SafetyConfig, SafetyPlane


def _tuple2(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-element list")
    return (float(value[0]), float(value[1]))


def _tuple3(value: Any, name: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element list")
    return (float(value[0]), float(value[1]), float(value[2]))


def _tuple6(value: Any, name: str) -> Tuple[float, float, float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{name} must be a 6-element list")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _plane_from_dict(value: Any, name: str) -> SafetyPlane:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if "normal" not in value:
        raise ValueError(f"{name}.normal is required")
    if "point" not in value:
        raise ValueError(f"{name}.point is required")
    return SafetyPlane(
        name=str(value.get("name", name)),
        normal=_tuple3(value["normal"], f"{name}.normal"),
        point=_tuple3(value["point"], f"{name}.point"),
        margin_m=float(value.get("margin_m", 0.0)),
    )


def load_config(path: str | Path | None) -> SafetyConfig:
    cfg = SafetyConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping")
    return config_from_dict(data, cfg)


def config_from_dict(data: Dict[str, Any], base: SafetyConfig | None = None) -> SafetyConfig:
    cfg = base or SafetyConfig()

    simple_fields = {
        "can_name": str,
        "rosbridge_host": str,
        "rosbridge_port": int,
        "ros_command_topic": str,
        "ros_heartbeat_topic": str,
        "ros_joint_topic": str,
        "ros_pose_topic": str,
        "ros_status_topic": str,
        "ros_arm_service": str,
        "judge_flag": bool,
        "can_auto_init": bool,
        "dh_is_offset": int,
        "start_sdk_joint_limit": bool,
        "start_sdk_gripper_limit": bool,
        "dry_run": bool,
        "require_manual_approval": bool,
        "speed_pct": int,
        "step_sleep_s": float,
        "control_hz": float,
        "require_status_available": bool,
        "max_start_pose_drift_m": float,
        "max_start_rpy_drift_deg": float,
        "max_start_joint_drift_deg": float,
        "action_scale_xyz": float,
        "action_scale_rpy": float,
        "robosuite_osc_xyz_scale_m": float,
        "robosuite_osc_rot_scale_rad": float,
        "robosuite_gripper_open_action": float,
        "robosuite_gripper_close_action": float,
        "robosuite_gripper_qpos_max_m": float,
        "robosuite_gripper_min_width_m": float,
        "reject_on_clip": bool,
        "reject_on_warning": bool,
        "min_z_m": float,
        "max_total_translation_m": float,
        "max_horizon": int,
        "free_space_speed_m_s": float,
        "near_table_speed_m_s": float,
        "max_cartesian_accel_m_s2": float,
        "max_rpy_speed_deg_s": float,
        "max_rpy_accel_deg_s2": float,
        "max_measured_joint_speed_deg_s": float,
        "max_commanded_joint_speed_deg_s": float,
        "joint_command_hz": float,
        "near_table_distance_m": float,
        "tracking_hold_error_m": float,
        "tracking_abort_error_m": float,
        "tracking_settle_timeout_s": float,
        "settle_each_vla_step": bool,
        "cartesian_execution_mode": str,
        "ik_position_tolerance_m": float,
        "ik_rotation_tolerance_deg": float,
        "ik_max_iterations": int,
        "ik_damping": float,
        "ik_jacobian_delta_rad": float,
        "ik_max_update_deg": float,
        "ik_path_sample_step_deg": float,
        "calibration_complete": bool,
        "table_z_m": float,
        "table_margin_m": float,
        "cylinder_diameter_m": float,
        "cylinder_height_m": float,
        "grasp_width_margin_m": float,
        "hybrid_test_lift_m": float,
        "hybrid_total_lift_m": float,
        "ready_return_speed_pct": int,
        "ready_return_max_step_deg": float,
        "ready_return_tracking_tolerance_deg": float,
        "ready_return_step_timeout_s": float,
        "ready_return_total_timeout_s": float,
        "ready_return_max_joint_speed_deg_s": float,
        "ready_return_workspace_tolerance_m": float,
        "ready_return_enforce_workspace_floor_polygon": bool,
        "workspace_floor_margin_m": float,
        "workspace_floor_max_fit_error_m": float,
        "joint_limit_margin_deg": float,
        "gripper_open_m": float,
        "gripper_closed_m": float,
        "gripper_effort_n_m": float,
        "log_dir": str,
    }
    for key, caster in simple_fields.items():
        if key in data:
            setattr(cfg, key, caster(data[key]))

    workspace = data.get("workspace_m")
    if workspace is not None:
        if not isinstance(workspace, dict):
            raise ValueError("workspace_m must be a mapping")
        if "x" in workspace:
            cfg.workspace_x_m = _tuple2(workspace["x"], "workspace_m.x")
        if "y" in workspace:
            cfg.workspace_y_m = _tuple2(workspace["y"], "workspace_m.y")
        if "z" in workspace:
            cfg.workspace_z_m = _tuple2(workspace["z"], "workspace_m.z")

    if "max_step_xyz_m" in data:
        cfg.max_step_xyz_m = _tuple3(data["max_step_xyz_m"], "max_step_xyz_m")
    if "ready_return_workspace_m" in data:
        ready_workspace = data["ready_return_workspace_m"]
        if not isinstance(ready_workspace, dict):
            raise ValueError("ready_return_workspace_m must be a mapping")
        if "x" in ready_workspace:
            cfg.ready_return_workspace_x_m = _tuple2(
                ready_workspace["x"], "ready_return_workspace_m.x"
            )
        if "y" in ready_workspace:
            cfg.ready_return_workspace_y_m = _tuple2(
                ready_workspace["y"], "ready_return_workspace_m.y"
            )
        if "z" in ready_workspace:
            cfg.ready_return_workspace_z_m = _tuple2(
                ready_workspace["z"], "ready_return_workspace_m.z"
            )
    if "max_step_rpy_deg" in data:
        cfg.max_step_rpy_deg = _tuple3(data["max_step_rpy_deg"], "max_step_rpy_deg")
    if "max_joint_step_deg" in data:
        cfg.max_joint_step_deg = _tuple6(data["max_joint_step_deg"], "max_joint_step_deg")
    if "expected_ready_joints_deg" in data:
        cfg.expected_ready_joints_deg = _tuple6(data["expected_ready_joints_deg"], "expected_ready_joints_deg")
    if "ready_joint_tolerance_deg" in data:
        cfg.ready_joint_tolerance_deg = _tuple6(
            data["ready_joint_tolerance_deg"], "ready_joint_tolerance_deg"
        )
    if "shutdown_joints_deg" in data:
        cfg.shutdown_joints_deg = (
            None
            if data["shutdown_joints_deg"] is None
            else _tuple6(data["shutdown_joints_deg"], "shutdown_joints_deg")
        )
    if "shutdown_joint_tolerance_deg" in data:
        cfg.shutdown_joint_tolerance_deg = _tuple6(
            data["shutdown_joint_tolerance_deg"], "shutdown_joint_tolerance_deg"
        )
    if "tool_points_m" in data:
        raw_points = data["tool_points_m"]
        if not isinstance(raw_points, list):
            raise ValueError("tool_points_m must be a list")
        cfg.tool_points_m = tuple(
            _tuple3(point, f"tool_points_m[{idx}]") for idx, point in enumerate(raw_points)
        )
    if "workspace_floor_corners_m" in data:
        raw_corners = data["workspace_floor_corners_m"]
        if raw_corners is None:
            cfg.workspace_floor_corners_m = ()
        elif isinstance(raw_corners, list):
            cfg.workspace_floor_corners_m = tuple(
                _tuple3(point, f"workspace_floor_corners_m[{idx}]")
                for idx, point in enumerate(raw_corners)
            )
        else:
            raise ValueError("workspace_floor_corners_m must be a list")

    if "joint_limits_deg" in data:
        raw = data["joint_limits_deg"]
        if not isinstance(raw, dict):
            raise ValueError("joint_limits_deg must be a mapping")
        cfg.joint_limits_deg = {k: _tuple2(v, f"joint_limits_deg.{k}") for k, v in raw.items()}

    if "safety_planes" in data:
        raw_planes = data["safety_planes"]
        if raw_planes is None:
            cfg.safety_planes = ()
        elif isinstance(raw_planes, list):
            cfg.safety_planes = tuple(
                _plane_from_dict(item, f"safety_planes[{idx}]")
                for idx, item in enumerate(raw_planes)
            )
        else:
            raise ValueError("safety_planes must be a list")

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: SafetyConfig) -> None:
    _finite_nonnegative(cfg.step_sleep_s, "step_sleep_s")
    _finite_positive(cfg.control_hz, "control_hz")
    _finite_nonnegative(cfg.max_start_pose_drift_m, "max_start_pose_drift_m")
    _finite_nonnegative(cfg.max_start_rpy_drift_deg, "max_start_rpy_drift_deg")
    _finite_nonnegative(cfg.max_start_joint_drift_deg, "max_start_joint_drift_deg")
    _finite_nonnegative(cfg.max_total_translation_m, "max_total_translation_m")
    _finite_nonnegative(cfg.robosuite_osc_xyz_scale_m, "robosuite_osc_xyz_scale_m")
    _finite_nonnegative(cfg.robosuite_osc_rot_scale_rad, "robosuite_osc_rot_scale_rad")
    _finite_nonnegative(cfg.robosuite_gripper_qpos_max_m, "robosuite_gripper_qpos_max_m")
    _finite_nonnegative(cfg.robosuite_gripper_min_width_m, "robosuite_gripper_min_width_m")
    _finite_positive(cfg.max_horizon, "max_horizon")
    for name in (
        "free_space_speed_m_s",
        "near_table_speed_m_s",
        "max_cartesian_accel_m_s2",
        "max_rpy_speed_deg_s",
        "max_rpy_accel_deg_s2",
        "max_measured_joint_speed_deg_s",
        "max_commanded_joint_speed_deg_s",
        "joint_command_hz",
        "tracking_abort_error_m",
        "ik_position_tolerance_m",
        "ik_rotation_tolerance_deg",
        "ik_max_iterations",
        "ik_damping",
        "ik_jacobian_delta_rad",
        "ik_max_update_deg",
        "ik_path_sample_step_deg",
        "ready_return_max_step_deg",
        "ready_return_tracking_tolerance_deg",
        "ready_return_step_timeout_s",
        "ready_return_total_timeout_s",
        "ready_return_max_joint_speed_deg_s",
    ):
        _finite_positive(getattr(cfg, name), name)
    for name in (
        "near_table_distance_m",
        "tracking_hold_error_m",
        "tracking_settle_timeout_s",
        "table_margin_m",
        "cylinder_diameter_m",
        "cylinder_height_m",
        "grasp_width_margin_m",
        "hybrid_test_lift_m",
        "hybrid_total_lift_m",
        "ready_return_workspace_tolerance_m",
        "workspace_floor_margin_m",
        "workspace_floor_max_fit_error_m",
    ):
        _finite_nonnegative(getattr(cfg, name), name)
    if cfg.tracking_hold_error_m > cfg.tracking_abort_error_m:
        raise ValueError("tracking_hold_error_m must be <= tracking_abort_error_m")
    if cfg.joint_command_hz < cfg.control_hz:
        raise ValueError("joint_command_hz must be >= control_hz")
    if cfg.max_commanded_joint_speed_deg_s > cfg.max_measured_joint_speed_deg_s:
        raise ValueError(
            "max_commanded_joint_speed_deg_s must be <= max_measured_joint_speed_deg_s"
        )
    if cfg.cartesian_execution_mode not in ("joint_ik", "end_pose"):
        raise ValueError("cartesian_execution_mode must be 'joint_ik' or 'end_pose'")
    if cfg.grasp_width_margin_m >= cfg.cylinder_diameter_m:
        raise ValueError("grasp_width_margin_m must be smaller than cylinder_diameter_m")
    if cfg.hybrid_test_lift_m > cfg.hybrid_total_lift_m:
        raise ValueError("hybrid_test_lift_m must be <= hybrid_total_lift_m")
    if cfg.robosuite_gripper_open_action == cfg.robosuite_gripper_close_action:
        raise ValueError("robosuite gripper open/close actions must differ")
    if cfg.robosuite_gripper_min_width_m > cfg.gripper_open_m:
        raise ValueError("robosuite_gripper_min_width_m must be <= gripper_open_m")

    if not 0 <= cfg.speed_pct <= 100:
        raise ValueError("speed_pct must be in [0, 100]")
    if not 1 <= cfg.ready_return_speed_pct <= 100:
        raise ValueError("ready_return_speed_pct must be in [1, 100]")
    if not 0.0 <= cfg.gripper_effort_n_m <= 5.0:
        raise ValueError("gripper_effort_n_m must be in [0, 5] N/m")

    for name, bounds in (
        ("workspace_m.x", cfg.workspace_x_m),
        ("workspace_m.y", cfg.workspace_y_m),
        ("workspace_m.z", cfg.workspace_z_m),
    ):
        _ordered_pair(bounds, name)
    for name, bounds in (
        ("ready_return_workspace_m.x", cfg.ready_return_workspace_x_m),
        ("ready_return_workspace_m.y", cfg.ready_return_workspace_y_m),
        ("ready_return_workspace_m.z", cfg.ready_return_workspace_z_m),
    ):
        if bounds is not None:
            _ordered_pair(bounds, name)
    for name, bounds in cfg.joint_limits_deg.items():
        _ordered_pair(bounds, f"joint_limits_deg.{name}")

    for name, values in (
        ("max_step_xyz_m", cfg.max_step_xyz_m),
        ("max_step_rpy_deg", cfg.max_step_rpy_deg),
        ("max_joint_step_deg", cfg.max_joint_step_deg),
    ):
        for value in values:
            _finite_nonnegative(value, name)

    for plane in cfg.safety_planes:
        if not plane.name:
            raise ValueError("safety plane name must not be empty")
        for value in plane.normal + plane.point:
            if not math.isfinite(value):
                raise ValueError(f"safety plane {plane.name} values must be finite")
        _finite_nonnegative(plane.margin_m, f"safety plane {plane.name} margin_m")
        if math.sqrt(sum(v * v for v in plane.normal)) <= 1e-12:
            raise ValueError(f"safety plane {plane.name} normal must be non-zero")
    for index, point in enumerate(cfg.tool_points_m):
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"tool_points_m[{index}] values must be finite")
    if cfg.workspace_floor_corners_m:
        if len(cfg.workspace_floor_corners_m) != 4:
            raise ValueError("workspace_floor_corners_m must contain exactly four XYZ corners")
        if not all(math.isfinite(value) for point in cfg.workspace_floor_corners_m for value in point):
            raise ValueError("workspace_floor_corners_m values must be finite")
        area_twice = sum(
            cfg.workspace_floor_corners_m[i][0] * cfg.workspace_floor_corners_m[(i + 1) % 4][1]
            - cfg.workspace_floor_corners_m[(i + 1) % 4][0] * cfg.workspace_floor_corners_m[i][1]
            for i in range(4)
        )
        if abs(area_twice) <= 1e-9:
            raise ValueError("workspace_floor_corners_m XY polygon is degenerate")


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _ordered_pair(bounds: Tuple[float, float], name: str) -> None:
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"{name} must be finite and ordered as [low, high]")
