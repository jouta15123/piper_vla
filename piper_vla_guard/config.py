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
        "judge_flag": bool,
        "can_auto_init": bool,
        "dh_is_offset": int,
        "start_sdk_joint_limit": bool,
        "start_sdk_gripper_limit": bool,
        "dry_run": bool,
        "require_manual_approval": bool,
        "speed_pct": int,
        "step_sleep_s": float,
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
    if "max_step_rpy_deg" in data:
        cfg.max_step_rpy_deg = _tuple3(data["max_step_rpy_deg"], "max_step_rpy_deg")
    if "max_joint_step_deg" in data:
        cfg.max_joint_step_deg = _tuple6(data["max_joint_step_deg"], "max_joint_step_deg")

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
    _finite_nonnegative(cfg.max_start_pose_drift_m, "max_start_pose_drift_m")
    _finite_nonnegative(cfg.max_start_rpy_drift_deg, "max_start_rpy_drift_deg")
    _finite_nonnegative(cfg.max_start_joint_drift_deg, "max_start_joint_drift_deg")
    _finite_nonnegative(cfg.max_total_translation_m, "max_total_translation_m")
    _finite_nonnegative(cfg.robosuite_osc_xyz_scale_m, "robosuite_osc_xyz_scale_m")
    _finite_nonnegative(cfg.robosuite_osc_rot_scale_rad, "robosuite_osc_rot_scale_rad")
    _finite_nonnegative(cfg.robosuite_gripper_qpos_max_m, "robosuite_gripper_qpos_max_m")
    _finite_nonnegative(cfg.robosuite_gripper_min_width_m, "robosuite_gripper_min_width_m")
    _finite_positive(cfg.max_horizon, "max_horizon")
    if cfg.robosuite_gripper_open_action == cfg.robosuite_gripper_close_action:
        raise ValueError("robosuite gripper open/close actions must differ")
    if cfg.robosuite_gripper_min_width_m > cfg.gripper_open_m:
        raise ValueError("robosuite_gripper_min_width_m must be <= gripper_open_m")

    if not 0 <= cfg.speed_pct <= 100:
        raise ValueError("speed_pct must be in [0, 100]")
    if not 0.0 <= cfg.gripper_effort_n_m <= 5.0:
        raise ValueError("gripper_effort_n_m must be in [0, 5] N/m")

    for name, bounds in (
        ("workspace_m.x", cfg.workspace_x_m),
        ("workspace_m.y", cfg.workspace_y_m),
        ("workspace_m.z", cfg.workspace_z_m),
    ):
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
