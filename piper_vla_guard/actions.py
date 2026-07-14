from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence


class ActionParseError(ValueError):
    pass


SUPPORTED_ACTION_MODES = [
    "robosuite_osc_pose",
    "delta_base_m_deg",
    "absolute_ee_m_deg",
    "joint_delta_deg",
]


def parse_action_json(text: str, action_mode: str) -> List[List[float]]:
    """Parse manual or OpenPI action JSON into a list of numeric rows."""
    if action_mode not in SUPPORTED_ACTION_MODES:
        raise ActionParseError(f"Unsupported action mode: {action_mode}")
    if text is None or not str(text).strip():
        raise ActionParseError("Action JSON is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON: {exc}") from exc
    rows = _extract_rows(data)
    if not rows:
        raise ActionParseError("No actions found")
    return [_normalize_row(row, action_mode) for row in rows]


def _extract_rows(data: Any) -> List[Any]:
    if isinstance(data, dict):
        for key in ("actions", "action", "action_chunk", "trajectory", "steps"):
            if key in data:
                return _extract_rows(data[key])
        # A single dict action.
        if any(key in data for key in ("dx", "x", "dpos", "j1", "dj1")):
            return [data]
        raise ActionParseError("JSON object must contain actions/action/action_chunk/trajectory/steps")
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(x, (int, float)) for x in data):
            return [data]
        return data
    raise ActionParseError("Actions must be a JSON list or object")


def _normalize_row(row: Any, action_mode: str) -> List[float]:
    if isinstance(row, dict):
        return _normalize_dict_row(row, action_mode)
    if isinstance(row, (list, tuple)):
        vals = [float(x) for x in row]
        return _normalize_list_row(vals, action_mode)
    raise ActionParseError(f"Unsupported action row type: {type(row).__name__}")


def _normalize_list_row(vals: List[float], action_mode: str) -> List[float]:
    if action_mode in ("robosuite_osc_pose", "delta_base_m_deg"):
        if len(vals) == 3:
            return vals + [0.0, 0.0, 0.0, float("nan")]
        if len(vals) == 4:
            return vals[:3] + [0.0, 0.0, 0.0, vals[3]]
        if len(vals) == 6:
            return vals + [float("nan")]
        if len(vals) == 7:
            return vals
        raise ActionParseError(f"{action_mode} rows must have 3, 4, 6, or 7 values")
    if action_mode == "absolute_ee_m_deg":
        if len(vals) == 6:
            return vals + [float("nan")]
        if len(vals) == 7:
            return vals
        raise ActionParseError("absolute_ee_m_deg rows must have 6 or 7 values")
    if action_mode == "joint_delta_deg":
        if len(vals) == 6:
            return vals + [float("nan")]
        if len(vals) == 7:
            return vals
        raise ActionParseError("joint_delta_deg rows must have 6 or 7 values")
    raise ActionParseError(f"Unsupported action mode: {action_mode}")


def _normalize_dict_row(row: Dict[str, Any], action_mode: str) -> List[float]:
    if action_mode in ("robosuite_osc_pose", "delta_base_m_deg"):
        if "dpos" in row:
            dpos = _float_seq(row["dpos"], 3, "dpos")
        else:
            dpos = [float(row.get("dx", 0.0)), float(row.get("dy", 0.0)), float(row.get("dz", 0.0))]
        if "drot" in row:
            drot = _float_seq(row["drot"], 3, "drot")
        else:
            drot = [
                float(row.get("drx", row.get("droll", 0.0))),
                float(row.get("dry", row.get("dpitch", 0.0))),
                float(row.get("drz", row.get("dyaw", 0.0))),
            ]
        g = _get_optional_float(row, ("gripper", "g", "grip"))
        return dpos + drot + [g]

    if action_mode == "absolute_ee_m_deg":
        vals = [
            float(row.get("x")),
            float(row.get("y")),
            float(row.get("z")),
            float(row.get("rx", row.get("roll"))),
            float(row.get("ry", row.get("pitch"))),
            float(row.get("rz", row.get("yaw"))),
        ]
        vals.append(_get_optional_float(row, ("gripper", "g", "grip")))
        return vals

    if action_mode == "joint_delta_deg":
        vals = []
        for i in range(1, 7):
            vals.append(float(row.get(f"dj{i}", row.get(f"djoint_{i}", row.get(f"j{i}", 0.0)))))
        vals.append(_get_optional_float(row, ("gripper", "g", "grip")))
        return vals

    raise ActionParseError(f"Unsupported action mode: {action_mode}")


def _float_seq(value: Any, n: int, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ActionParseError(f"{name} must be a list of length {n}")
    return [float(v) for v in value]


def _get_optional_float(row: Dict[str, Any], names: Iterable[str]) -> float:
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    return float("nan")
