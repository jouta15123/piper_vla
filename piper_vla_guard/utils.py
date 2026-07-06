from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_abs(value: float, limit: float) -> float:
    return clamp(value, -abs(limit), abs(limit))


def m_to_sdk_pos(value_m: float) -> int:
    """Meters to Piper SDK position units: 0.001 mm."""
    return int(round(value_m * 1_000_000.0))


def sdk_pos_to_m(value_sdk: float) -> float:
    """Piper SDK position units 0.001 mm to meters."""
    return float(value_sdk) / 1_000_000.0


def deg_to_sdk_angle(value_deg: float) -> int:
    """Degrees to Piper SDK angle units: 0.001 deg."""
    return int(round(value_deg * 1000.0))


def sdk_angle_to_deg(value_sdk: float) -> float:
    """Piper SDK angle units 0.001 deg to degrees."""
    return float(value_sdk) / 1000.0


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def get_first_attr(obj: Any, candidates: Iterable[str]) -> Optional[Any]:
    """Return the first attribute or dict key found from candidates."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in candidates:
            if name in obj:
                return obj[name]
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def unwrap_message(obj: Any, field_candidates: Iterable[str]) -> Any:
    """Unwrap common piper_sdk message containers.

    The SDK docs describe returns as objects that contain fields such as
    end_pose, joint_state, gripper_state, and arm_status. Some versions can
    expose tuple-like objects. This helper keeps the adapter resilient.
    """
    inner = get_first_attr(obj, field_candidates)
    if inner is not None:
        return inner
    if isinstance(obj, (list, tuple)) and obj:
        # Often: (time_stamp, Hz, payload). Prefer the last item.
        return obj[-1]
    return obj


def is_close_list(a: Sequence[float], b: Sequence[float], tol: float = 1e-12) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
