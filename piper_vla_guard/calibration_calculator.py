from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from .safety import _rpy_rotation_matrix
from .types import EEPose


class SampleError(RuntimeError):
    pass


def calculate_calibration_samples(payload: dict[str, Any]) -> dict[str, Any]:
    probe_offset = _vector(payload.get("probe_offset_m"), 3, "probe_offset_m")
    floor_raw = payload.get("floor_samples")
    if not isinstance(floor_raw, list) or len(floor_raw) != 4:
        raise SampleError("floor_samples must contain four perimeter-ordered samples")

    floor_points: list[list[float]] = []
    pixels: list[list[float]] = []
    names: list[str] = []
    for index, sample in enumerate(floor_raw):
        if not isinstance(sample, dict):
            raise SampleError(f"floor_samples[{index}] must be a mapping")
        pose = _pose(sample.get("ee_pose_m_deg"), f"floor_samples[{index}].ee_pose_m_deg")
        point = _world_point(pose, probe_offset)
        floor_points.append(point.tolist())
        pixels.append(list(_vector(sample.get("pixel_xy"), 2, f"floor_samples[{index}].pixel_xy")))
        names.append(str(sample.get("name", index)))

    design = np.asarray([[x, y, 1.0] for x, y, _ in floor_points], dtype=np.float64)
    heights = np.asarray([z for _, _, z in floor_points], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, heights, rcond=None)
    floor_fit_error = float(np.max(np.abs(design @ coefficients - heights)))
    area_twice = sum(
        floor_points[i][0] * floor_points[(i + 1) % 4][1]
        - floor_points[(i + 1) % 4][0] * floor_points[i][1]
        for i in range(4)
    )
    if abs(area_twice) <= 1e-9:
        raise SampleError("floor sample XY polygon is degenerate or not perimeter ordered")

    output: dict[str, Any] = {
        "workspace_floor_corners_m": floor_points,
        "overhead": {
            "source_points_px": pixels,
            "base_points_m": [[point[0], point[1]] for point in floor_points],
        },
        "diagnostics": {
            "floor_sample_names": names,
            "floor_plane_z_ax_by_c": coefficients.tolist(),
            "floor_max_fit_error_m": floor_fit_error,
            "floor_polygon_winding": "counter_clockwise" if area_twice > 0 else "clockwise",
        },
    }

    finger_raw = payload.get("finger_center_samples", [])
    if finger_raw:
        if not isinstance(finger_raw, list) or len(finger_raw) < 3:
            raise SampleError("finger_center_samples must contain at least three samples")
        offsets = []
        for index, sample in enumerate(finger_raw):
            if not isinstance(sample, dict):
                raise SampleError(f"finger_center_samples[{index}] must be a mapping")
            pose = _pose(
                sample.get("ee_pose_m_deg"),
                f"finger_center_samples[{index}].ee_pose_m_deg",
            )
            target = np.asarray(
                _vector(sample.get("target_point_m"), 3, f"finger_center_samples[{index}].target_point_m")
            )
            rotation = np.asarray(_rpy_rotation_matrix(*pose.rpy()), dtype=np.float64)
            offsets.append(rotation.T @ (target - np.asarray(pose.xyz(), dtype=np.float64)))
        offset_values = np.asarray(offsets)
        mean_offset = offset_values.mean(axis=0)
        max_residual = float(np.max(np.linalg.norm(offset_values - mean_offset, axis=1)))
        output["finger_center_offset_m"] = mean_offset.tolist()
        output["diagnostics"]["finger_center_max_residual_m"] = max_residual

    bounds = payload.get("tool_bounds_m")
    if bounds is not None:
        if not isinstance(bounds, dict):
            raise SampleError("tool_bounds_m must be a mapping")
        axes = [_vector(bounds.get(axis), 2, f"tool_bounds_m.{axis}") for axis in "xyz"]
        for axis, values in zip("xyz", axes):
            if values[0] >= values[1]:
                raise SampleError(f"tool_bounds_m.{axis} must be ordered [min, max]")
        output["tool_points_m"] = [list(point) for point in itertools.product(*axes)]

    return output


def _world_point(pose: EEPose, local_offset: Sequence[float]) -> np.ndarray:
    rotation = np.asarray(_rpy_rotation_matrix(*pose.rpy()), dtype=np.float64)
    return np.asarray(pose.xyz(), dtype=np.float64) + rotation @ np.asarray(local_offset, dtype=np.float64)


def _pose(value: Any, name: str) -> EEPose:
    values = _vector(value, 6, name)
    return EEPose(*values)


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SampleError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SampleError(f"{name} contains non-finite values")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Piper pick-calibration YAML fragments from read-only measurements."
    )
    parser.add_argument("--samples", required=True, help="Measurement YAML; see configs/calibration_samples.example.yaml")
    parser.add_argument("--output", default="", help="Optional output YAML. Default prints to stdout.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(Path(args.samples).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SampleError("sample YAML root must be a mapping")
    output = yaml.safe_dump(calculate_calibration_samples(payload), sort_keys=False, allow_unicode=True)
    if args.output:
        Path(args.output).expanduser().write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
