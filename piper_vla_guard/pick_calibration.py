from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraGeometry:
    source_points_px: tuple[tuple[float, float], ...]
    base_points_m: tuple[tuple[float, float], ...]
    output_size: tuple[int, int] = (224, 224)
    fill_rgb: tuple[int, int, int] = (0, 0, 0)
    rotate_deg: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def validate(self) -> None:
        if len(self.source_points_px) != 4 or len(self.base_points_m) != 4:
            raise CalibrationError("camera geometry requires four source and four base points")
        if self.output_size[0] <= 0 or self.output_size[1] <= 0:
            raise CalibrationError("camera output_size must be positive")
        if self.rotate_deg not in (0, 90, 180, 270):
            raise CalibrationError("rotate_deg must be 0, 90, 180, or 270")
        for name, points in (("source_points_px", self.source_points_px), ("base_points_m", self.base_points_m)):
            if not all(np.isfinite(value) for point in points for value in point):
                raise CalibrationError(f"{name} contains non-finite values")
            area = 0.5 * abs(
                sum(
                    points[i][0] * points[(i + 1) % 4][1]
                    - points[(i + 1) % 4][0] * points[i][1]
                    for i in range(4)
                )
            )
            if area <= 1e-9:
                raise CalibrationError(f"{name} polygon is degenerate")

    def canonical_points_px(self) -> np.ndarray:
        width, height = self.output_size
        return np.asarray(
            [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class PickCalibration:
    complete: bool
    overhead: Optional[CameraGeometry]
    wrist: Optional[CameraGeometry]
    table_z_m: float
    table_margin_m: float
    tool_points_m: tuple[tuple[float, float, float], ...]
    finger_center_offset_m: tuple[float, float, float]
    cylinder_diameter_m: float
    cylinder_height_m: float
    ready_joints_deg: tuple[float, float, float, float, float, float]
    ready_tolerance_deg: tuple[float, float, float, float, float, float]
    ready_path_joints_deg: tuple[tuple[float, float, float, float, float, float], ...] = ()
    workspace_floor_corners_m: tuple[tuple[float, float, float], ...] = ()

    def require_complete(self) -> None:
        if not self.complete:
            raise CalibrationError("pick calibration is not marked complete")
        if self.overhead is None:
            raise CalibrationError("complete pick calibration requires an overhead camera geometry")
        if not self.tool_points_m:
            raise CalibrationError("complete pick calibration requires tool_points_m")
        if len(self.workspace_floor_corners_m) != 4:
            raise CalibrationError("complete pick calibration requires four workspace_floor_corners_m")
        numeric = (
            self.table_z_m,
            self.table_margin_m,
            self.cylinder_diameter_m,
            self.cylinder_height_m,
            *self.finger_center_offset_m,
            *self.ready_joints_deg,
            *self.ready_tolerance_deg,
            *(value for point in self.tool_points_m for value in point),
            *(value for point in self.workspace_floor_corners_m for value in point),
        )
        if not all(np.isfinite(value) for value in numeric):
            raise CalibrationError("complete pick calibration contains non-finite values")
        if self.cylinder_diameter_m <= 0 or self.cylinder_height_m <= 0:
            raise CalibrationError("cylinder dimensions must be positive")
        if self.table_margin_m < 0 or any(value <= 0 for value in self.ready_tolerance_deg):
            raise CalibrationError("table margin must be non-negative and ready tolerances positive")

    def require_ready_path(self) -> None:
        path = self.resolved_ready_path_joints_deg()
        values = tuple(value for point in path for value in point)
        if not all(np.isfinite(value) for value in values):
            raise CalibrationError("automatic ready path contains non-finite values")
        final = path[-1]
        if max(abs(a - b) for a, b in zip(final, self.ready_joints_deg)) > 1e-6:
            raise CalibrationError("the final ready_path_joints_deg waypoint must equal ready_joints_deg")

    def resolved_ready_path_joints_deg(
        self,
    ) -> tuple[tuple[float, float, float, float, float, float], ...]:
        """Use a direct current-to-ready route when no corridor waypoints are configured."""
        return self.ready_path_joints_deg or (self.ready_joints_deg,)


def load_pick_calibration(path: str | Path | None) -> Optional[PickCalibration]:
    if path is None or not str(path).strip():
        return None
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise CalibrationError("calibration YAML root must be a mapping")
    calibration = PickCalibration(
        complete=bool(data.get("complete", False)),
        overhead=_camera_geometry(data.get("overhead"), "overhead"),
        wrist=_camera_geometry(data.get("wrist"), "wrist"),
        table_z_m=float(data.get("table_z_m", 0.0)),
        table_margin_m=float(data.get("table_margin_m", 0.005)),
        tool_points_m=tuple(_tuple(point, 3, f"tool_points_m[{i}]") for i, point in enumerate(data.get("tool_points_m", []))),
        finger_center_offset_m=_tuple(
            data.get("finger_center_offset_m", [0.0, 0.0, 0.0]), 3, "finger_center_offset_m"
        ),
        cylinder_diameter_m=float(data.get("cylinder_diameter_m", 0.030)),
        cylinder_height_m=float(data.get("cylinder_height_m", 0.030)),
        ready_joints_deg=_tuple(data.get("ready_joints_deg", [0, 100.73, -64.93, 0, 58.89, 0]), 6, "ready_joints_deg"),
        ready_tolerance_deg=_tuple(data.get("ready_tolerance_deg", [5, 8, 8, 5, 10, 5]), 6, "ready_tolerance_deg"),
        ready_path_joints_deg=tuple(
            _tuple(point, 6, f"ready_path_joints_deg[{i}]")
            for i, point in enumerate(data.get("ready_path_joints_deg", []))
        ),
        workspace_floor_corners_m=tuple(
            _tuple(point, 3, f"workspace_floor_corners_m[{i}]")
            for i, point in enumerate(data.get("workspace_floor_corners_m", []))
        ),
    )
    if calibration.overhead:
        calibration.overhead.validate()
    if calibration.wrist:
        calibration.wrist.validate()
    return calibration


def apply_calibration_to_safety(calibration: PickCalibration, cfg: Any) -> None:
    cfg.calibration_complete = calibration.complete
    cfg.table_z_m = calibration.table_z_m
    cfg.table_margin_m = calibration.table_margin_m
    cfg.tool_points_m = calibration.tool_points_m
    cfg.cylinder_diameter_m = calibration.cylinder_diameter_m
    cfg.cylinder_height_m = calibration.cylinder_height_m
    cfg.expected_ready_joints_deg = calibration.ready_joints_deg
    cfg.ready_joint_tolerance_deg = calibration.ready_tolerance_deg
    cfg.workspace_floor_corners_m = calibration.workspace_floor_corners_m
    if calibration.workspace_floor_corners_m:
        # The calibrated perimeter/floor supersedes the legacy rectangular
        # workspace and the example config's fixed horizontal table plane.
        # Keeping those defaults would silently reject valid calibrated poses.
        xs = [point[0] for point in calibration.workspace_floor_corners_m]
        ys = [point[1] for point in calibration.workspace_floor_corners_m]
        zs = [point[2] for point in calibration.workspace_floor_corners_m]
        cfg.workspace_x_m = (min(xs), max(xs))
        cfg.workspace_y_m = (min(ys), max(ys))

        # Use the stricter configured/calibrated clearance. The local fitted
        # floor check still supplies the exact height at each XY location.
        cfg.workspace_floor_margin_m = max(
            float(cfg.workspace_floor_margin_m), calibration.table_margin_m
        )
        floor_lower_z = min(zs) + cfg.workspace_floor_margin_m
        upper_z = float(cfg.workspace_z_m[1])
        if floor_lower_z >= upper_z:
            raise CalibrationError(
                "calibrated floor plus margin is not below workspace_z_m upper bound: "
                f"{floor_lower_z:.6f} >= {upper_z:.6f}"
            )
        cfg.workspace_z_m = (floor_lower_z, upper_z)
        cfg.min_z_m = floor_lower_z

        # The calibrated four-corner floor is authoritative (horizontal when
        # all Z values match). Remove only the duplicate legacy table plane;
        # preserve explicitly configured fixture/keep-out planes.
        cfg.safety_planes = tuple(
            plane for plane in cfg.safety_planes if plane.name != "table_clearance"
        )


def preprocess_camera_image(image: np.ndarray, geometry: Optional[CameraGeometry]) -> np.ndarray:
    arr = np.asarray(image)
    if geometry is None:
        return arr
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise CalibrationError("opencv-python is required for calibrated camera preprocessing") from exc

    if geometry.rotate_deg:
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        arr = cv2.rotate(arr, rotations[geometry.rotate_deg])
    if geometry.flip_horizontal:
        arr = cv2.flip(arr, 1)
    if geometry.flip_vertical:
        arr = cv2.flip(arr, 0)

    transform = cv2.getPerspectiveTransform(
        np.asarray(geometry.source_points_px, dtype=np.float32), geometry.canonical_points_px()
    )
    width, height = geometry.output_size
    return cv2.warpPerspective(
        arr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=geometry.fill_rgb,
    )


def canonical_pixel_to_base(pixel_xy: Sequence[float], geometry: CameraGeometry) -> tuple[float, float]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise CalibrationError("opencv-python is required for camera-to-base mapping") from exc
    transform = cv2.getPerspectiveTransform(
        geometry.canonical_points_px(), np.asarray(geometry.base_points_m, dtype=np.float32)
    )
    point = np.asarray([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, transform)[0, 0]
    return float(mapped[0]), float(mapped[1])


def _camera_geometry(value: Any, name: str) -> Optional[CameraGeometry]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CalibrationError(f"{name} must be a mapping")
    return CameraGeometry(
        source_points_px=tuple(_tuple(p, 2, f"{name}.source_points_px") for p in value.get("source_points_px", [])),
        base_points_m=tuple(_tuple(p, 2, f"{name}.base_points_m") for p in value.get("base_points_m", [])),
        output_size=tuple(int(v) for v in _tuple(value.get("output_size", [224, 224]), 2, f"{name}.output_size")),  # type: ignore[arg-type]
        fill_rgb=tuple(int(v) for v in _tuple(value.get("fill_rgb", [0, 0, 0]), 3, f"{name}.fill_rgb")),  # type: ignore[arg-type]
        rotate_deg=int(value.get("rotate_deg", 0)),
        flip_horizontal=bool(value.get("flip_horizontal", False)),
        flip_vertical=bool(value.get("flip_vertical", False)),
    )


def _tuple(value: Any, length: int, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CalibrationError(f"{name} must contain {length} values")
    return tuple(float(v) for v in value)
