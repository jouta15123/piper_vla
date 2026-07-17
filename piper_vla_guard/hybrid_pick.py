from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .executor import PlanExecutor
from .logging_utils import JsonlLogger
from .pick_calibration import CameraGeometry, PickCalibration, canonical_pixel_to_base
from .piper_adapter import RobotAdapter
from .safety import SafetyChecker, _rpy_rotation_matrix
from .types import EEPose, SafetyConfig


class HybridPickError(RuntimeError):
    pass


class HybridPhase(str, enum.Enum):
    VLA_APPROACH = "VLA_APPROACH"
    VISION_ALIGN = "VISION_ALIGN"
    DESCEND = "DESCEND"
    WAIT_APPROVAL = "WAIT_APPROVAL"
    CLOSE_RAMP = "CLOSE_RAMP"
    TEST_LIFT = "TEST_LIFT"
    LIFT = "LIFT"
    HOLD = "HOLD"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CylinderDetection:
    pixel_xy: tuple[float, float]
    base_xy_m: tuple[float, float]
    diameter_m: float
    area_px: float
    circularity: float


@dataclass(frozen=True)
class HybridProposal:
    phase: HybridPhase
    detection: CylinderDetection
    xy_error_m: float
    action: Optional[list[float]]


class HybridPickController:
    def __init__(
        self,
        cfg: SafetyConfig,
        calibration: PickCalibration,
        *,
        coarse_xy_m: float = 0.030,
        align_xy_m: float = 0.005,
        descend_z_m: float = 0.003,
    ) -> None:
        calibration.require_complete()
        self.cfg = cfg
        self.calibration = calibration
        self.phase = HybridPhase.VLA_APPROACH
        self.coarse_xy_m = float(coarse_xy_m)
        self.align_xy_m = float(align_xy_m)
        self.descend_z_m = float(descend_z_m)
        self.last_detection: Optional[CylinderDetection] = None

    def propose(self, pose: EEPose, overhead_image: np.ndarray) -> HybridProposal:
        geometry = self.calibration.overhead
        if geometry is None:
            raise HybridPickError("hybrid mode requires overhead calibration")
        detection = detect_white_cylinder(
            overhead_image,
            geometry,
            expected_diameter_m=self.calibration.cylinder_diameter_m,
        )
        self.last_detection = detection
        finger = self._finger_center(pose)
        dx = detection.base_xy_m[0] - finger[0]
        dy = detection.base_xy_m[1] - finger[1]
        xy_error = math.hypot(dx, dy)

        if self.phase == HybridPhase.VLA_APPROACH and xy_error <= self.coarse_xy_m:
            self.phase = HybridPhase.VISION_ALIGN
        if self.phase == HybridPhase.VISION_ALIGN:
            if xy_error <= self.align_xy_m:
                self.phase = HybridPhase.DESCEND
            else:
                return HybridProposal(self.phase, detection, xy_error, [dx, dy, 0.0, float("nan")])
        if self.phase == HybridPhase.DESCEND:
            # The finger centre, not the SDK EE origin, must reach cylinder mid-height.
            target_finger_z = self.calibration.table_z_m + self.calibration.cylinder_height_m / 2.0
            dz = target_finger_z - finger[2]
            if xy_error > self.align_xy_m:
                self.phase = HybridPhase.VISION_ALIGN
                return HybridProposal(self.phase, detection, xy_error, [dx, dy, 0.0, float("nan")])
            if abs(dz) <= self.descend_z_m:
                self.phase = HybridPhase.WAIT_APPROVAL
                return HybridProposal(self.phase, detection, xy_error, None)
            return HybridProposal(self.phase, detection, xy_error, [0.0, 0.0, dz, float("nan")])
        return HybridProposal(self.phase, detection, xy_error, None)

    def run_approved_grasp(
        self,
        robot: RobotAdapter,
        executor: PlanExecutor,
        checker: SafetyChecker,
        logger: JsonlLogger,
        *,
        dry_run: bool,
        approved: bool,
        verify_lift: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        if self.phase != HybridPhase.WAIT_APPROVAL:
            raise HybridPickError(f"grasp requested in phase {self.phase}")
        if not approved:
            raise HybridPickError("grasp close was not manually approved")
        target_width = max(
            self.cfg.gripper_closed_m,
            self.calibration.cylinder_diameter_m - self.cfg.grasp_width_margin_m,
        )
        payload: dict[str, Any] = {
            "phase": self.phase.value,
            "target_width_m": target_width,
            "test_lift_m": self.cfg.hybrid_test_lift_m,
            "full_lift_m": self.cfg.hybrid_total_lift_m,
            "dry_run": dry_run,
        }
        logger.write("hybrid_grasp_approved", payload)
        if dry_run:
            return {"ok": True, "dry_run": True, **payload}

        self.phase = HybridPhase.CLOSE_RAMP
        self._ramp_gripper(robot, target_width)
        time.sleep(1.0)
        gripper = robot.read_gripper_state()
        minimum_contact_width = self.calibration.cylinder_diameter_m - 0.002
        if gripper is None or gripper.opening_m < minimum_contact_width:
            self.phase = HybridPhase.FAILED
            opening = None if gripper is None else gripper.opening_m
            logger.write("hybrid_grasp_failed", {"reason": "empty_grasp", "opening_m": opening})
            raise HybridPickError(
                f"empty grasp suspected: opening={opening}, expected >= {minimum_contact_width:.4f} m"
            )

        self.phase = HybridPhase.TEST_LIFT
        self._execute_lift(robot, checker, executor, self.cfg.hybrid_test_lift_m)
        after_test = robot.read_gripper_state()
        if after_test is None or after_test.opening_m < minimum_contact_width:
            self.phase = HybridPhase.FAILED
            logger.write("hybrid_grasp_failed", {"reason": "lost_on_test_lift"})
            raise HybridPickError("gripper opening indicates the cylinder was lost during test lift")
        if verify_lift is not None and not verify_lift():
            self.phase = HybridPhase.FAILED
            logger.write("hybrid_grasp_failed", {"reason": "wrist_visual_verification_failed"})
            raise HybridPickError("wrist image did not verify the white cylinder after test lift")

        self.phase = HybridPhase.LIFT
        self._execute_lift(
            robot,
            checker,
            executor,
            self.cfg.hybrid_total_lift_m - self.cfg.hybrid_test_lift_m,
        )
        time.sleep(2.0)
        self.phase = HybridPhase.HOLD
        logger.write("hybrid_pick_complete", {"lift_m": self.cfg.hybrid_total_lift_m, "hold_s": 2.0})
        return {"ok": True, "phase": self.phase.value, "lift_m": self.cfg.hybrid_total_lift_m}

    def _finger_center(self, pose: EEPose) -> tuple[float, float, float]:
        rot = _rpy_rotation_matrix(pose.rx, pose.ry, pose.rz)
        offset = self.calibration.finger_center_offset_m
        return tuple(
            pose.xyz()[row] + sum(rot[row][col] * offset[col] for col in range(3)) for row in range(3)
        )  # type: ignore[return-value]

    def _ramp_gripper(self, robot: RobotAdapter, target_width: float) -> None:
        state = robot.read_gripper_state()
        if state is None:
            raise HybridPickError("gripper feedback is required for close ramp")
        width = state.opening_m
        max_step = 0.015 / self.cfg.control_hz
        while width - target_width > 1e-6:
            status = robot.read_arm_status()
            if status.get("fault") or not status.get("available", False):
                self.phase = HybridPhase.FAILED
                raise HybridPickError(f"arm fault during gripper close: {status}")
            width = max(target_width, width - max_step)
            robot.command_gripper(width, effort_n_m=self.cfg.gripper_effort_n_m)
            time.sleep(1.0 / self.cfg.control_hz)

    def _execute_lift(
        self,
        robot: RobotAdapter,
        checker: SafetyChecker,
        executor: PlanExecutor,
        dz_m: float,
    ) -> None:
        target_z = robot.read_ee_pose().z + dz_m
        max_iterations = max(10, int(math.ceil(dz_m / 0.00025)))
        for _ in range(max_iterations):
            pose = robot.read_ee_pose()
            remaining = target_z - pose.z
            if remaining <= 0.001:
                return
            request = min(remaining, self.cfg.max_step_xyz_m[2])
            plan = checker.build_plan(
                current_pose=pose,
                current_joints=robot.read_joint_state(),
                actions=[[0.0, 0.0, request, float("nan")]],
                action_mode="delta_base_m_deg",
            )
            result = executor.execute(plan, human_approved=True, dry_run=False)
            if not result.get("ok"):
                self.phase = HybridPhase.FAILED
                raise HybridPickError(f"lift blocked: {result.get('messages')}")
        self.phase = HybridPhase.FAILED
        raise HybridPickError(f"lift did not converge to z={target_z:.4f} m")


def detect_white_cylinder(
    image: np.ndarray,
    geometry: CameraGeometry,
    *,
    expected_diameter_m: float,
) -> CylinderDetection:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HybridPickError("opencv-python is required for hybrid cylinder detection") from exc
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray([0, 0, 170], np.uint8), np.asarray([179, 85, 255], np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CylinderDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 40.0 or perimeter <= 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        base_center = canonical_pixel_to_base((cx, cy), geometry)
        base_edge = canonical_pixel_to_base((cx + radius, cy), geometry)
        diameter_m = 2.0 * math.hypot(base_edge[0] - base_center[0], base_edge[1] - base_center[1])
        if not 0.55 * expected_diameter_m <= diameter_m <= 1.55 * expected_diameter_m:
            continue
        candidates.append(
            CylinderDetection((float(cx), float(cy)), base_center, diameter_m, area, circularity)
        )
    if len(candidates) != 1:
        raise HybridPickError(f"expected exactly one white cylinder, found {len(candidates)}")
    return candidates[0]


def wrist_white_object_present(image: np.ndarray) -> bool:
    """Conservative post-lift check: a white object must remain near the wrist-image centre."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HybridPickError("opencv-python is required for wrist verification") from exc
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray([0, 0, 175], np.uint8), np.asarray([179, 90, 255], np.uint8))
    height, width = mask.shape
    central = mask[height // 5 : 4 * height // 5, width // 5 : 4 * width // 5]
    fraction = float(np.count_nonzero(central)) / float(central.size)
    return 0.002 <= fraction <= 0.45
