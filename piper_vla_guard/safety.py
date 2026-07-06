from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .types import EEPose, JointState, SafetyConfig, TrajectoryPlan, TrajectoryStep
from .utils import clamp_abs, euclidean, is_close_list, require_finite


class SafetyChecker:
    """Builds and validates a trajectory plan before Piper execution."""

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def build_plan(
        self,
        current_pose: EEPose,
        actions: Sequence[Sequence[float]],
        action_mode: str = "delta_base_m_deg",
        current_joints: Optional[JointState] = None,
    ) -> TrajectoryPlan:
        if action_mode not in ("delta_base_m_deg", "absolute_ee_m_deg", "joint_delta_deg"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if len(actions) > self.cfg.max_horizon:
            initial_violation = f"horizon {len(actions)} > max_horizon {self.cfg.max_horizon}"
        else:
            initial_violation = ""

        steps: List[TrajectoryStep] = []
        pose = current_pose
        joints = current_joints

        for idx, raw in enumerate(actions):
            row = [require_finite(x, f"actions[{idx}] value") for x in raw]
            step = self._build_step(idx, action_mode, row, pose, joints, current_pose)
            if initial_violation:
                step.violations.append(initial_violation)
            steps.append(step)
            if step.target_pose is not None:
                pose = step.target_pose
            if step.target_joints is not None:
                joints = step.target_joints

        if current_joints is not None:
            margin_warnings = self._joint_margin_warnings(current_joints)
            if margin_warnings and steps:
                steps[0].warnings.extend(margin_warnings)

        approved = all(step.ok for step in steps)
        if self.cfg.reject_on_warning:
            approved = approved and all(not step.warnings for step in steps)
        summary = self._summary(steps, approved)
        return TrajectoryPlan(
            action_mode=action_mode,
            initial_pose=current_pose,
            initial_joints=current_joints,
            steps=steps,
            config_snapshot=self.cfg.snapshot(),
            approved_by_safety=approved,
            summary=summary,
        )

    def _build_step(
        self,
        index: int,
        action_mode: str,
        raw: List[float],
        pose: EEPose,
        joints: Optional[JointState],
        initial_pose: EEPose,
    ) -> TrajectoryStep:
        raw = self._normalize_row(raw, action_mode)
        if action_mode == "delta_base_m_deg":
            return self._step_delta_pose(index, raw, pose, initial_pose)
        if action_mode == "absolute_ee_m_deg":
            return self._step_absolute_pose(index, raw, pose, initial_pose)
        if action_mode == "joint_delta_deg":
            return self._step_joint_delta(index, raw, pose, joints)
        raise ValueError(action_mode)

    def _normalize_row(self, raw: List[float], action_mode: str) -> List[float]:
        if action_mode == "delta_base_m_deg":
            if len(raw) == 3:
                return raw + [0.0, 0.0, 0.0, float("nan")]
            if len(raw) == 4:
                return raw[:3] + [0.0, 0.0, 0.0, raw[3]]
            if len(raw) == 6:
                return raw + [float("nan")]
            if len(raw) == 7:
                return raw
            raise ValueError("delta_base_m_deg rows must have 3, 4, 6, or 7 values")
        if action_mode == "absolute_ee_m_deg":
            if len(raw) == 6:
                return raw + [float("nan")]
            if len(raw) == 7:
                return raw
            raise ValueError("absolute_ee_m_deg rows must have 6 or 7 values")
        if action_mode == "joint_delta_deg":
            if len(raw) == 6:
                return raw + [float("nan")]
            if len(raw) == 7:
                return raw
            raise ValueError("joint_delta_deg rows must have 6 or 7 values")
        raise ValueError(action_mode)

    def _step_delta_pose(
        self,
        index: int,
        raw: List[float],
        pose: EEPose,
        initial_pose: EEPose,
    ) -> TrajectoryStep:
        scaled = [
            raw[0] * self.cfg.action_scale_xyz,
            raw[1] * self.cfg.action_scale_xyz,
            raw[2] * self.cfg.action_scale_xyz,
            raw[3] * self.cfg.action_scale_rpy,
            raw[4] * self.cfg.action_scale_rpy,
            raw[5] * self.cfg.action_scale_rpy,
            raw[6],
        ]
        clipped = [
            clamp_abs(scaled[0], self.cfg.max_step_xyz_m[0]),
            clamp_abs(scaled[1], self.cfg.max_step_xyz_m[1]),
            clamp_abs(scaled[2], self.cfg.max_step_xyz_m[2]),
            clamp_abs(scaled[3], self.cfg.max_step_rpy_deg[0]),
            clamp_abs(scaled[4], self.cfg.max_step_rpy_deg[1]),
            clamp_abs(scaled[5], self.cfg.max_step_rpy_deg[2]),
            scaled[6],
        ]
        target = pose.moved_by(clipped[:3], clipped[3:6])
        step = TrajectoryStep(
            index=index,
            action_mode="delta_base_m_deg",
            raw_action=raw,
            scaled_action=scaled,
            clipped_action=clipped,
            start_pose=pose,
            target_pose=target,
            gripper_m=self._map_gripper(clipped[6]),
        )
        if not is_close_list(scaled[:6], clipped[:6]):
            msg = "action clipped by one-step limit"
            step.warnings.append(msg)
            if self.cfg.reject_on_clip:
                step.violations.append(msg)
        self._check_cartesian_step(step, initial_pose)
        self._check_gripper(step)
        return step

    def _step_absolute_pose(
        self,
        index: int,
        raw: List[float],
        pose: EEPose,
        initial_pose: EEPose,
    ) -> TrajectoryStep:
        scaled = list(raw)
        target = EEPose(*scaled[:6])
        deltas = [
            target.x - pose.x,
            target.y - pose.y,
            target.z - pose.z,
            target.rx - pose.rx,
            target.ry - pose.ry,
            target.rz - pose.rz,
        ]
        clipped_delta = [
            clamp_abs(deltas[0], self.cfg.max_step_xyz_m[0]),
            clamp_abs(deltas[1], self.cfg.max_step_xyz_m[1]),
            clamp_abs(deltas[2], self.cfg.max_step_xyz_m[2]),
            clamp_abs(deltas[3], self.cfg.max_step_rpy_deg[0]),
            clamp_abs(deltas[4], self.cfg.max_step_rpy_deg[1]),
            clamp_abs(deltas[5], self.cfg.max_step_rpy_deg[2]),
        ]
        clipped_target = pose.moved_by(clipped_delta[:3], clipped_delta[3:])
        clipped = clipped_target.as_list() + [raw[6]]
        step = TrajectoryStep(
            index=index,
            action_mode="absolute_ee_m_deg",
            raw_action=raw,
            scaled_action=scaled,
            clipped_action=clipped,
            start_pose=pose,
            target_pose=clipped_target,
            gripper_m=self._map_gripper(raw[6]),
        )
        if not is_close_list(deltas, clipped_delta):
            msg = "absolute target clipped by one-step limit"
            step.warnings.append(msg)
            if self.cfg.reject_on_clip:
                step.violations.append(msg)
        self._check_cartesian_step(step, initial_pose)
        self._check_gripper(step)
        return step

    def _step_joint_delta(
        self,
        index: int,
        raw: List[float],
        pose: EEPose,
        joints: Optional[JointState],
    ) -> TrajectoryStep:
        scaled = list(raw)
        if joints is None:
            step = TrajectoryStep(
                index=index,
                action_mode="joint_delta_deg",
                raw_action=raw,
                scaled_action=scaled,
                clipped_action=scaled,
                start_pose=pose,
                target_pose=None,
                start_joints=None,
                target_joints=None,
                gripper_m=self._map_gripper(raw[6]),
            )
            step.violations.append("joint_delta_deg requires current joint feedback")
            self._check_gripper(step)
            return step
        clipped_delta = [clamp_abs(scaled[i], self.cfg.max_joint_step_deg[i]) for i in range(6)]
        target_joints = JointState(tuple(joints.values_deg[i] + clipped_delta[i] for i in range(6)))  # type: ignore[arg-type]
        clipped = clipped_delta + [raw[6]]
        step = TrajectoryStep(
            index=index,
            action_mode="joint_delta_deg",
            raw_action=raw,
            scaled_action=scaled,
            clipped_action=clipped,
            start_pose=pose,
            start_joints=joints,
            target_joints=target_joints,
            gripper_m=self._map_gripper(raw[6]),
        )
        if not is_close_list(scaled[:6], clipped_delta):
            msg = "joint action clipped by max_joint_step_deg"
            step.warnings.append(msg)
            if self.cfg.reject_on_clip:
                step.violations.append(msg)
        self._check_joint_limits(step)
        self._check_gripper(step)
        return step

    def _check_cartesian_step(self, step: TrajectoryStep, initial_pose: EEPose) -> None:
        if step.start_pose is None or step.target_pose is None:
            return
        target = step.target_pose
        if not (self.cfg.workspace_x_m[0] <= target.x <= self.cfg.workspace_x_m[1]):
            step.violations.append(
                f"x out of workspace: {target.x:.4f} not in {self.cfg.workspace_x_m}"
            )
        if not (self.cfg.workspace_y_m[0] <= target.y <= self.cfg.workspace_y_m[1]):
            step.violations.append(
                f"y out of workspace: {target.y:.4f} not in {self.cfg.workspace_y_m}"
            )
        if not (self.cfg.workspace_z_m[0] <= target.z <= self.cfg.workspace_z_m[1]):
            step.violations.append(
                f"z out of workspace: {target.z:.4f} not in {self.cfg.workspace_z_m}"
            )
        if target.z < self.cfg.min_z_m:
            step.violations.append(f"below min_z_m: {target.z:.4f} < {self.cfg.min_z_m:.4f}")

        dxyz = [target.x - step.start_pose.x, target.y - step.start_pose.y, target.z - step.start_pose.z]
        drpy = [target.rx - step.start_pose.rx, target.ry - step.start_pose.ry, target.rz - step.start_pose.rz]
        for axis, value, limit in zip("xyz", dxyz, self.cfg.max_step_xyz_m):
            if abs(value) > limit + 1e-12:
                step.violations.append(f"d{axis} too large: {value:.6f} > {limit:.6f}")
        for axis, value, limit in zip(("rx", "ry", "rz"), drpy, self.cfg.max_step_rpy_deg):
            if abs(value) > limit + 1e-12:
                step.violations.append(f"d{axis} too large: {value:.6f} > {limit:.6f}")
        total = euclidean(target.xyz(), initial_pose.xyz())
        if total > self.cfg.max_total_translation_m:
            step.violations.append(
                f"total translation too large: {total:.4f} > {self.cfg.max_total_translation_m:.4f}"
            )

    def _check_joint_limits(self, step: TrajectoryStep) -> None:
        if step.target_joints is None:
            return
        for i, value in enumerate(step.target_joints.values_deg, start=1):
            low, high = self.cfg.joint_limits_deg[f"j{i}"]
            if not (low <= value <= high):
                step.violations.append(f"j{i} out of limit: {value:.3f} not in [{low}, {high}]")
            elif value < low + self.cfg.joint_limit_margin_deg:
                step.warnings.append(f"j{i} near lower limit: {value:.3f}")
            elif value > high - self.cfg.joint_limit_margin_deg:
                step.warnings.append(f"j{i} near upper limit: {value:.3f}")

    def _joint_margin_warnings(self, joints: JointState) -> List[str]:
        warnings: List[str] = []
        for i, value in enumerate(joints.values_deg, start=1):
            low, high = self.cfg.joint_limits_deg[f"j{i}"]
            if value < low + self.cfg.joint_limit_margin_deg:
                warnings.append(f"current j{i} near lower limit: {value:.3f}")
            if value > high - self.cfg.joint_limit_margin_deg:
                warnings.append(f"current j{i} near upper limit: {value:.3f}")
        return warnings

    def _map_gripper(self, normalized: float) -> Optional[float]:
        normalized = float(normalized)
        if math.isnan(normalized):
            return None
        require_finite(normalized, "gripper")
        return (1.0 - normalized) * self.cfg.gripper_open_m + normalized * self.cfg.gripper_closed_m

    def _check_gripper(self, step: TrajectoryStep) -> None:
        if step.gripper_m is None:
            return
        low, high = sorted([self.cfg.gripper_closed_m, self.cfg.gripper_open_m])
        if not (low <= step.gripper_m <= high):
            step.violations.append(
                f"gripper out of range: {step.gripper_m:.4f} m not in [{low:.4f}, {high:.4f}]"
            )
        raw_g = step.raw_action[-1]
        if not math.isnan(float(raw_g)) and not (0.0 <= float(raw_g) <= 1.0):
            step.violations.append(f"normalized gripper must be in [0, 1], got {raw_g}")

    def _summary(self, steps: Sequence[TrajectoryStep], approved: bool) -> str:
        violations = sum(len(step.violations) for step in steps)
        warnings = sum(len(step.warnings) for step in steps)
        if not steps:
            return "No steps."
        if approved:
            return f"SAFETY OK: {len(steps)} steps, {warnings} warnings."
        return f"REJECTED: {len(steps)} steps, {violations} violations, {warnings} warnings."
