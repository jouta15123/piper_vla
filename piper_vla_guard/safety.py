from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import (
    CartesianIKSolver,
    IKError,
    apply_world_pose_delta,
    world_pose_delta,
)
from .types import EEPose, JointState, SafetyConfig, TrajectoryPlan, TrajectoryStep
from .utils import clamp_abs, euclidean, is_close_list, require_finite


class SafetyChecker:
    """Builds and validates a trajectory plan before Piper execution."""

    def __init__(self, cfg: SafetyConfig, ik_solver: Optional[CartesianIKSolver] = None):
        self.cfg = cfg
        self.ik_solver = ik_solver

    def build_plan(
        self,
        current_pose: EEPose,
        actions: Sequence[Sequence[float]],
        action_mode: str = "delta_base_m_deg",
        current_joints: Optional[JointState] = None,
    ) -> TrajectoryPlan:
        if action_mode not in ("robosuite_osc_pose", "delta_base_m_deg", "absolute_ee_m_deg", "joint_delta_deg"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if len(actions) > self.cfg.max_horizon:
            initial_violation = f"horizon {len(actions)} > max_horizon {self.cfg.max_horizon}"
        else:
            initial_violation = ""

        steps: List[TrajectoryStep] = []
        pose = current_pose
        joints = current_joints

        for idx, raw in enumerate(actions):
            row = self._require_action_row(raw, idx)
            step = self._build_step(idx, action_mode, row, pose, joints, current_pose)
            if action_mode != "joint_delta_deg" and self.ik_solver is not None:
                self._attach_cartesian_ik(step, joints)
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

    def _attach_cartesian_ik(
        self,
        step: TrajectoryStep,
        current_joints: Optional[JointState],
    ) -> None:
        if step.start_pose is None or step.target_pose is None:
            return
        step.start_joints = current_joints
        if current_joints is None:
            step.violations.append("joint IK requires current joint feedback")
            return
        try:
            result = self.ik_solver.solve_pose_delta(
                current_joints,
                step.start_pose,
                step.target_pose,
            )
        except IKError as exc:
            step.violations.append(f"joint IK rejected target: {exc}")
            return
        step.target_joints = result.joints
        for index, (start, target, limit) in enumerate(
            zip(current_joints.values_deg, result.joints.values_deg, self.cfg.max_joint_step_deg),
            start=1,
        ):
            delta = target - start
            if abs(delta) > limit + 1e-9:
                step.violations.append(
                    f"IK j{index} step too large: {delta:.3f}deg > {limit:.3f}deg"
                )
            if self.cfg.settle_each_vla_step:
                commanded_speed = abs(delta) * self.cfg.control_hz
                if commanded_speed > self.cfg.max_commanded_joint_speed_deg_s + 1e-9:
                    step.violations.append(
                        f"IK j{index} cannot fit one {1.0 / self.cfg.control_hz:.3f}s action period: "
                        f"{commanded_speed:.2f}deg/s > {self.cfg.max_commanded_joint_speed_deg_s:.2f}deg/s"
                    )
        self._check_joint_limits(step)
        self._check_ik_path(step, current_joints, result.joints)
        step.warnings.append(
            "IK checked: "
            f"iterations={result.iterations}, position_error={result.position_error_m:.6f}m, "
            f"rotation_error={result.rotation_error_deg:.3f}deg"
        )

    def _check_ik_path(
        self,
        step: TrajectoryStep,
        start_joints: JointState,
        target_joints: JointState,
    ) -> None:
        if step.start_pose is None or self.ik_solver is None:
            return
        max_delta = max(
            abs(a - b) for a, b in zip(start_joints.values_deg, target_joints.values_deg)
        )
        sample_count = max(1, int(math.ceil(max_delta / self.cfg.ik_path_sample_step_deg)))
        fk_start = self.ik_solver.fk.pose(start_joints)
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            joints = JointState(
                tuple(
                    start_joints.values_deg[i]
                    + fraction * (target_joints.values_deg[i] - start_joints.values_deg[i])
                    for i in range(6)
                )  # type: ignore[arg-type]
            )
            fk_pose = self.ik_solver.fk.pose(joints)
            dxyz, rotvec = world_pose_delta(fk_start, fk_pose)
            predicted = apply_world_pose_delta(step.start_pose, dxyz, rotvec)
            sample_step = TrajectoryStep(
                index=step.index,
                action_mode=step.action_mode,
                raw_action=step.raw_action,
                scaled_action=step.scaled_action,
                clipped_action=step.clipped_action,
                start_pose=step.start_pose,
                target_pose=predicted,
            )
            self._check_cartesian_step(sample_step, step.start_pose)
            for violation in sample_step.violations:
                message = f"IK path sample {sample_index}/{sample_count}: {violation}"
                if message not in step.violations:
                    step.violations.append(message)

    def _require_action_row(self, raw: Sequence[float], index: int) -> List[float]:
        row = [float(x) for x in raw]
        if not row:
            raise ValueError(f"actions[{index}] must not be empty")
        for value_index, value in enumerate(row):
            if value_index == len(row) - 1 and math.isnan(value):
                continue
            require_finite(value, f"actions[{index}] value")
        return row

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
        if action_mode == "robosuite_osc_pose":
            return self._step_robosuite_osc_pose(index, raw, pose, initial_pose)
        if action_mode == "delta_base_m_deg":
            return self._step_delta_pose(index, raw, pose, initial_pose)
        if action_mode == "absolute_ee_m_deg":
            return self._step_absolute_pose(index, raw, pose, initial_pose)
        if action_mode == "joint_delta_deg":
            return self._step_joint_delta(index, raw, pose, joints)
        raise ValueError(action_mode)

    def _normalize_row(self, raw: List[float], action_mode: str) -> List[float]:
        if action_mode in ("robosuite_osc_pose", "delta_base_m_deg"):
            if len(raw) == 3:
                return raw + [0.0, 0.0, 0.0, float("nan")]
            if len(raw) == 4:
                return raw[:3] + [0.0, 0.0, 0.0, raw[3]]
            if len(raw) == 6:
                return raw + [float("nan")]
            if len(raw) == 7:
                return raw
            raise ValueError(f"{action_mode} rows must have 3, 4, 6, or 7 values")
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

    def _step_robosuite_osc_pose(
        self,
        index: int,
        raw: List[float],
        pose: EEPose,
        initial_pose: EEPose,
    ) -> TrajectoryStep:
        scaled = [
            raw[0] * self.cfg.robosuite_osc_xyz_scale_m,
            raw[1] * self.cfg.robosuite_osc_xyz_scale_m,
            raw[2] * self.cfg.robosuite_osc_xyz_scale_m,
            math.degrees(raw[3] * self.cfg.robosuite_osc_rot_scale_rad),
            math.degrees(raw[4] * self.cfg.robosuite_osc_rot_scale_rad),
            math.degrees(raw[5] * self.cfg.robosuite_osc_rot_scale_rad),
            raw[6],
        ]
        clipped = (
            _uniform_limit(scaled[:3], self.cfg.max_step_xyz_m)
            + _uniform_limit(scaled[3:6], self.cfg.max_step_rpy_deg)
            + [scaled[6]]
        )
        target = apply_world_pose_delta(
            pose,
            clipped[:3],
            np.radians(np.asarray(clipped[3:6], dtype=np.float64)),
        )
        step = TrajectoryStep(
            index=index,
            action_mode="robosuite_osc_pose",
            raw_action=raw,
            scaled_action=scaled,
            clipped_action=clipped,
            start_pose=pose,
            target_pose=target,
            gripper_m=self._map_gripper_robosuite(clipped[6]),
        )
        if not is_close_list(scaled[:6], clipped[:6]):
            msg = "robosuite OSC action clipped by uniform one-step scaling"
            step.warnings.append(msg)
            if self.cfg.reject_on_clip:
                step.violations.append(msg)
        self._check_cartesian_step(step, initial_pose)
        self._check_gripper(step)
        return step

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
        clipped = (
            _uniform_limit(scaled[:3], self.cfg.max_step_xyz_m)
            + _uniform_limit(scaled[3:6], self.cfg.max_step_rpy_deg)
            + [scaled[6]]
        )
        target = pose.moved_by(clipped[:3], clipped[3:6])
        step = TrajectoryStep(
            index=index,
            action_mode="delta_base_m_deg",
            raw_action=raw,
            scaled_action=scaled,
            clipped_action=clipped,
            start_pose=pose,
            target_pose=target,
            gripper_m=self._map_gripper_normalized_01(clipped[6]),
        )
        if not is_close_list(scaled[:6], clipped[:6]):
            msg = "action clipped by uniform one-step scaling"
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
        clipped_delta = _uniform_limit(deltas[:3], self.cfg.max_step_xyz_m) + _uniform_limit(
            deltas[3:6], self.cfg.max_step_rpy_deg
        )
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
            gripper_m=self._map_gripper_normalized_01(raw[6]),
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
                gripper_m=self._map_gripper_normalized_01(raw[6]),
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
            gripper_m=self._map_gripper_normalized_01(raw[6]),
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
        floor_error = workspace_floor_error(target.xyz(), self.cfg)
        if floor_error:
            step.violations.append(floor_error)
        self._check_tool_envelope(step)
        self._check_safety_planes(step)

        dxyz = [target.x - step.start_pose.x, target.y - step.start_pose.y, target.z - step.start_pose.z]
        drpy = (
            list(step.clipped_action[3:6])
            if step.action_mode == "robosuite_osc_pose" and len(step.clipped_action) >= 6
            else [_angle_delta_deg(a, b) for a, b in zip(step.start_pose.rpy(), target.rpy())]
        )
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

    def _check_safety_planes(self, step: TrajectoryStep) -> None:
        if step.target_pose is None:
            return
        target = step.target_pose
        for plane in self.cfg.safety_planes:
            norm = math.sqrt(sum(v * v for v in plane.normal))
            if norm <= 1e-12:
                step.violations.append(f"safety plane {plane.name} normal is zero")
                continue
            signed_distance = (
                plane.normal[0] * (target.x - plane.point[0])
                + plane.normal[1] * (target.y - plane.point[1])
                + plane.normal[2] * (target.z - plane.point[2])
            ) / norm
            if signed_distance < plane.margin_m:
                step.violations.append(
                    f"safety plane {plane.name} violated: signed distance "
                    f"{signed_distance:.4f} < margin {plane.margin_m:.4f}"
                )

    def _check_tool_envelope(self, step: TrajectoryStep) -> None:
        if step.target_pose is None or not self.cfg.tool_points_m:
            return
        target = step.target_pose
        rot = _rpy_rotation_matrix(target.rx, target.ry, target.rz)
        floor = self.cfg.table_z_m + self.cfg.table_margin_m
        for index, local in enumerate(self.cfg.tool_points_m):
            world = tuple(
                target.xyz()[row] + sum(rot[row][column] * local[column] for column in range(3))
                for row in range(3)
            )
            world_z = world[2]
            if world_z < floor:
                step.violations.append(
                    f"tool point {index} below table clearance: {world_z:.4f} < {floor:.4f}"
                )
            floor_error = workspace_floor_error(world, self.cfg)
            if floor_error:
                step.violations.append(f"tool point {index} {floor_error}")

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

    def _map_gripper_normalized_01(self, normalized: float) -> Optional[float]:
        normalized = float(normalized)
        if math.isnan(normalized):
            return None
        require_finite(normalized, "gripper")
        # Clamp to [0, 1] — VLA models often output values slightly outside
        # this range (e.g. -0.1 for "fully open").
        normalized = max(0.0, min(1.0, normalized))
        return (1.0 - normalized) * self.cfg.gripper_open_m + normalized * self.cfg.gripper_closed_m

    def _map_gripper_robosuite(self, action: float) -> Optional[float]:
        action = float(action)
        if math.isnan(action):
            return None
        require_finite(action, "gripper")
        low = self.cfg.robosuite_gripper_open_action
        high = self.cfg.robosuite_gripper_close_action
        t = (action - low) / (high - low)
        t = max(0.0, min(1.0, t))
        # Training geometry: fingertip separation = 20 mm mechanical offset
        # + two 0..35 mm finger qpos values. This gives 90 mm at action=-1
        # and 20 mm at action=0, independent of Piper's 95 mm hard stroke.
        training_open_width = (
            self.cfg.robosuite_gripper_min_width_m
            + 2.0 * self.cfg.robosuite_gripper_qpos_max_m
        )
        width = (1.0 - t) * training_open_width + t * self.cfg.robosuite_gripper_min_width_m
        return min(self.cfg.gripper_open_m, max(self.cfg.gripper_closed_m, width))

    def _check_gripper(self, step: TrajectoryStep) -> None:
        if step.gripper_m is None:
            return
        low, high = sorted([self.cfg.gripper_closed_m, self.cfg.gripper_open_m])
        if not (low <= step.gripper_m <= high):
            step.violations.append(
                f"gripper out of range: {step.gripper_m:.4f} m not in [{low:.4f}, {high:.4f}]"
            )
        raw_g = step.raw_action[-1]
        if math.isnan(float(raw_g)):
            return
        if step.action_mode == "robosuite_osc_pose":
            low, high = sorted([self.cfg.robosuite_gripper_open_action, self.cfg.robosuite_gripper_close_action])
            if not (low <= float(raw_g) <= high):
                step.warnings.append(f"robosuite gripper outside [{low}, {high}], got {raw_g} (clamped)")
        elif not (0.0 <= float(raw_g) <= 1.0):
            step.warnings.append(f"normalized gripper outside [0, 1], got {raw_g} (clamped)")

    def _summary(self, steps: Sequence[TrajectoryStep], approved: bool) -> str:
        violations = sum(len(step.violations) for step in steps)
        warnings = sum(len(step.warnings) for step in steps)
        if not steps:
            return "No steps."
        if approved:
            return f"SAFETY OK: {len(steps)} steps, {warnings} warnings."
        return f"REJECTED: {len(steps)} steps, {violations} violations, {warnings} warnings."


def _uniform_limit(values: Sequence[float], limits: Sequence[float]) -> List[float]:
    """Scale a vector without changing its direction so every axis fits its limit."""
    scale = 1.0
    for value, limit in zip(values, limits):
        if abs(float(value)) > float(limit) and abs(float(value)) > 1e-15:
            scale = min(scale, float(limit) / abs(float(value)))
    return [float(value) * scale for value in values]


def _angle_delta_deg(start: float, end: float) -> float:
    return (float(end) - float(start) + 180.0) % 360.0 - 180.0


def _rpy_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> Tuple[Tuple[float, ...], ...]:
    """Return Rz(yaw) * Ry(pitch) * Rx(roll), matching Piper's fixed-axis RPY."""
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def workspace_floor_error(point_xyz: Sequence[float], cfg: SafetyConfig) -> Optional[str]:
    """Validate a point against the calibrated four-corner work surface."""
    corners = cfg.workspace_floor_corners_m
    if not corners:
        return None
    point = tuple(float(value) for value in point_xyz)
    area_twice = sum(
        corners[i][0] * corners[(i + 1) % 4][1]
        - corners[(i + 1) % 4][0] * corners[i][1]
        for i in range(4)
    )
    orientation = 1.0 if area_twice > 0.0 else -1.0
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        signed_distance = orientation * (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / length
        if signed_distance < cfg.workspace_floor_margin_m:
            return (
                f"outside four-corner workspace at edge {index}: signed distance "
                f"{signed_distance:.4f}m < margin {cfg.workspace_floor_margin_m:.4f}m"
            )

    design = np.asarray([[x, y, 1.0] for x, y, _ in corners], dtype=np.float64)
    heights = np.asarray([z for _, _, z in corners], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, heights, rcond=None)
    residual = float(np.max(np.abs(design @ coefficients - heights)))
    if residual > cfg.workspace_floor_max_fit_error_m:
        return (
            f"four-corner floor is not planar: fit error {residual:.4f}m > "
            f"{cfg.workspace_floor_max_fit_error_m:.4f}m"
        )
    floor_z = float(coefficients[0] * point[0] + coefficients[1] * point[1] + coefficients[2])
    required_z = floor_z + cfg.workspace_floor_margin_m
    if point[2] < required_z:
        return f"below four-corner floor: z={point[2]:.4f}m < {required_z:.4f}m"
    return None
