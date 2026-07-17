from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from .logging_utils import JsonlLogger
from .piper_adapter import RobotAdapter
from .safety import _rpy_rotation_matrix, workspace_floor_error
from .types import EEPose, JointState, SafetyConfig, TrajectoryPlan
from .utils import euclidean


class PlanExecutor:
    def __init__(self, robot: RobotAdapter, cfg: SafetyConfig, logger: Optional[JsonlLogger] = None):
        self.robot = robot
        self.cfg = cfg
        self.logger = logger

    def execute(self, plan: TrajectoryPlan, human_approved: bool, dry_run: Optional[bool] = None) -> Dict[str, Any]:
        use_dry_run = self.cfg.dry_run if dry_run is None else bool(dry_run)
        result: Dict[str, Any] = {
            "ok": False,
            "dry_run": use_dry_run,
            "executed_steps": 0,
            "max_tracking_error_m": 0.0,
            "stop_kind": None,
            "messages": [],
        }
        if not plan.approved_by_safety:
            result["messages"].append("Plan rejected by safety checker; execution blocked.")
            self._log("execution_blocked", {"reason": "safety_rejected", "plan": plan.to_dict()})
            return result
        if self.cfg.require_manual_approval and not human_approved:
            result["messages"].append("Manual approval checkbox is not set; execution blocked.")
            self._log("execution_blocked", {"reason": "manual_approval_missing"})
            return result
        if use_dry_run:
            result["ok"] = True
            result["messages"].append("Dry run: no commands sent to Piper.")
            result["executed_steps"] = len(plan.steps)
            self._log("dry_run_plan", {"plan": plan.to_dict()})
            return result

        if plan.action_mode != "joint_delta_deg" and self.cfg.cartesian_execution_mode == "joint_ik":
            missing = [step.index for step in plan.steps if step.target_pose is not None and step.target_joints is None]
            if missing:
                result["messages"].append(
                    f"Joint-IK execution requires checked target_joints; missing at steps {missing}."
                )
                self._log("execution_blocked", {"reason": "joint_ik_targets_missing", "steps": missing})
                return result

        preflight_error = self._preflight_error(plan)
        if preflight_error:
            result["messages"].append(preflight_error)
            self._log("execution_blocked", {"reason": "preflight_failed", "message": preflight_error})
            return result

        last_command_pose = self.robot.read_ee_pose()
        last_velocity = (0.0, 0.0, 0.0)
        last_rpy_velocity = (0.0, 0.0, 0.0)
        last_joints = self.robot.read_joint_state()
        for step in plan.steps:
            status = self.robot.read_arm_status()
            status_error = self._status_error(status)
            if status_error:
                result["messages"].append(f"Arm status fault before step {step.index}: {status_error}; {status}")
                self._log("execution_aborted", {"step": step.index, "status": status})
                return result
            actual_pose = self.robot.read_ee_pose()
            workspace_error = self._workspace_error(actual_pose)
            if workspace_error:
                result["messages"].append(f"Actual EE safety check failed before step {step.index}: {workspace_error}")
                self._log("execution_aborted", {"step": step.index, "reason": workspace_error})
                return result
            current_joints = self.robot.read_joint_state()
            joint_error = self._joint_state_error(current_joints)
            if joint_error:
                result["messages"].append(f"Actual joint safety check failed before step {step.index}: {joint_error}")
                self._log("execution_aborted", {"step": step.index, "reason": joint_error})
                return result

            use_joint_target = plan.action_mode == "joint_delta_deg" or (
                self.cfg.cartesian_execution_mode == "joint_ik" and step.target_joints is not None
            )
            if use_joint_target:
                if step.target_joints is None:
                    result["messages"].append(f"Step {step.index} has no target_joints")
                    return result
                if current_joints is None:
                    result["messages"].append(f"Step {step.index}: joint feedback is unavailable")
                    return result
                if step.start_joints is not None:
                    drift_error = self._joint_drift_error(step.start_joints, current_joints)
                    if drift_error:
                        stop_message = self.robot.pause_hold()
                        result["stop_kind"] = "pause_hold"
                        result["messages"].append(
                            f"Step {step.index} start-joint mismatch: {drift_error}; {stop_message}"
                        )
                        return result
                measured_joints, interpolation_error, stop_kind = self._stream_joint_target(
                    current_joints,
                    step.target_joints,
                    speed_pct=self.cfg.speed_pct,
                )
                if interpolation_error:
                    result["stop_kind"] = stop_kind
                    result["messages"].append(
                        f"Joint interpolation failed at step {step.index}: {interpolation_error}"
                    )
                    self._log(
                        "execution_aborted",
                        {"step": step.index, "reason": "joint_interpolation", "message": interpolation_error},
                    )
                    return result
                if step.target_pose is not None:
                    last_command_pose = step.target_pose
            else:
                if step.target_pose is None:
                    result["messages"].append(f"Step {step.index} has no target_pose")
                    return result
                command_pose, last_velocity, last_rpy_velocity = self._shape_cartesian_command(
                    last_command_pose, step.target_pose, last_velocity, last_rpy_velocity
                )
                command_started = time.monotonic()
                self.robot.command_end_pose(command_pose, speed_pct=self.cfg.speed_pct, move_mode="L")
                last_command_pose = command_pose
                time.sleep(max(0.0, 1.0 / self.cfg.control_hz))
                measured_joints = self.robot.read_joint_state()
                if measured_joints is not None and current_joints is not None:
                    elapsed = max(1e-6, time.monotonic() - command_started)
                    max_speed = max(
                        abs(a - b) / elapsed
                        for a, b in zip(measured_joints.values_deg, current_joints.values_deg)
                    )
                    if max_speed > self.cfg.max_measured_joint_speed_deg_s:
                        stop_message = self.robot.pause_hold()
                        result["stop_kind"] = "pause_hold"
                        result["messages"].append(
                            f"Measured joint speed {max_speed:.2f} deg/s > "
                            f"{self.cfg.max_measured_joint_speed_deg_s:.2f} deg/s; {stop_message}."
                        )
                        return result

            if step.gripper_m is not None:
                self.robot.command_gripper(step.gripper_m, effort_n_m=self.cfg.gripper_effort_n_m)

            result["executed_steps"] += 1
            last_joints = measured_joints
            if step.target_pose is not None:
                if self.cfg.settle_each_vla_step:
                    after_pose, target_error_m = self._wait_for_tracking(
                        last_command_pose,
                        refresh_cartesian_target=not use_joint_target,
                    )
                else:
                    after_pose = self.robot.read_ee_pose()
                    target_error_m = euclidean(last_command_pose.xyz(), after_pose.xyz())
                result["max_tracking_error_m"] = max(result["max_tracking_error_m"], target_error_m)
                if self.cfg.settle_each_vla_step and target_error_m > self.cfg.tracking_abort_error_m:
                    stop_message = self.robot.pause_hold()
                    result["stop_kind"] = "pause_hold"
                    result["messages"].append(
                        f"Tracking error after step {step.index}: {target_error_m:.6f} m > "
                        f"{self.cfg.tracking_abort_error_m:.6f} m; {stop_message}."
                    )
                    self._log(
                        "execution_aborted",
                        {"step": step.index, "reason": "tracking_error", "error_m": target_error_m},
                    )
                    return result
                if self.cfg.settle_each_vla_step and target_error_m > self.cfg.tracking_hold_error_m:
                    stop_message = self.robot.pause_hold()
                    result["stop_kind"] = "pause_hold"
                    result["messages"].append(
                        f"Tracking did not settle after step {step.index}: {target_error_m:.6f} m > "
                        f"{self.cfg.tracking_hold_error_m:.6f} m; "
                        f"requested XYZ={_fmt_xyz(last_command_pose)}, "
                        f"after XYZ={_fmt_xyz(after_pose)}; {stop_message}."
                    )
                    self._log(
                        "execution_aborted",
                        {"step": step.index, "reason": "tracking_not_settled", "error_m": target_error_m},
                    )
                    return result
                workspace_error = self._workspace_error(after_pose)
                if workspace_error:
                    stop_message = self.robot.pause_hold()
                    result["stop_kind"] = "pause_hold"
                    result["messages"].append(
                        f"Actual EE safety check failed after step {step.index}: "
                        f"{workspace_error}; {stop_message}"
                    )
                    self._log("execution_aborted", {"step": step.index, "reason": workspace_error})
                    return result
                result["messages"].append(
                    f"Step {step.index}: requested XYZ={_fmt_xyz(step.target_pose)}, "
                    f"commanded XYZ={_fmt_xyz(last_command_pose)}, "
                    f"after XYZ={_fmt_xyz(after_pose)}, target error={target_error_m:.6f} m"
                )
            self._log("step_executed", {"step": step.to_dict(), "status_before": status})

        result["ok"] = True
        result["messages"].append(f"Executed {result['executed_steps']} steps.")
        self._log("execution_complete", result)
        return result

    def _stream_joint_target(
        self,
        start: JointState,
        target: JointState,
        *,
        speed_pct: int,
    ) -> tuple[Optional[JointState], Optional[str], Optional[str]]:
        """Send a checked linear joint path at control_hz with live guards."""
        if not self.cfg.settle_each_vla_step:
            # Robosuite-style position-servo update: publish the checked IK
            # target once, advance one 20 Hz action period, then let the next
            # VLA action rebase on measured feedback. The arm is not required
            # to reach the full Cartesian delta within this single period.
            previous_time = time.monotonic()
            self.robot.command_joints(target, speed_pct=speed_pct)
            time.sleep(1.0 / self.cfg.control_hz)
            status_error = self._status_error(self.robot.read_arm_status())
            if status_error:
                stop = self.robot.pause_hold()
                return start, f"realtime servo update: {status_error}; {stop}", "pause_hold"
            measured = self.robot.read_joint_state()
            if measured is None:
                stop = self.robot.pause_hold()
                return None, f"realtime servo update: joint feedback lost; {stop}", "pause_hold"
            elapsed = max(1e-6, time.monotonic() - previous_time)
            max_speed = max(
                abs(a - b) / elapsed for a, b in zip(measured.values_deg, start.values_deg)
            )
            if max_speed > self.cfg.max_measured_joint_speed_deg_s:
                stop = self.robot.pause_hold()
                return (
                    measured,
                    f"realtime servo measured joint speed {max_speed:.2f}deg/s > "
                    f"{self.cfg.max_measured_joint_speed_deg_s:.2f}deg/s; {stop}",
                    "pause_hold",
                )
            joint_error = self._joint_state_error(measured)
            pose_error = self._workspace_error(self.robot.read_ee_pose())
            if joint_error or pose_error:
                stop = self.robot.pause_hold()
                reason = joint_error or pose_error
                return measured, f"realtime servo update: {reason}; {stop}", "pause_hold"
            return measured, None, None

        max_delta = max(abs(a - b) for a, b in zip(start.values_deg, target.values_deg))
        count = max(1, int(math.ceil(max_delta / self.cfg.ik_path_sample_step_deg)))
        max_commands_per_action = max(1, int(self.cfg.joint_command_hz // self.cfg.control_hz))
        if count > max_commands_per_action:
            stop = self.robot.pause_hold()
            return (
                start,
                f"{count} interpolation commands do not fit one action period; "
                f"limit={max_commands_per_action}; {stop}",
                "pause_hold",
            )
        command_period_s = (1.0 / self.cfg.control_hz) / count
        if command_period_s + 1e-12 < 1.0 / self.cfg.joint_command_hz:
            stop = self.robot.pause_hold()
            return (
                start,
                f"joint command rate would exceed configured joint_command_hz; {stop}",
                "pause_hold",
            )
        previous = start
        previous_time = time.monotonic()
        for sample_index in range(1, count + 1):
            fraction = sample_index / count
            command = JointState(
                tuple(
                    start.values_deg[index]
                    + fraction * (target.values_deg[index] - start.values_deg[index])
                    for index in range(6)
                )  # type: ignore[arg-type]
            )
            self.robot.command_joints(command, speed_pct=speed_pct)
            time.sleep(command_period_s)
            status_error = self._status_error(self.robot.read_arm_status())
            if status_error:
                stop = self.robot.pause_hold()
                return previous, f"sample {sample_index}/{count}: {status_error}; {stop}", "pause_hold"
            measured = self.robot.read_joint_state()
            if measured is None:
                stop = self.robot.pause_hold()
                return None, f"sample {sample_index}/{count}: joint feedback lost; {stop}", "pause_hold"
            now = time.monotonic()
            elapsed = max(1e-6, now - previous_time)
            max_speed = max(
                abs(a - b) / elapsed for a, b in zip(measured.values_deg, previous.values_deg)
            )
            if max_speed > self.cfg.max_measured_joint_speed_deg_s:
                stop = self.robot.pause_hold()
                return (
                    measured,
                    f"sample {sample_index}/{count}: measured joint speed {max_speed:.2f}deg/s > "
                    f"{self.cfg.max_measured_joint_speed_deg_s:.2f}deg/s; {stop}",
                    "pause_hold",
                )
            joint_error = self._joint_state_error(measured)
            pose_error = self._workspace_error(self.robot.read_ee_pose())
            if joint_error or pose_error:
                stop = self.robot.pause_hold()
                reason = joint_error or pose_error
                return measured, f"sample {sample_index}/{count}: {reason}; {stop}", "pause_hold"
            previous = measured
            previous_time = now
        return previous, None, None

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.write(event, payload)

    def _preflight_error(self, plan: TrajectoryPlan) -> Optional[str]:
        status_error = self._status_error(self.robot.read_arm_status())
        if status_error:
            return f"Preflight status check failed: {status_error}"

        current_pose = self.robot.read_ee_pose()
        pose_error = self._pose_drift_error(plan.initial_pose, current_pose)
        if pose_error:
            return pose_error

        if plan.initial_joints is not None:
            current_joints = self.robot.read_joint_state()
            if current_joints is None:
                return "Preflight joint check failed: current joint feedback is unavailable."
            joint_error = self._joint_drift_error(plan.initial_joints, current_joints)
            if joint_error:
                return joint_error

        return None

    def _shape_cartesian_command(
        self,
        current_command: EEPose,
        requested: EEPose,
        last_velocity: tuple[float, float, float],
        last_rpy_velocity: tuple[float, float, float],
    ) -> tuple[EEPose, tuple[float, float, float], tuple[float, float, float]]:
        dt = 1.0 / self.cfg.control_hz
        delta = tuple(b - a for a, b in zip(current_command.xyz(), requested.xyz()))
        distance = math.sqrt(sum(v * v for v in delta))
        near_table = min(current_command.z, requested.z) - self.cfg.table_z_m <= self.cfg.near_table_distance_m
        speed_limit = self.cfg.near_table_speed_m_s if near_table else self.cfg.free_space_speed_m_s
        if distance <= 1e-12:
            desired_velocity = (0.0, 0.0, 0.0)
        else:
            desired_speed = min(distance / dt, speed_limit)
            desired_velocity = tuple(v / distance * desired_speed for v in delta)

        dv = tuple(desired_velocity[i] - last_velocity[i] for i in range(3))
        dv_norm = math.sqrt(sum(v * v for v in dv))
        max_dv = self.cfg.max_cartesian_accel_m_s2 * dt
        if dv_norm > max_dv > 0:
            dv = tuple(v / dv_norm * max_dv for v in dv)
        velocity = tuple(last_velocity[i] + dv[i] for i in range(3))
        step_delta = tuple(v * dt for v in velocity)
        if math.sqrt(sum(v * v for v in step_delta)) > distance and distance > 0:
            step_delta = delta
            velocity = tuple(v / dt for v in delta)
        rpy_delta = tuple(b - a for a, b in zip(current_command.rpy(), requested.rpy()))
        rpy_distance = math.sqrt(sum(v * v for v in rpy_delta))
        if rpy_distance <= 1e-12:
            desired_rpy_velocity = (0.0, 0.0, 0.0)
        else:
            desired_rpy_speed = min(rpy_distance / dt, self.cfg.max_rpy_speed_deg_s)
            desired_rpy_velocity = tuple(v / rpy_distance * desired_rpy_speed for v in rpy_delta)
        rpy_dv = tuple(desired_rpy_velocity[i] - last_rpy_velocity[i] for i in range(3))
        rpy_dv_norm = math.sqrt(sum(v * v for v in rpy_dv))
        max_rpy_dv = self.cfg.max_rpy_accel_deg_s2 * dt
        if rpy_dv_norm > max_rpy_dv > 0:
            rpy_dv = tuple(v / rpy_dv_norm * max_rpy_dv for v in rpy_dv)
        rpy_velocity = tuple(last_rpy_velocity[i] + rpy_dv[i] for i in range(3))
        rpy_step = tuple(v * dt for v in rpy_velocity)
        if math.sqrt(sum(v * v for v in rpy_step)) > rpy_distance and rpy_distance > 0:
            rpy_step = rpy_delta
            rpy_velocity = tuple(v / dt for v in rpy_delta)
        return current_command.moved_by(step_delta, rpy_step), velocity, rpy_velocity

    def _wait_for_tracking(
        self,
        target: EEPose,
        *,
        refresh_cartesian_target: bool = False,
    ) -> tuple[EEPose, float]:
        pose = self.robot.read_ee_pose()
        error = euclidean(target.xyz(), pose.xyz())
        if error <= self.cfg.tracking_hold_error_m:
            return pose, error
        deadline = time.monotonic() + self.cfg.tracking_settle_timeout_s
        while error > self.cfg.tracking_hold_error_m and time.monotonic() < deadline:
            # Piper vendor examples continuously refresh position targets.
            # Keep the checked EndPoseCtrl target alive while waiting for the
            # controller to converge; joint-IK execution retains its own
            # streamed JointCtrl target and does not enter this branch.
            if refresh_cartesian_target:
                self.robot.command_end_pose(
                    target,
                    speed_pct=self.cfg.speed_pct,
                    move_mode="L",
                )
            time.sleep(1.0 / self.cfg.control_hz)
            status_error = self._status_error(self.robot.read_arm_status())
            if status_error:
                return pose, max(error, self.cfg.tracking_abort_error_m + 1e-9)
            pose = self.robot.read_ee_pose()
            error = euclidean(target.xyz(), pose.xyz())
            if error > self.cfg.tracking_abort_error_m:
                break
        return pose, error

    def _workspace_error(self, pose: EEPose) -> Optional[str]:
        for axis, value, bounds in (
            ("x", pose.x, self.cfg.workspace_x_m),
            ("y", pose.y, self.cfg.workspace_y_m),
            ("z", pose.z, self.cfg.workspace_z_m),
        ):
            if not bounds[0] <= value <= bounds[1]:
                return f"{axis}={value:.6f} outside workspace {bounds}"
        if pose.z < self.cfg.min_z_m:
            return f"z={pose.z:.6f} below min_z_m={self.cfg.min_z_m:.6f}"
        floor_error = workspace_floor_error(pose.xyz(), self.cfg)
        if floor_error:
            return floor_error
        for plane in self.cfg.safety_planes:
            norm = math.sqrt(sum(value * value for value in plane.normal))
            if norm <= 1e-12:
                return f"safety plane {plane.name} has a zero normal"
            distance = sum(
                plane.normal[index] * (pose.xyz()[index] - plane.point[index])
                for index in range(3)
            ) / norm
            if distance < plane.margin_m:
                return (
                    f"safety plane {plane.name} violated: signed distance "
                    f"{distance:.6f} < margin {plane.margin_m:.6f}"
                )
        if self.cfg.tool_points_m:
            rot = _rpy_rotation_matrix(pose.rx, pose.ry, pose.rz)
            floor = self.cfg.table_z_m + self.cfg.table_margin_m
            for index, local in enumerate(self.cfg.tool_points_m):
                world = tuple(
                    pose.xyz()[row] + sum(rot[row][column] * local[column] for column in range(3))
                    for row in range(3)
                )
                world_z = world[2]
                if world_z < floor:
                    return f"tool point {index} z={world_z:.6f} below table clearance {floor:.6f}"
                floor_error = workspace_floor_error(world, self.cfg)
                if floor_error:
                    return f"tool point {index} {floor_error}"
        return None

    def _joint_state_error(self, joints: Optional[JointState]) -> Optional[str]:
        if joints is None:
            return "joint feedback is unavailable"
        for index, value in enumerate(joints.values_deg, start=1):
            bounds = self.cfg.joint_limits_deg[f"j{index}"]
            if not bounds[0] <= value <= bounds[1]:
                return f"j{index}={value:.3f} outside limits {bounds}"
        return None

    def _status_error(self, status: Dict[str, Any]) -> Optional[str]:
        if self.cfg.require_status_available and not status.get("available", False):
            return "arm status API is unavailable"
        enable_status = status.get("enable_status")
        if isinstance(enable_status, (list, tuple)) and enable_status and not all(enable_status):
            return f"arm motors are not all enabled: {enable_status}"
        if status.get("fault"):
            return "arm status reports fault"
        return None

    def _pose_drift_error(self, expected: EEPose, current: EEPose) -> Optional[str]:
        xyz_drift = euclidean(expected.xyz(), current.xyz())
        rpy_drift = max(abs(a - b) for a, b in zip(expected.rpy(), current.rpy()))
        if xyz_drift > self.cfg.max_start_pose_drift_m:
            return (
                "Preflight pose check failed: current XYZ drift "
                f"{xyz_drift:.6f} m > {self.cfg.max_start_pose_drift_m:.6f} m."
            )
        if rpy_drift > self.cfg.max_start_rpy_drift_deg:
            return (
                "Preflight pose check failed: current RPY drift "
                f"{rpy_drift:.6f} deg > {self.cfg.max_start_rpy_drift_deg:.6f} deg."
            )
        return None

    def _joint_drift_error(self, expected: JointState, current: JointState) -> Optional[str]:
        max_drift = max(abs(a - b) for a, b in zip(expected.values_deg, current.values_deg))
        if max_drift > self.cfg.max_start_joint_drift_deg:
            return (
                "Preflight joint check failed: current joint drift "
                f"{max_drift:.6f} deg > {self.cfg.max_start_joint_drift_deg:.6f} deg."
            )
        return None


def _fmt_xyz(pose: EEPose) -> str:
    return f"({pose.x:.6f}, {pose.y:.6f}, {pose.z:.6f})"
