from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .logging_utils import JsonlLogger
from .piper_adapter import RobotAdapter
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

        preflight_error = self._preflight_error(plan)
        if preflight_error:
            result["messages"].append(preflight_error)
            self._log("execution_blocked", {"reason": "preflight_failed", "message": preflight_error})
            return result

        for step in plan.steps:
            status = self.robot.read_arm_status()
            status_error = self._status_error(status)
            if status_error:
                result["messages"].append(f"Arm status fault before step {step.index}: {status_error}; {status}")
                self._log("execution_aborted", {"step": step.index, "status": status})
                return result

            if plan.action_mode == "joint_delta_deg":
                if step.target_joints is None:
                    result["messages"].append(f"Step {step.index} has no target_joints")
                    return result
                self.robot.command_joints(step.target_joints, speed_pct=self.cfg.speed_pct)
            else:
                if step.target_pose is None:
                    result["messages"].append(f"Step {step.index} has no target_pose")
                    return result
                self.robot.command_end_pose(step.target_pose, speed_pct=self.cfg.speed_pct, move_mode="L")

            if step.gripper_m is not None:
                self.robot.command_gripper(step.gripper_m, effort_n_m=self.cfg.gripper_effort_n_m)

            result["executed_steps"] += 1
            self._log("step_executed", {"step": step.to_dict(), "status_before": status})
            time.sleep(max(0.0, self.cfg.step_sleep_s))

        result["ok"] = True
        result["messages"].append(f"Executed {result['executed_steps']} steps.")
        self._log("execution_complete", result)
        return result

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

    def _status_error(self, status: Dict[str, Any]) -> Optional[str]:
        if self.cfg.require_status_available and not status.get("available", False):
            return "arm status API is unavailable"
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
