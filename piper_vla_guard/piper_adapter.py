from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, Optional, Protocol

from .types import EEPose, GripperState, JointState, SafetyConfig
from .utils import deg_to_sdk_angle, get_first_attr, m_to_sdk_pos, sdk_angle_to_deg, sdk_pos_to_m, unwrap_message


class RobotAdapter(Protocol):
    def connect(self) -> str: ...
    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str: ...
    def bootstrap_teaching_to_can(self, speed_pct: int = 10, timeout_s: float = 30.0) -> str: ...
    def read_ee_pose(self) -> EEPose: ...
    def read_joint_state(self) -> Optional[JointState]: ...
    def read_gripper_state(self) -> Optional[GripperState]: ...
    def read_arm_status(self) -> Dict[str, Any]: ...
    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None: ...
    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None: ...
    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None: ...
    def activate_cartesian_hold(self, speed_pct: int = 10, move_mode: str = "L") -> str: ...
    def hold_current_pose(self) -> str: ...
    def hold_current_joints(self) -> str: ...
    def pause_hold(self) -> str: ...
    def shutdown_at_safe_pose(self) -> str: ...
    def emergency_stop(self) -> str: ...


MOVE_MODE_TO_CODE = {
    "P": 0x00,
    "J": 0x01,
    "L": 0x02,
    "C": 0x03,
    "M": 0x04,
    "CPV": 0x05,
}


class ArmInitializationRefused(RuntimeError):
    """The adapter refused initialization before sending any control command."""


class ROSBridgeAdapter:
    """Robot adapter for the ROS-side safety filter.

    This process never imports Piper SDK and never opens CAN.  rosbridge carries
    feedback and checked targets; the ROS follower remains the sole CAN owner.
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.ros = None
        self._topics = []
        self._joint_state: Optional[JointState] = None
        self._pose: Optional[EEPose] = None
        self._gripper: Optional[GripperState] = None
        self._status: Dict[str, Any] = {"available": False, "fault": True, "armed": False}
        self._last_feedback_at = 0.0
        self._sequence = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def connect(self) -> str:
        try:
            import roslibpy  # type: ignore
        except Exception as exc:
            raise RuntimeError("ROS transport requires: uv sync --extra ros") from exc
        self._roslibpy = roslibpy
        self.ros = roslibpy.Ros(host=self.cfg.rosbridge_host, port=self.cfg.rosbridge_port)
        self.ros.run(timeout=5)
        if not self.ros.is_connected:
            raise RuntimeError("rosbridge connection failed; no Piper command was sent")

        joint_topic = roslibpy.Topic(self.ros, self.cfg.ros_joint_topic, "sensor_msgs/JointState")
        pose_topic = roslibpy.Topic(self.ros, self.cfg.ros_pose_topic, "sensor_msgs/JointState")
        status_topic = roslibpy.Topic(self.ros, self.cfg.ros_status_topic, "std_msgs/String")
        joint_topic.subscribe(self._on_joints)
        pose_topic.subscribe(self._on_pose)
        status_topic.subscribe(self._on_status)
        self._command_topic = roslibpy.Topic(
            self.ros, self.cfg.ros_command_topic, "piper_msgs/SafeJointCommand"
        )
        self._heartbeat_topic = roslibpy.Topic(
            self.ros, self.cfg.ros_heartbeat_topic, "std_msgs/Empty"
        )
        self._topics = [joint_topic, pose_topic, status_topic, self._command_topic, self._heartbeat_topic]
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._joint_state is not None and self._pose is not None:
                return "Connected to ROS safety filter (CAN remains owned by follower)"
            time.sleep(0.02)
        raise RuntimeError("ROS follower feedback was not received; no Piper command was sent")

    def _on_joints(self, message: Dict[str, Any]) -> None:
        values = message.get("position", [])
        if len(values) < 6:
            return
        self._joint_state = JointState(tuple(math.degrees(float(v)) for v in values[:6]))
        if len(values) >= 7:
            self._gripper = GripperState(float(values[6]))
        self._last_feedback_at = time.monotonic()

    def _on_pose(self, message: Dict[str, Any]) -> None:
        values = message.get("position", [])
        if len(values) < 6:
            return
        self._pose = EEPose(
            float(values[0]), float(values[1]), float(values[2]),
            math.degrees(float(values[3])), math.degrees(float(values[4])), math.degrees(float(values[5])),
        )

    def _on_status(self, message: Dict[str, Any]) -> None:
        try:
            payload = json.loads(str(message.get("data", "{}")))
        except (TypeError, ValueError):
            return
        self._status = {
            **payload,
            "available": bool(payload.get("feedback_fresh") and payload.get("pose_fresh")),
            "fault": bool(payload.get("reason")) and bool(payload.get("armed")),
            "ctrl_mode": 1.0 if payload.get("armed") else 0.0,
        }

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(0.05):
            try:
                self._heartbeat_topic.publish(self._roslibpy.Message({}))
            except Exception:
                return

    def _set_armed(self, enabled: bool) -> str:
        service = self._roslibpy.Service(self.ros, self.cfg.ros_arm_service, "std_srvs/SetBool")
        result = service.call(self._roslibpy.ServiceRequest({"data": bool(enabled)}), timeout=3)
        if not bool(result.get("success")):
            raise ArmInitializationRefused(str(result.get("message", "ROS filter refused arm")))
        return str(result.get("message", ""))

    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        del speed_pct, move_mode
        return self._set_armed(True)

    def read_ee_pose(self) -> EEPose:
        if self._pose is None:
            raise RuntimeError("ROS end-pose feedback is unavailable")
        return self._pose

    def read_joint_state(self) -> Optional[JointState]:
        return self._joint_state

    def read_gripper_state(self) -> Optional[GripperState]:
        return self._gripper

    def read_arm_status(self) -> Dict[str, Any]:
        return dict(self._status)

    def _publish_command(
        self, control_mode: int, joints: JointState, pose: EEPose, gripper_m: float, speed_pct: int
    ) -> None:
        self._sequence += 1
        payload = {
            "sequence": self._sequence,
            "control_mode": int(control_mode),
            "joints_rad": [math.radians(v) for v in joints.values_deg],
            "gripper_m": float(gripper_m),
            "target_pose_m_rad": [*pose.xyz(), *(math.radians(v) for v in pose.rpy())],
            "speed_pct": float(speed_pct),
        }
        self._command_topic.publish(self._roslibpy.Message(payload))

    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None:
        if move_mode.upper() != "L":
            raise RuntimeError("ROS safety bridge currently accepts Cartesian MoveL only")
        joints = self.read_joint_state()
        if joints is None:
            raise RuntimeError("joint feedback unavailable")
        gripper = 0.0 if self._gripper is None else self._gripper.opening_m
        self._publish_command(2, joints, pose, gripper, speed_pct)

    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None:
        pose = self._fk_pose(joints)
        gripper = 0.0 if self._gripper is None else self._gripper.opening_m
        self._publish_command(1, joints, pose, gripper, speed_pct)

    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None:
        del effort_n_m
        joints = self.read_joint_state()
        if joints is None:
            raise RuntimeError("joint feedback unavailable")
        self._publish_command(2, joints, self.read_ee_pose(), gripper_m, self.cfg.speed_pct)

    def _fk_pose(self, joints: JointState) -> EEPose:
        try:
            from piper_sdk import C_PiperForwardKinematics  # type: ignore
        except Exception as exc:
            raise RuntimeError("piper_sdk FK is required, but CAN is not opened by this adapter") from exc
        links = C_PiperForwardKinematics(self.cfg.dh_is_offset).CalFK(
            [math.radians(v) for v in joints.values_deg]
        )
        x_mm, y_mm, z_mm, rx, ry, rz = links[-1]
        return EEPose(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, rx, ry, rz)

    def activate_cartesian_hold(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        self.command_end_pose(self.read_ee_pose(), speed_pct, move_mode)
        return "ROS follower commanded to retain measured Cartesian pose"

    def hold_current_pose(self) -> str:
        self.command_end_pose(self.read_ee_pose(), self.cfg.speed_pct, "L")
        return "ROS follower commanded to retain measured pose"

    def hold_current_joints(self) -> str:
        joints = self.read_joint_state()
        if joints is None:
            raise RuntimeError("joint feedback unavailable for hold")
        self.command_joints(joints, self.cfg.speed_pct)
        return "ROS follower commanded to retain measured joints"

    def pause_hold(self) -> str:
        self._set_armed(False)
        return "VLA forwarding paused; follower retains its last valid target and motor torque"

    def emergency_stop(self) -> str:
        raise RuntimeError("ROS emergency-stop transport is not implemented; use the physical E-stop")

    def shutdown_at_safe_pose(self) -> str:
        self._require_shutdown_pose()
        return self._set_armed(False)

    def _require_shutdown_pose(self) -> None:
        expected = self.cfg.shutdown_joints_deg
        measured = self.read_joint_state()
        if expected is None:
            raise RuntimeError("shutdown_joints_deg is not configured; motor disable refused")
        if measured is None:
            raise RuntimeError("joint feedback unavailable; motor disable refused")
        failures = [
            f"J{i + 1}={actual:.3f}, expected {target:.3f}+/-{tol:.3f}deg"
            for i, (actual, target, tol) in enumerate(
                zip(measured.values_deg, expected, self.cfg.shutdown_joint_tolerance_deg)
            )
            if abs(actual - target) > tol
        ]
        if failures:
            raise RuntimeError("safe shutdown posture not verified: " + "; ".join(failures))


class PiperSDKAdapter:
    """Thin adapter over piper_sdk.C_PiperInterface_V2.

    The adapter keeps SDK-specific units and object field names out of the
    safety layer. It uses documented field candidates but still performs
    defensive introspection because piper_sdk message classes have changed over
    time.
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.piper: Any = None
        self.connected = False
        self._last_joint_state: Optional[JointState] = None
        self._last_joint_read_time: Optional[float] = None
        self._active_move_mode = "L"

    def connect(self) -> str:
        try:
            from piper_sdk import C_PiperInterface_V2  # type: ignore
            try:
                from piper_sdk import LogLevel  # type: ignore
            except Exception:  # pragma: no cover
                LogLevel = None  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "piper_sdk is not installed. Install with: pip install piper_sdk python-can"
            ) from exc

        kwargs = {
            "can_name": self.cfg.can_name,
            "judge_flag": self.cfg.judge_flag,
            "can_auto_init": self.cfg.can_auto_init,
            "dh_is_offset": self.cfg.dh_is_offset,
            "start_sdk_joint_limit": self.cfg.start_sdk_joint_limit,
            "start_sdk_gripper_limit": self.cfg.start_sdk_gripper_limit,
        }
        if LogLevel is not None:
            kwargs["logger_level"] = LogLevel.WARNING
        self.piper = C_PiperInterface_V2(**kwargs)
        self.piper.ConnectPort()
        self.connected = True
        self._wait_for_initial_feedback(timeout_s=2.0)
        return f"Connected to Piper on {self.cfg.can_name}"

    def _wait_for_initial_feedback(self, timeout_s: float) -> None:
        """Fail closed until all feedback streams have produced a real frame."""
        p = self._require()
        getters = ("GetArmStatus", "GetArmEndPoseMsgs", "GetArmJointMsgs")
        deadline = time.monotonic() + timeout_s
        missing = list(getters)
        while time.monotonic() < deadline:
            missing = []
            for name in getters:
                getter = getattr(p, name, None)
                if getter is None:
                    missing.append(name)
                    continue
                message = getter()
                timestamp = get_first_attr(message, ["time_stamp", "timestamp"])
                if timestamp is not None and float(timestamp) <= 0.0:
                    missing.append(name)
            if not missing:
                return
            time.sleep(0.02)
        raise RuntimeError(
            "Timed out waiting for fresh Piper feedback before control initialization: "
            + ", ".join(missing)
        )

    def _require(self) -> Any:
        if self.piper is None or not self.connected:
            raise RuntimeError("Piper is not connected")
        return self.piper

    def bootstrap_teaching_to_can(
        self,
        speed_pct: int = 10,
        timeout_s: float = 30.0,
    ) -> str:
        """Run the vendor-example first-start Teaching -> CAN/MoveJ sequence.

        This is intentionally separate from :meth:`enable`: it sends a real
        EmergencyStop(0x01), waits for the arm to damp down into the joint
        window used by ``test_ctrlPiperJoint_can0_2.py``, disables the motors,
        resumes with EmergencyStop(0x02), and enables again in CAN/MoveJ.
        Callers must obtain explicit operator approval before entering here.
        """
        p = self._require()
        status = self.read_arm_status()
        if status.get("arm_status") != 0.0 or status.get("ctrl_mode") != 2.0:
            raise ArmInitializationRefused(
                "Vendor Teaching bootstrap requires NORMAL TEACHING_MODE at entry; "
                f"no command sent: {status}"
            )
        if timeout_s <= 0.0:
            raise ValueError("Teaching bootstrap timeout must be positive")

        # Match the senior/vendor example: request enable before stopping out
        # of Teaching, then enter the real E-stop/damping state.
        initial_enable = self._enable_piper(timeout_s=min(3.0, timeout_s))
        if (
            not isinstance(initial_enable, (list, tuple))
            or len(initial_enable) != 6
            or not all(bool(value) for value in initial_enable)
        ):
            raise ArmInitializationRefused(
                "Teaching bootstrap could not enable all six motors before stop; "
                f"no E-stop command sent: enable_status={initial_enable}"
            )
        p.EmergencyStop(0x01)
        time.sleep(1.0)

        # Original limits are radians: |J2|,|J3| < 0.1745 and
        # 0.2094 < J5 < 0.7854. Feedback in this adapter is degrees.
        deadline = time.monotonic() + timeout_s
        safe_joints: Optional[JointState] = None
        while time.monotonic() < deadline:
            joints = self._read_joint_state_raw()
            if joints is not None:
                q = joints.values_deg
                if abs(q[1]) < 9.998 and abs(q[2]) < 9.998 and 11.998 < q[4] < 45.000:
                    safe_joints = joints
                    break
            time.sleep(0.01)
        if safe_joints is None:
            raise ArmInitializationRefused(
                "Teaching bootstrap timed out while waiting for the vendor safe joint window "
                "(|J2|<10deg, |J3|<10deg, 12deg<J5<45deg). Piper remains in the "
                "E-stop/damping state; physically support the arm and use the vendor recovery "
                "procedure. No disable or resume command was sent after the timeout."
            )

        # Piper.disable_arm() is exactly DisablePiper() followed by
        # EmergencyStop(0x02). Use the low-level calls because this adapter
        # owns C_PiperInterface_V2 directly.
        if hasattr(p, "DisablePiper"):
            p.DisablePiper()
        elif hasattr(p, "DisableArm"):
            p.DisableArm(7, 0x01)
        else:
            raise RuntimeError("Piper SDK has no DisablePiper/DisableArm API")
        p.EmergencyStop(0x02)
        time.sleep(1.0)

        # Match the example's mode loop, with a bounded timeout and feedback
        # checks. Keep the freshly measured folded position as the first
        # JointCtrl target so no older retained target is reused on enable.
        mode_deadline = time.monotonic() + timeout_s
        hold_joints = self._read_joint_state_raw()
        if hold_joints is None:
            raise ArmInitializationRefused(
                "Joint feedback was unavailable after Teaching bootstrap resume; "
                "CAN transition was not attempted."
            )
        joint_target = tuple(deg_to_sdk_angle(value) for value in hold_joints.values_deg)
        while time.monotonic() < mode_deadline:
            if hasattr(p, "MotionCtrl_2"):
                p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            elif hasattr(p, "ModeCtrl"):
                p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            p.JointCtrl(*joint_target)
            mode_status = self.read_arm_status()
            if mode_status.get("ctrl_mode") == 1.0:
                break
            time.sleep(0.01)
        else:
            raise ArmInitializationRefused(
                "Teaching bootstrap could not enter CAN_CTRL before timeout; "
                f"last status={self.read_arm_status()}"
            )

        enable_deadline = time.monotonic() + min(3.0, timeout_s)
        enabled: Any = None
        while time.monotonic() < enable_deadline:
            p.JointCtrl(*joint_target)
            if hasattr(p, "EnablePiper"):
                p.EnablePiper()
            elif hasattr(p, "EnableArm"):
                p.EnableArm(7, 0x02)
            if hasattr(p, "MotionCtrl_2"):
                p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            elif hasattr(p, "ModeCtrl"):
                p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            p.JointCtrl(*joint_target)
            enabled = self._read_enable_status()
            final_status = self.read_arm_status()
            if (
                isinstance(enabled, (list, tuple))
                and len(enabled) == 6
                and all(bool(value) for value in enabled)
                and final_status.get("arm_status") == 0.0
                and final_status.get("ctrl_mode") == 1.0
                and final_status.get("mode_feed") == 1.0
            ):
                self._active_move_mode = "J"
                return (
                    "VENDOR TEACHING BOOTSTRAP COMPLETE: stop/damping safe window verified, "
                    "DisablePiper+resume completed, CAN/MoveJ and all six motors verified; "
                    f"initial_enable={initial_enable}, hold_joints={hold_joints.as_list()}"
                )
            time.sleep(0.01)
        raise ArmInitializationRefused(
            "Teaching bootstrap did not reach NORMAL CAN/MoveJ with all six motors enabled; "
            f"status={self.read_arm_status()}, enable_status={enabled}"
        )

    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        p = self._require()
        # Capture the current Cartesian pose before changing control mode. A
        # stale target retained in Piper can otherwise be applied as soon as
        # the motors enable.
        hold_pose = self.read_ee_pose()
        hold_joints = self.read_joint_state()
        status = self.read_arm_status()
        if status.get("arm_status") not in (None, 0, 0.0):
            raise ArmInitializationRefused(
                "Refusing to initialize control while Piper arm_status is not NORMAL: "
                f"{status}. If an emergency stop is latched, recover it explicitly with the "
                "vendor procedure while physically supporting the arm; this application never "
                "auto-resumes an E-stop."
            )
        if status.get("ctrl_mode") == 2.0:
            raise ArmInitializationRefused(
                "Refusing automatic takeover from TEACHING_MODE. Switching enabled motors from "
                "teaching to CAN control can apply a retained JointCtrl target. Physically support "
                "the arm and use the vendor-approved recovery procedure; this application sends "
                "no Teaching-exit, reset/resume, or disable command from this state."
            )
        enable_status = self._read_enable_status()
        if (
            self.cfg.sdk_attach_enabled_can
            and isinstance(enable_status, (list, tuple))
            and any(bool(value) for value in enable_status)
        ):
            attach_message = self._attach_enabled_can_hold(
                status=status,
                enable_status=enable_status,
                hold_pose=hold_pose,
                hold_joints=hold_joints,
                speed_pct=speed_pct,
                move_mode="J",
            )
            if move_mode.upper() == "J":
                return attach_message
            if move_mode.upper() not in ("L", "P"):
                raise ArmInitializationRefused(
                    f"Enabled-CAN Cartesian transition supports MoveL/MoveP only, got {move_mode}."
                )
            cartesian_message = self.activate_cartesian_hold(
                speed_pct=speed_pct,
                move_mode=move_mode,
            )
            final_status = self.read_arm_status()
            expected_mode = float(MOVE_MODE_TO_CODE[move_mode.upper()])
            final_enable = self._read_enable_status()
            if (
                final_status.get("fault")
                or final_status.get("ctrl_mode") != 1.0
                or final_status.get("mode_feed") != expected_mode
                or not isinstance(final_enable, (list, tuple))
                or len(final_enable) != 6
                or not all(bool(value) for value in final_enable)
            ):
                hold_message = "no additional hold command sent because Piper reported a fault"
                if not final_status.get("fault"):
                    hold_message = self.pause_hold()
                raise ArmInitializationRefused(
                    "Measured-pose Cartesian transition verification failed: "
                    f"status={final_status}, enable_status={final_enable}; {hold_message}"
                )
            return attach_message + "; " + cartesian_message
        if status.get("ctrl_mode") not in (0, 0.0):
            raise ArmInitializationRefused(
                f"Initialization requires STANDBY ctrl_mode=0 before motor enable; got "
                f"ctrl_mode={status.get('ctrl_mode')} ({status.get('ctrl_mode_name')})."
            )
        if (
            isinstance(enable_status, (list, tuple))
            and any(bool(value) for value in enable_status)
        ):
            raise ArmInitializationRefused(
                "Refusing to initialize control while one or more motors are already enabled. "
                "No motion, stop, mode, or standby command was sent. The previous controller target may "
                "still be active; support the arm, recover it with the vendor procedure, and retry from "
                "a physically clear pose. Piper has no fixed startup pose."
            )
        if move_mode.upper() == "J" and hold_joints is None:
            raise ArmInitializationRefused("Cannot prime joint control: joint feedback is unavailable.")
        code = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])

        # Follow the installed vendor demos: enable every motor while the arm
        # is still in STANDBY. Never select CAN/MoveJ before enabling, because
        # that can activate a retained controller target.
        if hasattr(p, "MotionCtrl_1"):
            for track_ctrl in (0x06, 0x04):
                p.MotionCtrl_1(0x00, track_ctrl, 0x00)
                time.sleep(0.02)
        enabled = self._enable_piper(timeout_s=3.0)
        if not isinstance(enabled, (list, tuple)) or not enabled or not all(bool(v) for v in enabled):
            raise RuntimeError(f"Piper motor enable did not complete in STANDBY: {enabled}")

        # Verify that enable alone did not move the arm before selecting a
        # command mode. No JointCtrl/EndPoseCtrl has been sent at this point.
        standby_samples = max(10, int(round(0.5 * max(1.0, self.cfg.control_hz))))
        for _ in range(standby_samples):
            time.sleep(1.0 / max(1.0, self.cfg.control_hz))
            standby_status = self.read_arm_status()
            if standby_status.get("fault") or standby_status.get("ctrl_mode") != 0.0:
                raise RuntimeError(
                    f"Piper left healthy STANDBY during motor enable: {standby_status}; "
                    "no automatic E-stop was sent because it can remove gravity hold"
                )
            after_pose = self.read_ee_pose()
            after_joints = self.read_joint_state()
            ee_drift = sum((a - b) ** 2 for a, b in zip(after_pose.xyz(), hold_pose.xyz())) ** 0.5
            joint_drift = (
                None
                if hold_joints is None or after_joints is None
                else max(abs(a - b) for a, b in zip(after_joints.values_deg, hold_joints.values_deg))
            )
            if ee_drift > self.cfg.max_start_pose_drift_m or (
                joint_drift is not None and joint_drift > self.cfg.max_start_joint_drift_deg
            ):
                raise RuntimeError(
                    "Piper drifted while enabling in STANDBY before any motion target: "
                    f"ee={ee_drift:.6f}m, joints={joint_drift}; no automatic E-stop was sent"
                )

        # Seed the measured target in standby, select CAN control, then publish
        # it again immediately without the old 10 ms mode/target gap.
        for _ in range(3):
            self._send_hold_target(hold_pose, hold_joints, move_mode)
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, code, int(speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, code, int(speed_pct), 0x00)
        for _ in range(3):
            self._send_hold_target(hold_pose, hold_joints, move_mode)

        # Continue publishing and monitor long enough for delayed feedback and
        # retained-target effects to become observable.
        verify_samples = max(20, int(round(1.0 * max(1.0, self.cfg.control_hz))))
        max_ee_drift = 0.0
        max_joint_drift = 0.0
        for _ in range(verify_samples):
            self._send_hold_target(hold_pose, hold_joints, move_mode)
            time.sleep(1.0 / max(1.0, self.cfg.control_hz))
            control_status = self.read_arm_status()
            if control_status.get("fault") or control_status.get("ctrl_mode") != 1.0:
                stop_message = self.pause_hold()
                raise RuntimeError(
                    f"Piper did not remain in healthy CAN control during hold: {control_status}; "
                    f"{stop_message}"
                )
            after_pose = self.read_ee_pose()
            ee_drift = sum((a - b) ** 2 for a, b in zip(after_pose.xyz(), hold_pose.xyz())) ** 0.5
            max_ee_drift = max(max_ee_drift, ee_drift)
            after_joints = self.read_joint_state()
            joint_drift = (
                None
                if hold_joints is None or after_joints is None
                else max(abs(a - b) for a, b in zip(after_joints.values_deg, hold_joints.values_deg))
            )
            if joint_drift is not None:
                max_joint_drift = max(max_joint_drift, joint_drift)
            if ee_drift > self.cfg.max_start_pose_drift_m or (
                joint_drift is not None and joint_drift > self.cfg.max_start_joint_drift_deg
            ):
                stop_message = self.pause_hold()
                raise RuntimeError(
                    "Piper drifted while establishing measured-position hold: "
                    f"ee={ee_drift:.6f}m, joints={joint_drift}; {stop_message}"
                )
        if move_mode.upper() == "J":
            self._active_move_mode = "J"
            return (
                f"Enabled in STANDBY then selected mode=J, arm={enabled}, speed={speed_pct}%, "
                f"primed hold joints={hold_joints.as_list()}, "
                f"verified drift<={max_ee_drift:.6f}m/{max_joint_drift:.3f}deg"
            )
        self._active_move_mode = move_mode.upper()
        return (
            f"Enabled arm={enabled}, mode={move_mode}, speed={speed_pct}%, "
            f"primed hold XYZ=({hold_pose.x:.6f}, {hold_pose.y:.6f}, {hold_pose.z:.6f})"
        )

    def _attach_enabled_can_hold(
        self,
        *,
        status: Dict[str, Any],
        enable_status: Any,
        hold_pose: EEPose,
        hold_joints: Optional[JointState],
        speed_pct: int,
        move_mode: str,
    ) -> str:
        """Attach to the vendor demo's active CAN/MoveJ hold without reset.

        This path intentionally mirrors Piper.move_j(): ModeCtrl/JointCtrl are
        refreshed together. It never calls EnablePiper, MotionCtrl_1,
        EmergencyStop(resume), or DisablePiper.
        """
        if move_mode.upper() != "J":
            raise ArmInitializationRefused(
                "Enabled-CAN attach is allowed only for checked joint_ik/MoveJ control."
            )
        if status.get("ctrl_mode") != 1.0 or status.get("mode_feed") != 1.0:
            raise ArmInitializationRefused(
                "Enabled-CAN attach requires ctrl_mode=CAN_CTRL and mode_feed=MOVE_J; "
                f"got {status}. No reset, resume, mode, or target command was sent."
            )
        if not isinstance(enable_status, (list, tuple)) or len(enable_status) != 6:
            raise ArmInitializationRefused(
                "Enabled-CAN attach requires six-axis enable feedback; "
                f"got {enable_status}. No command was sent."
            )
        if hold_joints is None:
            raise ArmInitializationRefused(
                "Enabled-CAN attach requires fresh joint feedback; no command was sent."
            )

        p = self._require()
        joint_target = tuple(deg_to_sdk_angle(value) for value in hold_joints.values_deg)
        # The controller is already in CAN/MoveJ, so first overwrite the
        # retained target with measured joints before refreshing ModeCtrl.
        for _ in range(3):
            p.JointCtrl(*joint_target)
            time.sleep(0.02)

        restored_enable = not all(bool(value) for value in enable_status)
        if restored_enable:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if hasattr(p, "MotionCtrl_2"):
                    p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
                elif hasattr(p, "ModeCtrl"):
                    p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
                p.JointCtrl(*joint_target)
                if hasattr(p, "EnablePiper"):
                    p.EnablePiper()
                elif hasattr(p, "EnableArm"):
                    p.EnableArm(7, 0x02)
                p.JointCtrl(*joint_target)
                current_enable = self._read_enable_status()
                if (
                    isinstance(current_enable, (list, tuple))
                    and len(current_enable) == 6
                    and all(bool(value) for value in current_enable)
                ):
                    break
                time.sleep(0.02)
            else:
                raise ArmInitializationRefused(
                    "Enabled-CAN attach could not restore all six motors while continuously "
                    "holding measured joints. No reset or E-stop resume was sent."
                )

        verify_samples = max(20, int(round(1.0 * max(1.0, self.cfg.control_hz))))
        max_ee_drift = 0.0
        max_joint_drift = 0.0
        for _ in range(verify_samples):
            if hasattr(p, "MotionCtrl_2"):
                p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            elif hasattr(p, "ModeCtrl"):
                p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
            p.JointCtrl(*joint_target)
            time.sleep(1.0 / max(1.0, self.cfg.control_hz))

            current_status = self.read_arm_status()
            if (
                current_status.get("fault")
                or current_status.get("ctrl_mode") != 1.0
                or current_status.get("mode_feed") != 1.0
            ):
                raise ArmInitializationRefused(
                    "Piper left healthy CAN/MoveJ during attach hold: "
                    f"{current_status}. No reset or E-stop resume was sent."
                )
            after_pose = self.read_ee_pose()
            after_joints = self.read_joint_state()
            ee_drift = sum((a - b) ** 2 for a, b in zip(after_pose.xyz(), hold_pose.xyz())) ** 0.5
            joint_drift = (
                float("inf")
                if after_joints is None
                else max(abs(a - b) for a, b in zip(after_joints.values_deg, hold_joints.values_deg))
            )
            max_ee_drift = max(max_ee_drift, ee_drift)
            max_joint_drift = max(max_joint_drift, joint_drift)
            if (
                ee_drift > self.cfg.max_start_pose_drift_m
                or joint_drift > self.cfg.max_start_joint_drift_deg
            ):
                raise ArmInitializationRefused(
                    "Piper drifted while attaching to measured-joint hold: "
                    f"ee={ee_drift:.6f}m, joints={joint_drift:.3f}deg. "
                    "No reset or E-stop resume was sent."
                )

        self._active_move_mode = "J"
        return (
            "Attached to already-enabled CAN/MoveJ hold using vendor-style "
            "ModeCtrl+JointCtrl; no reset/resume/disable sent; "
            f"restored_missing_enable={restored_enable}; "
            f"joints={hold_joints.as_list()}, "
            f"verified drift<={max_ee_drift:.6f}m/{max_joint_drift:.3f}deg"
        )

    def _send_hold_target(
        self,
        pose: EEPose,
        joints: Optional[JointState],
        move_mode: str,
    ) -> None:
        p = self._require()
        if move_mode.upper() == "J":
            if joints is None:
                raise RuntimeError("Joint hold target is unavailable")
            p.JointCtrl(*[deg_to_sdk_angle(v) for v in joints.values_deg])
            return
        p.EndPoseCtrl(
            m_to_sdk_pos(pose.x),
            m_to_sdk_pos(pose.y),
            m_to_sdk_pos(pose.z),
            deg_to_sdk_angle(pose.rx),
            deg_to_sdk_angle(pose.ry),
            deg_to_sdk_angle(pose.rz),
        )

    def _enable_piper(self, timeout_s: float = 3.0) -> Any:
        p = self._require()
        deadline = time.time() + timeout_s
        last_status: Any = None
        while time.time() < deadline:
            if hasattr(p, "EnablePiper"):
                last_status = p.EnablePiper()
                if last_status:
                    return self._read_enable_status()
            elif hasattr(p, "EnableArm"):
                p.EnableArm(7, 0x02)
                last_status = self._read_enable_status()
                if isinstance(last_status, (list, tuple)) and all(last_status):
                    return last_status
            time.sleep(0.01)
        return last_status if last_status is not None else self._read_enable_status()

    def _read_enable_status(self) -> Any:
        p = self._require()
        if hasattr(p, "GetArmEnableStatus"):
            return p.GetArmEnableStatus()
        return None

    def read_ee_pose(self) -> EEPose:
        p = self._require()
        msg = p.GetArmEndPoseMsgs()
        payload = unwrap_message(msg, ["end_pose", "pose", "arm_end_pose"])
        x = _required_number(payload, ["X_axis", "x_axis", "X", "x"])
        y = _required_number(payload, ["Y_axis", "y_axis", "Y", "y"])
        z = _required_number(payload, ["Z_axis", "z_axis", "Z", "z"])
        rx = _required_number(payload, ["RX_axis", "rx_axis", "RX", "rx", "roll"])
        ry = _required_number(payload, ["RY_axis", "ry_axis", "RY", "ry", "pitch"])
        rz = _required_number(payload, ["RZ_axis", "rz_axis", "RZ", "rz", "yaw"])
        return EEPose(
            x=sdk_pos_to_m(x),
            y=sdk_pos_to_m(y),
            z=sdk_pos_to_m(z),
            rx=sdk_angle_to_deg(rx),
            ry=sdk_angle_to_deg(ry),
            rz=sdk_angle_to_deg(rz),
        )

    def read_joint_state(self) -> Optional[JointState]:
        candidate = self._read_joint_state_raw()
        if candidate is None:
            return None
        now = time.monotonic()
        if self._last_joint_state is None or self._last_joint_read_time is None:
            samples = [candidate]
            for _ in range(4):
                time.sleep(0.015)
                sample = self._read_joint_state_raw()
                if sample is not None:
                    samples.append(sample)
            candidate = JointState(
                tuple(
                    sorted(sample.values_deg[index] for sample in samples)[len(samples) // 2]
                    for index in range(6)
                )
            )
        else:
            elapsed = max(1e-3, now - self._last_joint_read_time)
            apparent_speed = max(
                abs(a - b) / elapsed
                for a, b in zip(candidate.values_deg, self._last_joint_state.values_deg)
            )
            # Piper's combined joint feedback can briefly contain zeros from
            # one CAN sub-frame while the other sub-frame is already fresh.
            # Confirm only physically implausible jumps; do not smooth normal
            # motion or hide a consistently confirmed overspeed.
            feedback_glitch_speed = max(180.0, 4.0 * self.cfg.max_measured_joint_speed_deg_s)
            if apparent_speed > feedback_glitch_speed:
                samples = [candidate]
                for _ in range(4):
                    time.sleep(0.015)
                    sample = self._read_joint_state_raw()
                    if sample is not None:
                        samples.append(sample)
                candidate = min(
                    samples,
                    key=lambda sample: max(
                        abs(a - b)
                        for a, b in zip(sample.values_deg, self._last_joint_state.values_deg)
                    ),
                )
        self._last_joint_state = candidate
        self._last_joint_read_time = time.monotonic()
        return candidate

    def _read_joint_state_raw(self) -> Optional[JointState]:
        p = self._require()
        if not hasattr(p, "GetArmJointMsgs"):
            return None
        msg = p.GetArmJointMsgs()
        payload = unwrap_message(msg, ["joint_state", "joint_states", "arm_joint", "joints"])
        vals = []
        for i in range(1, 7):
            raw = get_first_attr(payload, [f"joint_{i}", f"joint{i}", f"j{i}"])
            if raw is None:
                return None
            vals.append(sdk_angle_to_deg(float(raw)))
        return JointState(tuple(vals))  # type: ignore[arg-type]

    def read_gripper_state(self) -> Optional[GripperState]:
        p = self._require()
        if not hasattr(p, "GetArmGripperMsgs"):
            return None
        msg = p.GetArmGripperMsgs()
        payload = unwrap_message(msg, ["gripper_state", "gripper_feedback", "gripper"])
        raw_angle = get_first_attr(payload, ["grippers_angle", "gripper_angle", "angle", "opening"])
        if raw_angle is None:
            return None
        raw_effort = get_first_attr(payload, ["grippers_effort", "gripper_effort", "effort"])
        raw_status = get_first_attr(payload, ["status_code", "foc_status", "status"])
        return GripperState(
            opening_m=sdk_pos_to_m(float(raw_angle)),
            effort_n_m=None if raw_effort is None else float(raw_effort) / 1000.0,
            status_code=None if raw_status is None else float(raw_status),
        )

    def read_arm_status(self) -> Dict[str, Any]:
        p = self._require()
        if not hasattr(p, "GetArmStatus"):
            return {"available": False}
        msg = p.GetArmStatus()
        payload = unwrap_message(msg, ["arm_status", "status"])
        fields = {
            "available": True,
            "ctrl_mode": _optional_number(payload, ["ctrl_mode", "control_mode"]),
            "arm_status": _optional_number(payload, ["arm_status", "status", "arm_status_code"]),
            "mode_feed": _optional_number(payload, ["mode_feed", "mode"]),
            "motion_status": _optional_number(payload, ["motion_status"]),
            "trajectory_num": _optional_number(payload, ["trajectory_num"]),
            "enable_status": self._read_enable_status(),
        }
        fields["ctrl_mode_name"] = {
            0.0: "STANDBY",
            1.0: "CAN_CTRL",
            2.0: "TEACHING_MODE",
            3.0: "ETHERNET_CTRL",
            4.0: "WIFI_CTRL",
            5.0: "REMOTE_CTRL",
            6.0: "LINKAGE_TEACHING",
            7.0: "OFFLINE_TRAJECTORY",
        }.get(fields["ctrl_mode"], "UNKNOWN")
        fields["arm_status_name"] = {
            0.0: "NORMAL",
            1.0: "EMERGENCY_STOP",
            2.0: "NO_SOLUTION",
            3.0: "SINGULARITY_POINT",
            4.0: "TARGET_POS_EXCEEDS_LIMIT",
            5.0: "JOINT_COMMUNICATION_ERR",
            6.0: "JOINT_BRAKE_NOT_RELEASED",
            7.0: "COLLISION_OCCURRED",
            8.0: "TEACHING_OVERSPEED",
            9.0: "JOINT_STATUS_ERR",
            10.0: "OTHER_ERR",
        }.get(fields["arm_status"], "UNKNOWN")
        fields["mode_feed_name"] = {
            0.0: "MOVE_P",
            1.0: "MOVE_J",
            2.0: "MOVE_L",
            3.0: "MOVE_C",
            4.0: "MOVE_M",
            5.0: "MOVE_CPV",
        }.get(fields["mode_feed"], "UNKNOWN")
        err_obj = get_first_attr(payload, ["err_status", "error_status", "err"])
        fields["err_status"] = _err_status_to_dict(err_obj)
        code = fields["arm_status"]
        fields["fault"] = bool(code not in (None, 0, 0.0)) or _err_status_has_fault(err_obj)
        return fields

    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None:
        p = self._require()
        code = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])
        if hasattr(p, "MotionCtrl_1"):
            p.MotionCtrl_1(0x00, 0x00, 0x00)
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, code, int(speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, code, int(speed_pct), 0x00)
        p.EndPoseCtrl(
            m_to_sdk_pos(pose.x),
            m_to_sdk_pos(pose.y),
            m_to_sdk_pos(pose.z),
            deg_to_sdk_angle(pose.rx),
            deg_to_sdk_angle(pose.ry),
            deg_to_sdk_angle(pose.rz),
        )
        self._active_move_mode = move_mode.upper()

    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None:
        p = self._require()
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(speed_pct), 0x00)
        p.JointCtrl(*[deg_to_sdk_angle(v) for v in joints.values_deg])
        self._active_move_mode = "J"

    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None:
        p = self._require()
        p.GripperCtrl(
            m_to_sdk_pos(gripper_m),
            int(round(effort_n_m * 1000.0)),
            0x01,
            0x00,
        )

    def hold_current_pose(self) -> str:
        """Replace the active Cartesian target with measured pose and keep position control active."""
        p = self._require()
        pose = self.read_ee_pose()
        for _ in range(5):
            p.EndPoseCtrl(
                m_to_sdk_pos(pose.x),
                m_to_sdk_pos(pose.y),
                m_to_sdk_pos(pose.z),
                deg_to_sdk_angle(pose.rx),
                deg_to_sdk_angle(pose.ry),
                deg_to_sdk_angle(pose.rz),
            )
            time.sleep(0.02)
        return (
            "ACTIVE HOLD: measured pose registered as target; motors remain enabled at "
            f"XYZ=({pose.x:.6f}, {pose.y:.6f}, {pose.z:.6f})"
        )

    def hold_current_joints(self) -> str:
        """Replace the retained MoveJ target with measured joints."""
        p = self._require()
        joints = self.read_joint_state()
        if joints is None:
            raise RuntimeError("joint feedback unavailable for hold")
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(self.cfg.speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(self.cfg.speed_pct), 0x00)
        for _ in range(5):
            p.JointCtrl(*[deg_to_sdk_angle(value) for value in joints.values_deg])
            time.sleep(0.02)
        self._active_move_mode = "J"
        return f"ACTIVE JOINT HOLD: measured joints registered as target {joints.as_list()}"

    def activate_cartesian_hold(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        """Prime measured pose, switch an enabled arm to Cartesian control, and verify drift."""
        p = self._require()
        pose = self.read_ee_pose()
        for _ in range(3):
            self._send_hold_target(pose, None, move_mode)
            time.sleep(0.02)
        code = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, code, int(speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, code, int(speed_pct), 0x00)
        for _ in range(5):
            self._send_hold_target(pose, None, move_mode)
            time.sleep(0.02)
        self._active_move_mode = move_mode.upper()
        after = self.read_ee_pose()
        drift = sum((a - b) ** 2 for a, b in zip(pose.xyz(), after.xyz())) ** 0.5
        if drift > self.cfg.max_start_pose_drift_m:
            stop_message = self.pause_hold()
            raise RuntimeError(
                f"Cartesian hold transition drifted {drift:.6f} m; {stop_message}"
            )
        return (
            f"Cartesian {move_mode} hold active at XYZ=({after.x:.6f}, {after.y:.6f}, {after.z:.6f}); "
            f"transition drift={drift:.6f}m"
        )

    def emergency_stop(self) -> str:
        """Enter Piper's real E-stop state; this may let the arm descend."""
        p = self._require()
        p.EmergencyStop(0x01)
        acknowledged = self._wait_for_status_code(0x01, timeout_s=0.5)
        ack_text = (
            "E-stop acknowledged by arm status"
            if acknowledged is True
            else "WARNING: E-stop acknowledgement was not observed"
            if acknowledged is False
            else "E-stop acknowledgement unavailable"
        )
        return (
            "REAL EMERGENCY STOP sent (EmergencyStop(0x01)); Piper may descend under damping; "
            + ack_text
        )

    def _wait_for_status_code(self, expected: int, timeout_s: float) -> Optional[bool]:
        p = self._require()
        if not hasattr(p, "GetArmStatus"):
            return None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                status = self.read_arm_status()
            except Exception:  # pragma: no cover - hardware dependent
                return None
            if status.get("arm_status") == float(expected):
                return True
            time.sleep(0.02)
        return False

    def pause_hold(self) -> str:
        """Pause VLA execution by holding measured joints; never enter E-stop/reset/disable."""
        p = self._require()
        status = self.read_arm_status()
        if status.get("arm_status") != 0.0 or status.get("ctrl_mode") != 1.0:
            raise ArmInitializationRefused(
                "Pause/hold requires NORMAL CAN control; no command sent: " + str(status)
            )
        enabled = self._read_enable_status()
        if not isinstance(enabled, (list, tuple)) or not all(bool(value) for value in enabled):
            raise ArmInitializationRefused(
                f"Pause/hold requires all six motors enabled; no command sent: {enabled}"
            )
        joints = self.read_joint_state()
        if joints is None:
            raise ArmInitializationRefused("Pause/hold requires fresh joint feedback; no command sent")
        target = [deg_to_sdk_angle(value) for value in joints.values_deg]
        # Prime the retained MoveJ target before selecting MoveJ, then refresh it
        # immediately. No CAN 0x150 stop/reset/trajectory command is used.
        for _ in range(3):
            p.JointCtrl(*target)
        if hasattr(p, "MotionCtrl_2"):
            p.MotionCtrl_2(0x01, MOVE_MODE_TO_CODE["J"], int(self.cfg.speed_pct), 0x00)
        elif hasattr(p, "ModeCtrl"):
            p.ModeCtrl(0x01, MOVE_MODE_TO_CODE["J"], int(self.cfg.speed_pct), 0x00)
        for _ in range(5):
            p.JointCtrl(*target)
            time.sleep(0.02)
        self._active_move_mode = "J"
        return (
            "VLA PAUSE/HOLD: policy queue discarded by caller; measured JointCtrl target retained; "
            "no EmergencyStop, reset, trajectory-stop, or motor-disable command sent"
        )

    def shutdown_at_safe_pose(self) -> str:
        """Disable motors only when the configured shutdown posture is verified."""
        p = self._require()
        expected = self.cfg.shutdown_joints_deg
        measured = self.read_joint_state()
        if expected is None:
            raise RuntimeError("shutdown_joints_deg is not configured; motor disable refused")
        if measured is None:
            raise RuntimeError("joint feedback unavailable; motor disable refused")
        failures = [
            f"J{i + 1}={actual:.3f}, expected {target:.3f}+/-{tol:.3f}deg"
            for i, (actual, target, tol) in enumerate(
                zip(measured.values_deg, expected, self.cfg.shutdown_joint_tolerance_deg)
            )
            if abs(actual - target) > tol
        ]
        if failures:
            raise RuntimeError("safe shutdown posture not verified: " + "; ".join(failures))
        self.pause_hold()
        for _ in range(5):
            if hasattr(p, "DisablePiper"):
                p.DisablePiper()
            elif hasattr(p, "DisableArm"):
                p.DisableArm(7, 0x01)
            else:
                raise RuntimeError("Piper SDK has no DisablePiper/DisableArm API")
            time.sleep(0.02)
            enabled = self._read_enable_status()
            if isinstance(enabled, (list, tuple)) and not any(bool(value) for value in enabled):
                return "NORMAL SHUTDOWN: configured safe posture verified; all motors disabled"
        raise RuntimeError(f"motor-disable feedback did not clear all axes: {enabled}")


class MockPiperAdapter:
    """A deterministic mock adapter for UI dry-run and tests."""

    def __init__(self, cfg: Optional[SafetyConfig] = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self.connected = False
        self.pose = EEPose(0.30, 0.00, 0.20, 0.0, 0.0, 0.0)
        self.joints = JointState((0.0, 60.0, -80.0, 0.0, 20.0, 0.0))
        self.gripper_m = 0.070
        self.status: Dict[str, Any] = {
            "available": True,
            "ctrl_mode": 1,
            "arm_status": 0,
            "mode_feed": 2,
            "motion_status": 0,
            "trajectory_num": 0,
            "err_status": None,
            "fault": False,
        }

    def connect(self) -> str:
        self.connected = True
        return "Connected to mock Piper"

    def enable(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        self.status["enable_status"] = [True] * 6
        return f"Mock enabled, mode={move_mode}, speed={speed_pct}%"

    def bootstrap_teaching_to_can(self, speed_pct: int = 10, timeout_s: float = 30.0) -> str:
        del timeout_s
        if self.status.get("ctrl_mode") != 2 or self.status.get("arm_status") != 0:
            raise ArmInitializationRefused("Mock Teaching bootstrap requires NORMAL TEACHING_MODE")
        self.status.update(
            ctrl_mode=1,
            arm_status=0,
            mode_feed=1,
            enable_status=[True] * 6,
        )
        return f"Mock Teaching bootstrap complete, speed={speed_pct}%"

    def read_ee_pose(self) -> EEPose:
        return self.pose

    def read_joint_state(self) -> Optional[JointState]:
        return self.joints

    def read_gripper_state(self) -> Optional[GripperState]:
        return GripperState(opening_m=self.gripper_m, effort_n_m=0.0, status_code=0.0)

    def read_arm_status(self) -> Dict[str, Any]:
        return dict(self.status)

    def command_end_pose(self, pose: EEPose, speed_pct: int = 10, move_mode: str = "L") -> None:
        self.pose = pose
        self.status["trajectory_num"] = int(self.status.get("trajectory_num", 0)) + 1

    def command_joints(self, joints: JointState, speed_pct: int = 10) -> None:
        self.joints = joints
        self.status["trajectory_num"] = int(self.status.get("trajectory_num", 0)) + 1

    def command_gripper(self, gripper_m: float, effort_n_m: float) -> None:
        self.gripper_m = float(gripper_m)
        self.status["last_gripper_m"] = gripper_m
        self.status["last_gripper_effort_n_m"] = effort_n_m

    def hold_current_pose(self) -> str:
        return f"Mock active hold XYZ={self.pose.xyz()}"

    def hold_current_joints(self) -> str:
        return f"Mock active joint hold={self.joints.as_list()}"

    def activate_cartesian_hold(self, speed_pct: int = 10, move_mode: str = "L") -> str:
        self.status["ctrl_mode"] = 1
        self.status["mode_feed"] = MOVE_MODE_TO_CODE.get(move_mode.upper(), MOVE_MODE_TO_CODE["L"])
        return f"Mock Cartesian {move_mode} hold XYZ={self.pose.xyz()}"

    def emergency_stop(self) -> str:
        self.status["arm_status"] = 1
        self.status["fault"] = True
        return "Mock emergency stop"

    def pause_hold(self) -> str:
        self.status["motion_status"] = 0
        return "Mock VLA pause/joint hold"

    def shutdown_at_safe_pose(self) -> str:
        expected = self.cfg.shutdown_joints_deg
        if expected is None:
            raise RuntimeError("shutdown_joints_deg is not configured; motor disable refused")
        if any(
            abs(actual - target) > tolerance
            for actual, target, tolerance in zip(
                self.joints.values_deg, expected, self.cfg.shutdown_joint_tolerance_deg
            )
        ):
            raise RuntimeError("safe shutdown posture not verified")
        self.pause_hold()
        self.status["ctrl_mode"] = 0
        return "Mock normal shutdown"


def _required_number(obj: Any, candidates: list[str]) -> float:
    value = get_first_attr(obj, candidates)
    if value is None:
        raise AttributeError(f"Could not find any of fields {candidates} in {obj!r}")
    return float(value)


def _optional_number(obj: Any, candidates: list[str]) -> Optional[float]:
    value = get_first_attr(obj, candidates)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _err_status_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, (int, float, bool)):
        return int(obj) if isinstance(obj, bool) else obj
    result: Dict[str, Any] = {}
    for name in (
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
        "communication_status_joint_1",
        "communication_status_joint_2",
        "communication_status_joint_3",
        "communication_status_joint_4",
        "communication_status_joint_5",
        "communication_status_joint_6",
    ):
        if hasattr(obj, name):
            result[name] = bool(getattr(obj, name))
    return result or str(obj)


def _err_status_has_fault(obj: Any) -> bool:
    normalized = _err_status_to_dict(obj)
    if normalized is None:
        return False
    if isinstance(normalized, dict):
        return any(bool(value) for value in normalized.values())
    if isinstance(normalized, (int, float, bool)):
        return bool(normalized)
    return str(normalized).strip() not in ("", "0", "None")
