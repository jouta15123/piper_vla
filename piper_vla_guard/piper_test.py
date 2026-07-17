#!/usr/bin/env python3
"""Small Piper joint-following diagnostic with gravity-safe shutdown behavior."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Sequence

from piper_sdk import Piper


class InitializationRefused(RuntimeError):
    """No motor, mode, stop, or target command has been sent."""


def _sdk_joint_value(radians: float) -> int:
    return round(float(radians) * 1000.0 * 180.0 / math.pi)


def _send_joint_hold(interface, piper: Piper, speed_pct: int) -> None:
    """Replace the retained JointCtrl target without CAN 0x150 stop/reset."""
    joints = piper.get_joint_states()[0]
    for _ in range(3):
        interface.JointCtrl(*[_sdk_joint_value(value) for value in joints])
    interface.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
    for _ in range(5):
        interface.JointCtrl(*[_sdk_joint_value(value) for value in joints])
        time.sleep(0.02)
    print(f"ACTIVE JOINT HOLD: {[round(float(value), 6) for value in joints]}")


def _prepare_can_hold_from_teaching(
    interface,
    piper: Piper,
    speed_pct: int,
    *,
    stability_samples: int = 10,
    verify_samples: int = 20,
) -> list[float]:
    """Refuse the unverified Teaching-to-CAN transition without sending commands."""
    raise InitializationRefused(
        "Teaching-to-CAN preparation is disabled: hardware testing showed that this "
        "transition/failure path can remove gravity-holding torque and let the arm fall. "
        "No motor, mode, stop, enable, or target command was sent."
    )


def _enable_with_joint_hold(interface, piper: Piper, speed_pct: int, timeout_s: float = 3.0) -> list[float]:
    status = interface.GetArmStatus().arm_status
    if int(status.arm_status) != 0:
        raise InitializationRefused(f"Refusing to auto-resume non-normal arm status: {status}")
    enabled = interface.GetArmEnableStatus()
    if any(bool(value) for value in enabled):
        raise InitializationRefused(
            f"Refusing initialization while motors are already enabled: {enabled}; "
            "the retained target is unknown"
        )
    joints = list(piper.get_joint_states()[0])
    interface.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
    for _ in range(3):
        interface.JointCtrl(*[_sdk_joint_value(value) for value in joints])
        time.sleep(0.02)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if piper.enable_arm():
            for _ in range(5):
                interface.JointCtrl(*[_sdk_joint_value(value) for value in joints])
                time.sleep(0.02)
            return joints
        time.sleep(0.01)
    raise RuntimeError("Piper motor enable timed out")


def _recover_target_limit_with_joint_hold(
    interface,
    piper: Piper,
    speed_pct: int,
    timeout_s: float = 3.0,
) -> list[float]:
    """Recover arm_status=4 by replacing MoveL with measured MoveJ hold.

    This deliberately uses no EmergencyStop, reset, disable, or Teaching
    transition. It mirrors the working vendor-style MoveJ loop: measured
    JointCtrl target, CAN/MoveJ mode, enable refresh, then feedback checks.
    """
    status = interface.GetArmStatus().arm_status
    if int(status.arm_status) != 4 or int(status.ctrl_mode) != 1:
        raise InitializationRefused(
            "Target-limit recovery requires arm_status=4 TARGET_POS_EXCEEDS_LIMIT "
            f"in CAN_CTRL; no command sent: {status}"
        )
    joints = list(piper.get_joint_states()[0])
    target = [_sdk_joint_value(value) for value in joints]
    for _ in range(3):
        interface.JointCtrl(*target)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        interface.JointCtrl(*target)
        interface.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
        interface.JointCtrl(*target)
        interface.EnableArm(7, 0x02)
        interface.JointCtrl(*target)
        current_status = interface.GetArmStatus().arm_status
        enabled = interface.GetArmEnableStatus()
        if (
            int(current_status.arm_status) == 0
            and int(current_status.ctrl_mode) == 1
            and int(current_status.mode_feed) == 1
            and len(enabled) == 6
            and all(bool(value) for value in enabled)
        ):
            measured = list(piper.get_joint_states()[0])
            max_drift = max(abs(a - b) for a, b in zip(measured, joints))
            if max_drift > math.radians(1.0):
                raise RuntimeError(
                    f"Target-limit recovery changed a joint by {math.degrees(max_drift):.3f}deg"
                )
            return measured
        time.sleep(0.02)
    raise RuntimeError(
        "Target-limit recovery timed out without NORMAL CAN/MoveJ all-axis enable; "
        f"status={interface.GetArmStatus().arm_status}, enable={interface.GetArmEnableStatus()}"
    )


def run(args: argparse.Namespace) -> int:
    piper = Piper(args.can)
    interface = piper.init()
    piper.connect()
    time.sleep(0.2)

    print(interface.GetArmStatus())
    print("Enable_Status:", interface.GetArmEnableStatus())
    print("Joint_Pos:", piper.get_joint_states()[0])
    print("End_Pos_Euler:", piper.get_end_pose_euler()[0])
    if args.recover_target_limit_hold:
        if not args.execute or not args.yes:
            raise RuntimeError(
                "Target-limit recovery requires --recover-target-limit-hold --execute --yes"
            )
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive terminal is required for target-limit recovery")
        entered = input(
            f"Type {args.recovery_approval_word!r} to replace the faulting Cartesian target "
            "with measured CAN/MoveJ hold: "
        ).strip()
        if entered != args.recovery_approval_word:
            raise InitializationRefused("Target-limit recovery cancelled; no command sent")
        joints = _recover_target_limit_with_joint_hold(
            interface,
            piper,
            args.speed_pct,
            timeout_s=args.recovery_timeout_s,
        )
        print(
            "TARGET-LIMIT RECOVERY COMPLETE: NORMAL CAN/MoveJ, all axes enabled, "
            f"measured joint hold={[round(float(value), 6) for value in joints]}; "
            "no E-stop/reset/disable sent"
        )
        return 0
    if args.prepare_can_hold:
        _prepare_can_hold_from_teaching(interface, piper, args.speed_pct)
    if not args.execute:
        print("Observation only: add --execute --yes to run the bounded joint diagnostic.")
        return 0
    if not args.yes:
        raise RuntimeError("Refusing real motion without --yes")

    try:
        base = _enable_with_joint_hold(interface, piper, args.speed_pct)
        if args.gripper:
            piper.enable_gripper()

        omega = 2.0 * math.pi / args.period_s
        grip_omega = 2.0 * math.pi / args.gripper_period_s
        started = time.monotonic()
        while time.monotonic() - started < args.duration_s:
            elapsed = time.monotonic() - started
            joints = base.copy()
            joints[0] = base[0] + args.radius_rad * math.cos(omega * elapsed)
            joints[1] = base[1] + args.radius_rad * math.sin(omega * elapsed)
            piper.move_j(joints, args.speed_pct)
            if args.gripper:
                phase = (math.sin(grip_omega * elapsed) + 1.0) / 2.0
                opening = args.gripper_min_m + phase * (args.gripper_max_m - args.gripper_min_m)
                piper.move_gripper(opening, 1.0)
            time.sleep(args.control_period_s)
    except InitializationRefused:
        raise
    except BaseException:
        try:
            _send_joint_hold(interface, piper, args.speed_pct)
        except BaseException as hold_exc:
            print(
                "CRITICAL: measured-joint hold failed; support the arm and use the physical "
                f"E-stop if required: {hold_exc}",
                file=sys.stderr,
            )
        raise
    else:
        _send_joint_hold(interface, piper, args.speed_pct)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed-pct", type=int, default=2)
    parser.add_argument("--radius-rad", type=float, default=0.01)
    parser.add_argument("--period-s", type=float, default=4.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--control-period-s", type=float, default=0.02)
    parser.add_argument("--gripper", action="store_true")
    parser.add_argument("--gripper-min-m", type=float, default=0.01)
    parser.add_argument("--gripper-max-m", type=float, default=0.08)
    parser.add_argument("--gripper-period-s", type=float, default=4.0)
    parser.add_argument(
        "--prepare-can-hold",
        action="store_true",
        help="DISABLED: unsafe unverified Teaching-to-CAN transition (sends no commands).",
    )
    parser.add_argument(
        "--recover-target-limit-hold",
        action="store_true",
        help=(
            "Explicitly recover arm_status=4 in CAN control by replacing the Cartesian target "
            "with measured CAN/MoveJ hold; sends no E-stop/reset/disable."
        ),
    )
    parser.add_argument("--recovery-approval-word", default="RECOVER")
    parser.add_argument("--recovery-timeout-s", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_can_hold:
        parser.error(
            "--prepare-can-hold is disabled after a hardware test showed loss of "
            "gravity-holding torque; no CAN connection was opened"
        )
    if not 1 <= args.speed_pct <= 100:
        parser.error("--speed-pct must be in [1, 100]")
    if args.recovery_timeout_s <= 0.0:
        parser.error("--recovery-timeout-s must be positive")
    if not 0.0 < args.radius_rad <= 0.02:
        parser.error("--radius-rad must be in (0, 0.02]")
    if args.period_s <= 0 or args.duration_s <= 0 or args.control_period_s <= 0:
        parser.error("period, duration, and control period must be positive")
    if not 0.0 <= args.gripper_min_m <= args.gripper_max_m <= 0.095:
        parser.error("gripper bounds must satisfy 0 <= min <= max <= 0.095")
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
