from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from .camera_adapter import CameraCaptureError, PersistentCamera
from .config import load_config
from .executor import PlanExecutor
from .hybrid_pick import HybridPhase, HybridPickController, HybridPickError, wrist_white_object_present
from .kinematics import CartesianIKSolver
from .logging_utils import JsonlLogger
from .pick_calibration import (
    PickCalibration,
    apply_calibration_to_safety,
    load_pick_calibration,
    preprocess_camera_image,
)
from .piper_adapter import (
    ArmInitializationRefused,
    MockPiperAdapter,
    PiperSDKAdapter,
    ROSBridgeAdapter,
    RobotAdapter,
)
from .policy_adapter import OpenPIPolicyClient, response_to_json
from .safety import SafetyChecker
from .types import EEPose, GripperState, JointState, SafetyConfig


ACTION_MODE = "robosuite_osc_pose"
EXPECTED_DATASET_REPO_ID = "local/piper_topdown_lift"
EXPECTED_CHECKPOINT_STEP = 30000
EXPECTED_PROMPT = "lift the white cylinder 10cm"
EXPECTED_FPS = 20
EXPECTED_ACTION_HORIZON = 20
PICK_FLOOR_EXECUTION_BUFFER_M = 0.001
DEFAULT_NORM_STATS_PATH = str(
    pathlib.Path(__file__).resolve().parents[2]
    / "docker_vla_share_clean/workspace/openpi_vla_proj/custom_scripts_piper"
    / "fine_tuned_piper_model/pi0_piper_stack_lora/my_finetune_PBL3/30000"
    / "assets/local/piper_topdown_lift/norm_stats.json"
)


def robot_state_vector(robot: RobotAdapter, cfg: SafetyConfig) -> list[float]:
    """Return the OpenPI PiPER state: 6 joint radians + 2 robosuite gripper qpos."""
    joints = robot.read_joint_state()
    if joints is None:
        raise RuntimeError("Cannot build OpenPI observation: Piper joint feedback is unavailable.")
    gripper = robot.read_gripper_state()
    return [math.radians(v) for v in joints.values_deg] + robosuite_gripper_qpos_pair(gripper, cfg)


def robosuite_gripper_qpos_pair(
    gripper: Optional[GripperState],
    cfg: SafetyConfig,
) -> list[float]:
    if gripper is None:
        return [0.0, 0.0]
    qpos = (gripper.opening_m - cfg.robosuite_gripper_min_width_m) / 2.0
    qpos = max(0.0, min(cfg.robosuite_gripper_qpos_max_m, qpos))
    return [qpos, qpos]


def build_real_observation(
    robot: RobotAdapter,
    cfg: SafetyConfig,
    prompt: str,
    overhead_camera: PersistentCamera,
    wrist_camera: PersistentCamera,
    calibration: Optional[PickCalibration] = None,
) -> dict[str, Any]:
    """Build the same policy-facing observation shape used by the robosuite eval loop."""
    obs: dict[str, Any] = {
        "prompt": prompt,
        "observation/state": np.asarray(robot_state_vector(robot, cfg), dtype=np.float32),
    }
    overhead = overhead_camera.read_rgb()
    wrist = wrist_camera.read_rgb()
    if overhead is not None:
        overhead = preprocess_camera_image(overhead, None if calibration is None else calibration.overhead)
        obs["observation/image"] = overhead
    if wrist is not None:
        wrist = preprocess_camera_image(wrist, None if calibration is None else calibration.wrist)
        obs["observation/wrist_image"] = wrist
    return obs


def select_action_chunk(response: Any, chunk_size: int, max_horizon: int) -> np.ndarray:
    if not isinstance(response, dict) or "actions" not in response:
        keys = list(response.keys()) if isinstance(response, dict) else type(response).__name__
        raise RuntimeError(f"OpenPI response has no actions key: {keys}")
    all_actions = np.asarray(response["actions"], dtype=np.float32)
    if all_actions.ndim != 2 or all_actions.shape[1] != 7 or len(all_actions) == 0:
        raise ValueError(f"Invalid action shape: {all_actions.shape}. Expected (action_horizon, 7).")
    use_steps = min(max(1, int(chunk_size)), len(all_actions), max(1, int(max_horizon)))
    return all_actions[:use_steps]


def action_chunk_stats(actions: Any) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
        raise ValueError(f"Invalid action shape for statistics: {values.shape}")
    translation = values[:, :3]
    return {
        "shape": list(values.shape),
        "translation_min": translation.min(axis=0).tolist(),
        "translation_max": translation.max(axis=0).tolist(),
        "translation_norm_min": float(np.linalg.norm(translation, axis=1).min()),
        "translation_norm_max": float(np.linalg.norm(translation, axis=1).max()),
    }


def apply_action_xyz_signs(actions: np.ndarray, signs_text: str) -> np.ndarray:
    signs = _parse_three_floats(signs_text, "--action-xyz-signs")
    transformed = np.asarray(actions, dtype=np.float32).copy()
    transformed[:, 0] *= signs[0]
    transformed[:, 1] *= signs[1]
    transformed[:, 2] *= signs[2]
    return transformed


def run_real_loop(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    calibration = load_pick_calibration(args.calibration)
    if calibration is not None:
        apply_calibration_to_safety(calibration, cfg)
    cfg.can_name = args.can
    cfg.sdk_attach_enabled_can = bool(args.attach_enabled_can)
    cfg.speed_pct = int(args.speed_pct)
    if args.step_sleep_s is not None:
        cfg.step_sleep_s = float(args.step_sleep_s)
    cfg.control_hz = float(EXPECTED_FPS)
    cfg.step_sleep_s = 1.0 / cfg.control_hz
    cfg.dry_run = not bool(args.execute) or args.mode == "observe"
    if args.max_step_xyz_m is not None:
        step = float(args.max_step_xyz_m)
        cfg.max_step_xyz_m = (step, step, step)
    if args.max_horizon is not None:
        cfg.max_horizon = int(args.max_horizon)
    if args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla"):
        arm_test_max_xy_m = float(args.arm_test_max_xy_m)
        if not 0.0 < arm_test_max_xy_m <= 0.003:
            raise RuntimeError("--arm-test-max-xy-m must be in (0, 0.003]")
        arm_test_tracking_tolerance_m = float(args.arm_test_tracking_tolerance_m)
        if not 0.0 < arm_test_tracking_tolerance_m <= 0.001:
            raise RuntimeError("--arm-test-tracking-tolerance-m must be in (0, 0.001]")
        pick_vla_tracking_tolerance_m = float(args.pick_vla_tracking_tolerance_m)
        if not 0.0 < pick_vla_tracking_tolerance_m <= 0.005:
            raise RuntimeError("--pick-vla-tracking-tolerance-m must be in (0, 0.005]")
        xyz_vla_max_z_m = float(args.xyz_vla_max_z_m)
        if not 0.0 < xyz_vla_max_z_m <= 0.003:
            raise RuntimeError("--xyz-vla-max-z-m must be in (0, 0.003]")
        pick_vla_max_xyz_m = float(args.pick_vla_max_xyz_m)
        if not 0.0 < pick_vla_max_xyz_m <= 0.05:
            raise RuntimeError("--pick-vla-max-xyz-m must be in (0, 0.05]")
        if args.mode == "arm-test":
            if int(args.max_cycles) != 1:
                raise RuntimeError("arm-test is limited to exactly one cycle")
            if int(args.chunk_size) != 1:
                print("arm-test forces --chunk-size 1")
                args.chunk_size = 1
        elif args.mode == "xy-vla":
            if not 1 <= int(args.max_cycles) <= 5:
                raise RuntimeError("xy-vla requires --max-cycles in [1, 5]")
            if not 1 <= int(args.chunk_size) <= 5:
                raise RuntimeError("xy-vla requires --chunk-size in [1, 5]")
            if not 0.0 < float(args.max_loop_translation_m) <= 0.05:
                raise RuntimeError("xy-vla requires --max-loop-translation-m in (0, 0.05]")
        elif args.mode == "xyz-vla":
            if not 1 <= int(args.max_cycles) <= 2:
                raise RuntimeError("xyz-vla requires --max-cycles in [1, 2]")
            if not 1 <= int(args.chunk_size) <= 3:
                raise RuntimeError("xyz-vla requires --chunk-size in [1, 3]")
            if not 0.0 < float(args.max_loop_translation_m) <= 0.03:
                raise RuntimeError("xyz-vla requires --max-loop-translation-m in (0, 0.03]")
        else:
            if not 1 <= int(args.max_cycles) <= 40:
                raise RuntimeError("pick-vla requires --max-cycles in [1, 40]")
            if not 1 <= int(args.chunk_size) <= 10:
                raise RuntimeError("pick-vla requires --chunk-size in [1, 10]")
            if not 0.0 < float(args.max_loop_translation_m) <= 0.12:
                raise RuntimeError("pick-vla requires --max-loop-translation-m in (0, 0.12]")
            # The bounded pick run streams checked joint interpolation targets
            # and still aborts on the lower-level measured-speed guard.  Allow
            # the 5 mm Cartesian step seen on hardware (about 23 deg/s at J5)
            # without weakening the global 30 deg/s measured-speed limit.
            cfg.max_commanded_joint_speed_deg_s = min(
                cfg.max_measured_joint_speed_deg_s,
                25.0,
            )
            # Match robosuite env.step semantics: each raw OSC action updates
            # the position-servo target once at 20 Hz; it is not forced to
            # settle before the next VLA action arrives.
            cfg.control_hz = float(EXPECTED_FPS)
            cfg.step_sleep_s = 1.0 / cfg.control_hz
            cfg.max_joint_step_deg = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        if args.mode != "pick-vla":
            args.no_gripper = True
        args.pick_assist = False
        xy_limit = pick_vla_max_xyz_m if args.mode == "pick-vla" else arm_test_max_xy_m
        z_limit = (
            pick_vla_max_xyz_m
            if args.mode == "pick-vla"
            else xyz_vla_max_z_m if args.mode == "xyz-vla" else arm_test_max_xy_m
        )
        if getattr(args, "mode", None) == "pick-vla":
            cfg.max_step_xyz_m = (xy_limit, xy_limit, z_limit)
        else:
            cfg.max_step_xyz_m = (
                min(cfg.max_step_xyz_m[0], xy_limit),
                min(cfg.max_step_xyz_m[1], xy_limit),
                min(cfg.max_step_xyz_m[2], z_limit),
            )
        cfg.max_step_rpy_deg = (0.0, 0.0, 0.0)
        if args.mode == "pick-vla":
            # Each following VLA delta is rebased on measured feedback, so a
            # small steady-state target residual does not accumulate. Keep a
            # separate 5 mm abort boundary while checking the measured pose
            # against the workspace/floor after every action.
            cfg.tracking_hold_error_m = pick_vla_tracking_tolerance_m
            cfg.tracking_abort_error_m = 0.005
        else:
            cfg.tracking_hold_error_m = min(
                cfg.tracking_hold_error_m,
                arm_test_tracking_tolerance_m,
            )
            cfg.tracking_abort_error_m = min(cfg.tracking_abort_error_m, 0.001)
        cfg.tracking_settle_timeout_s = max(cfg.tracking_settle_timeout_s, 3.0)
        cfg.settle_each_vla_step = args.mode != "pick-vla"
        print(
            f"{args.mode} constrains VLA execution to "
            f"{'XYZ' if args.mode in ('xyz-vla', 'pick-vla') else 'XY'} only; "
            f"{'RPY' if args.mode == 'pick-vla' else 'RPY/gripper' if args.mode == 'xyz-vla' else 'Z/RPY/gripper'} are suppressed; "
            f"max XY step={xy_limit:.6f}m; "
            f"max Z step={z_limit:.6f}m; "
            f"planned joint speed limit={cfg.max_commanded_joint_speed_deg_s:.1f}deg/s; "
            f"measured joint speed limit={cfg.max_measured_joint_speed_deg_s:.1f}deg/s; "
            f"tracking tolerance="
            f"{pick_vla_tracking_tolerance_m if args.mode == 'pick-vla' else arm_test_tracking_tolerance_m:.6f}m; "
            f"tracking abort={cfg.tracking_abort_error_m:.6f}m"
        )

    logger = JsonlLogger(cfg.log_dir)
    if args.mock:
        robot: RobotAdapter = MockPiperAdapter()
    elif args.transport == "ros":
        cfg.rosbridge_host = args.rosbridge_host
        cfg.rosbridge_port = int(args.rosbridge_port)
        robot = ROSBridgeAdapter(cfg)
    else:
        execution_text = (
            "checked joint IK/JointCtrl"
            if cfg.cartesian_execution_mode == "joint_ik"
            else "Piper EndPoseCtrl (controller-side Cartesian kinematics)"
        )
        print(
            "Direct Piper SDK/CAN transport selected; Cartesian targets will execute through "
            + execution_text
        )
        robot = PiperSDKAdapter(cfg)
    print(robot.connect())
    startup_status = robot.read_arm_status()
    print("Piper startup status: " + json.dumps(startup_status, ensure_ascii=False, default=str))
    logger.write("piper_startup_status", startup_status)
    ready_errors = _ready_pose_errors(robot, cfg)
    logger.write("ready_pose_check", {"ok": not ready_errors, "errors": ready_errors})
    if ready_errors:
        print("Ready-pose warning: " + "; ".join(ready_errors))
    else:
        joints = robot.read_joint_state()
        values = "unavailable" if joints is None else _fmt_list(joints.as_list())
        print(f"Ready-pose check: OK, measured joints={values}")
    if args.execute and args.mode in ("hybrid", "pure-vla"):
        if calibration is None:
            raise RuntimeError("--calibration is required for hybrid/pure-vla execution")
        calibration.require_complete()
    if args.execute and args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla"):
        if calibration is None:
            raise RuntimeError("--calibration is required for XY VLA execution")
        if not args.save_observation_dir:
            raise RuntimeError("XY VLA execution requires --save-observation-dir")
        if not str(args.overhead_camera_source).strip() or not str(args.wrist_camera_source).strip():
            raise RuntimeError("XY VLA execution requires overhead and wrist cameras")
    if args.execute and args.mode == "hybrid" and not args.save_observation_dir:
        raise RuntimeError("hybrid execution requires --save-observation-dir for grasp approval evidence")
    if args.execute and args.mode == "hybrid" and not str(args.wrist_camera_source).strip():
        raise RuntimeError("hybrid execution requires --wrist-camera-source for post-lift verification")
    if args.execute and args.mode == "pure-vla":
        _require_pure_vla_gate(args.pure_vla_gate_file)

    if args.execute and not args.yes:
        raise RuntimeError("Refusing to send real Piper commands without --yes.")
    # --auto-ready means move to the configured checkpoint joints, not merely
    # accept a pose somewhere inside the relatively broad inference gate.
    wants_ready_validation = bool(args.auto_ready)
    needs_ready_move = bool(args.execute and wants_ready_validation)
    if wants_ready_validation:
        if calibration is None:
            raise RuntimeError("--auto-ready requires --calibration")
        calibration.require_ready_path()
        if not args.execute:
            current, path, _ = _preflight_ready_path(robot, cfg, calibration)
            print(
                "AUTO-READY DRY-RUN PREFLIGHT OK: "
                f"start={_fmt_list(current.as_list())}, waypoints={len(path)}; no motor command sent."
            )
    # Every mode that can execute a VLA-derived action must start from the
    # checkpoint's training-ready pose.  --auto-ready is the explicit way to
    # reach it when current feedback is outside the configured tolerance.
    ready_required_modes = ("arm-test", "xy-vla", "xyz-vla", "pick-vla", "hybrid", "pure-vla")
    if args.execute and args.mode in ready_required_modes and not needs_ready_move:
        _require_ready_pose(robot, cfg)

    client = OpenPIPolicyClient(host=args.policy_host, port=args.policy_port)
    norm_stats_sha256 = _sha256_file(args.expected_norm_stats)
    policy_identity = client.validate_identity(
        dataset_repo_id=EXPECTED_DATASET_REPO_ID,
        checkpoint_step=EXPECTED_CHECKPOINT_STEP,
        prompt=args.prompt,
        fps=EXPECTED_FPS,
        norm_stats_sha256=norm_stats_sha256,
    )
    logger.write("policy_identity_verified", policy_identity)
    args._arm_prepared = False
    defer_arm_test_enable = bool(
        args.execute and args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla")
    )
    overhead_camera: Optional[PersistentCamera] = None
    wrist_camera: Optional[PersistentCamera] = None
    if args.execute:
        overhead_camera, wrist_camera = _preflight_camera_sources(args)
    try:
        if args.vendor_teaching_bootstrap and not args.execute:
            print(
                "VENDOR TEACHING BOOTSTRAP DRY-RUN: requested but no command will be sent "
                "without --execute --yes."
            )
        elif args.vendor_teaching_bootstrap:
            if args.mock or args.transport != "sdk":
                raise ArmInitializationRefused(
                    "--vendor-teaching-bootstrap is available only with real --transport sdk."
                )
            current_status = robot.read_arm_status()
            if current_status.get("ctrl_mode") == 2.0:
                print("\n=== VENDOR TEACHING -> CAN BOOTSTRAP ===")
                print(
                    "This follows test_ctrlPiperJoint_can0_2.py: EmergencyStop(0x01), "
                    "damped descent to |J2|/|J3|<10deg and 12deg<J5<45deg, then "
                    "DisablePiper+resume, CAN/MoveJ, enable, and measured-joint hold."
                )
                print(
                    "The arm WILL descend during this operation. Keep the workspace clear, "
                    "support the arm as required, and keep the physical E-stop reachable."
                )
                if not sys.stdin.isatty():
                    raise ArmInitializationRefused(
                        "Interactive terminal is required for Teaching bootstrap approval."
                    )
                entered = input(
                    f"Type {args.bootstrap_approval_word!r} to run the vendor bootstrap: "
                ).strip()
                if entered != args.bootstrap_approval_word:
                    raise ArmInitializationRefused(
                        "Teaching bootstrap cancelled by operator; no bootstrap command sent."
                    )
                print(
                    robot.bootstrap_teaching_to_can(
                        speed_pct=cfg.ready_return_speed_pct,
                        timeout_s=args.bootstrap_timeout_s,
                    )
                )
                cfg.sdk_attach_enabled_can = True
                args._arm_prepared = True
                post_bootstrap = robot.read_arm_status()
                logger.write("vendor_teaching_bootstrap_complete", post_bootstrap)
                print(
                    "Piper post-bootstrap status: "
                    + json.dumps(post_bootstrap, ensure_ascii=False, default=str)
                )
            else:
                print(
                    "Vendor Teaching bootstrap skipped because current ctrl_mode is not "
                    f"TEACHING_MODE: {current_status.get('ctrl_mode_name')}"
                )
        if needs_ready_move:
            assert calibration is not None
            _move_robot_to_ready(
                robot,
                cfg,
                calibration,
                logger=logger,
                approval_word=args.ready_approval_word,
            )
            _require_ready_pose(robot, cfg, announce=True)
            args._arm_prepared = True
        elif (args.execute or args.enable) and not defer_arm_test_enable:
            move_mode = "J" if cfg.cartesian_execution_mode == "joint_ik" else "L"
            print(robot.enable(speed_pct=cfg.speed_pct, move_mode=move_mode))
            args._arm_prepared = True
        if args.execute and args.mode in ready_required_modes:
            _require_ready_pose(robot, cfg, announce=True)
    except ArmInitializationRefused as exc:
        _close_cameras(overhead_camera, wrist_camera)
        print(
            "Arm initialization stopped: "
            f"{exc} No additional automatic E-stop will be sent.",
            file=sys.stderr,
        )
        return 4
    except CameraCaptureError as exc:
        _close_cameras(overhead_camera, wrist_camera)
        logger.write("camera_capture_aborted", {"error": str(exc), "arm_prepared": args._arm_prepared})
        if args._arm_prepared:
            try:
                print(
                    "Camera capture failed after arm preparation; maintaining measured-joint hold: "
                    + robot.pause_hold()
                )
            except Exception as hold_exc:
                print(
                    "CRITICAL: camera failed and measured-joint hold refresh also failed: "
                    f"{hold_exc}. Use the physical E-stop while supporting the arm.",
                    file=sys.stderr,
                )
        print(f"Camera capture failed; VLA action was not executed: {exc}", file=sys.stderr)
        return 6
    except BaseException:
        _close_cameras(overhead_camera, wrist_camera)
        if args.execute or args.enable:
            try:
                print("Exception during arm preparation; applying VLA pause/joint hold: " + robot.pause_hold())
            except Exception as stop_exc:
                print(
                    "CRITICAL: pause/joint hold failed; support the arm and use the physical "
                    f"E-stop if required: {stop_exc}",
                    file=sys.stderr,
                )
        raise
    executor = PlanExecutor(robot, cfg, logger=logger)
    ik_solver = CartesianIKSolver(cfg) if cfg.cartesian_execution_mode == "joint_ik" else None
    checker = SafetyChecker(cfg, ik_solver=ik_solver)
    if overhead_camera is None:
        overhead_camera = _make_camera(args, args.overhead_camera_source, "overhead")
    else:
        overhead_camera.label = "overhead"
    if wrist_camera is None:
        wrist_camera = _make_camera(args, args.wrist_camera_source, "wrist")
    else:
        wrist_camera.label = "wrist"
    hybrid_controller = (
        HybridPickController(cfg, calibration)
        if args.mode == "hybrid" and calibration is not None
        else None
    )

    print(
        "Starting Piper VLA loop: "
        f"dry_run={cfg.dry_run}, cycles={args.max_cycles}, chunk_size={args.chunk_size}, "
        f"policy={args.policy_host}:{args.policy_port}"
    )
    if cfg.dry_run:
        print("Dry-run is ON: policy and safety loop run, but no robot commands are sent.")

    loop_initial_pose = robot.read_ee_pose()
    print(f"Loop origin XYZ=({_fmt_xyz_values(loop_initial_pose.xyz())})")

    try:
        for cycle in range(int(args.max_cycles)):
            result_code = _run_cycle(
                cycle=cycle,
                args=args,
                cfg=cfg,
                robot=robot,
                checker=checker,
                executor=executor,
                client=client,
                logger=logger,
                overhead_camera=overhead_camera,
                wrist_camera=wrist_camera,
                loop_initial_pose=loop_initial_pose,
                calibration=calibration,
                hybrid_controller=hybrid_controller,
            )
            if result_code is not None:
                if args.execute and getattr(args, "_arm_prepared", False):
                    print("Real loop exit; retaining measured-joint hold: " + robot.pause_hold())
                    _maybe_return_pick_to_ready(
                        args=args,
                        cfg=cfg,
                        robot=robot,
                        calibration=calibration,
                        logger=logger,
                    )
                return result_code
    except ArmInitializationRefused as exc:
        print(
            "Arm initialization stopped: "
            f"{exc} No additional automatic E-stop will be sent.",
            file=sys.stderr,
        )
        return 4
    except BaseException:
        if args.execute or args.enable:
            try:
                print("Exception during VLA loop; applying VLA pause/joint hold: " + robot.pause_hold())
            except Exception as stop_exc:
                print(
                    "CRITICAL: pause/joint hold failed; support the arm and use the physical "
                    f"E-stop if required: {stop_exc}",
                    file=sys.stderr,
                )
        raise
    finally:
        overhead_camera.close()
        wrist_camera.close()

    if args.execute and getattr(args, "_arm_prepared", False):
        print("Normal VLA completion; retaining measured-joint hold: " + robot.pause_hold())
        _maybe_return_pick_to_ready(
            args=args,
            cfg=cfg,
            robot=robot,
            calibration=calibration,
            logger=logger,
        )
    print("Finished Piper VLA loop.")
    return 0


def _make_camera(args: argparse.Namespace, source: Any, label: str) -> PersistentCamera:
    return PersistentCamera(
        source,
        retries=args.camera_retries,
        retry_sleep_s=args.camera_retry_sleep_s,
        label=label,
        warmup_frames=args.camera_warmup_frames,
        max_identical_frames=args.camera_max_identical_frames,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        fourcc=args.camera_fourcc,
        buffer_size=args.camera_buffer_size,
        read_timeout_ms=args.camera_read_timeout_ms,
    )


def _close_cameras(*cameras: Optional[PersistentCamera]) -> None:
    for camera in cameras:
        if camera is not None:
            camera.close()


def _maybe_return_pick_to_ready(
    *,
    args: argparse.Namespace,
    cfg: SafetyConfig,
    robot: RobotAdapter,
    calibration: Optional[PickCalibration],
    logger: Optional[JsonlLogger],
) -> None:
    """Return a normally controlled pick run to checkpoint-ready joints.

    Faulted/non-CAN states remain held for operator recovery; automatic motion
    is attempted only from NORMAL CAN control after the VLA queue is gone.
    """
    if args.mode != "pick-vla" or not args.auto_ready or calibration is None:
        return
    status = robot.read_arm_status()
    if (
        bool(status.get("fault"))
        or status.get("ctrl_mode") != 1.0
        or status.get("arm_status") != 0.0
    ):
        print(
            "READY RETURN SKIPPED: arm is not NORMAL CAN control; retaining hold: "
            + json.dumps(status, ensure_ascii=False, default=str),
            file=sys.stderr,
        )
        return
    print("\n=== POST-PICK RETURN TO CHECKPOINT READY ===")
    _move_robot_to_ready(
        robot,
        cfg,
        calibration,
        logger=logger,
        approval_word=args.ready_approval_word,
    )
    _require_ready_pose(robot, cfg, announce=True)


def _preflight_camera_sources(
    args: argparse.Namespace,
) -> tuple[PersistentCamera, PersistentCamera]:
    """Require both live camera streams before any ready/enable command."""
    cameras = (
        _make_camera(args, args.overhead_camera_source, "overhead-preflight"),
        _make_camera(args, args.wrist_camera_source, "wrist-preflight"),
    )
    try:
        frames = [camera.read_rgb() for camera in cameras]
        missing = [camera.label for camera, frame in zip(cameras, frames) if frame is None]
        if missing:
            raise CameraCaptureError(f"required camera sources are empty: {missing}")
        shapes = [list(np.asarray(frame).shape) for frame in frames]
        print(f"CAMERA PREFLIGHT OK before arm command: shapes={shapes}")
        return cameras
    except BaseException:
        _close_cameras(*cameras)
        raise


def _run_cycle(
    *,
    cycle: int,
    args: argparse.Namespace,
    cfg: SafetyConfig,
    robot: RobotAdapter,
    checker: SafetyChecker,
    executor: PlanExecutor,
    client: OpenPIPolicyClient,
    logger: JsonlLogger,
    overhead_camera: PersistentCamera,
    wrist_camera: PersistentCamera,
    loop_initial_pose: Any,
    calibration: Optional[PickCalibration],
    hybrid_controller: Optional[HybridPickController],
) -> Optional[int]:
        started = time.time()
        pose = robot.read_ee_pose()
        joints = robot.read_joint_state()
        observation = build_real_observation(
            robot=robot,
            cfg=cfg,
            prompt=args.prompt,
            overhead_camera=overhead_camera,
            wrist_camera=wrist_camera,
            calibration=calibration,
        )
        action_mode = ACTION_MODE
        proposal = None
        if hybrid_controller is not None:
            overhead = observation.get("observation/image")
            if overhead is None:
                raise HybridPickError("hybrid mode requires an overhead image")
            proposal = hybrid_controller.propose(pose, np.asarray(overhead))
            logger.write(
                "hybrid_proposal",
                {
                    "phase": proposal.phase.value,
                    "xy_error_m": proposal.xy_error_m,
                    "detection": proposal.detection.__dict__,
                    "action": proposal.action,
                },
            )
            print(f"hybrid phase={proposal.phase.value} xy_error={proposal.xy_error_m:.4f}m")
            if proposal.phase == HybridPhase.WAIT_APPROVAL:
                _save_approval_snapshot(args.save_observation_dir, cycle, observation, pose, proposal)
                approved = _request_grasp_approval(args, cfg, pose, proposal)
                result = hybrid_controller.run_approved_grasp(
                    robot,
                    executor,
                    checker,
                    logger,
                    dry_run=cfg.dry_run,
                    approved=approved,
                    verify_lift=lambda: wrist_white_object_present(
                        preprocess_camera_image(
                            wrist_camera.read_rgb(),
                            None if calibration is None else calibration.wrist,
                        )
                    ),
                )
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 0
        if proposal is not None and proposal.action is not None:
            action_mode = "delta_base_m_deg"
            response = {"actions": [proposal.action], "source": "hybrid_vision_servo"}
            action_chunk = np.asarray([proposal.action], dtype=np.float32)
        else:
            response = client.client.infer(observation)
            predicted = np.asarray(response.get("actions") if isinstance(response, dict) else None)
            if predicted.ndim != 2 or predicted.shape != (EXPECTED_ACTION_HORIZON, 7):
                raise RuntimeError(
                    f"Policy output mismatch: got {predicted.shape}, "
                    f"expected ({EXPECTED_ACTION_HORIZON}, 7)"
                )
            action_chunk = select_action_chunk(response, args.chunk_size, cfg.max_horizon)
        if action_mode == ACTION_MODE:
            stats = action_chunk_stats(response["actions"])
            print(f"VLA output stats: {json.dumps(stats, ensure_ascii=False)}")
            print("VLA selected output before guard transforms:")
            for action_index, action in enumerate(np.asarray(action_chunk)):
                print(f"  vla_action[{action_index}]={_fmt_list(action)}")
        if action_mode == ACTION_MODE and args.action_xyz_signs != "1,1,1":
            action_chunk = apply_action_xyz_signs(action_chunk, args.action_xyz_signs)
        if args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla"):
            action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
            if args.mode in ("xyz-vla", "pick-vla"):
                action_chunk[:, 3:6] = 0.0
            else:
                action_chunk[:, 2:6] = 0.0
        if args.mode == "pick-vla":
            action_chunk, plan = _clamp_pick_descent_to_floor(
                checker=checker,
                current_pose=pose,
                current_joints=joints,
                action_chunk=np.asarray(action_chunk, dtype=np.float32),
                action_mode=action_mode,
            )
        else:
            plan = checker.build_plan(
                current_pose=pose,
                current_joints=joints,
                actions=action_chunk.tolist(),
                action_mode=action_mode,
            )
        _enforce_loop_translation_limit(plan, loop_initial_pose, args.max_loop_translation_m)
        if args.no_gripper:
            _remove_gripper_commands(plan)
        if args.save_observation_dir:
            _save_cycle_debug(args.save_observation_dir, cycle, observation, response, action_chunk, plan)
        logger.write(
            "real_loop_cycle",
            {
                "cycle": cycle,
                "prompt": args.prompt,
                "observation_keys": sorted(observation.keys()),
                "raw_response": _jsonable(response),
                "selected_action_count": len(action_chunk),
                "raw_action_stats": (
                    action_chunk_stats(response["actions"])
                    if action_mode == ACTION_MODE
                    else action_chunk_stats(action_chunk)
                ),
                "plan": plan.to_dict(),
            },
        )

        print(f"\ncycle={cycle} actions={len(action_chunk)} safety={plan.approved_by_safety}")
        print(plan.summary)
        for step in plan.steps:
            print(f"  step={step.index} raw_action={_fmt_list(step.raw_action)}")
            print(f"  step={step.index} clipped_action={_fmt_list(step.clipped_action)}")
            if step.target_pose is not None:
                print(f"  step={step.index} target_pose={_fmt_list(step.target_pose.as_list())}")
            if step.gripper_m is not None:
                print(f"  step={step.index} gripper_target_m={step.gripper_m:.6f}")
            for warning in step.warnings:
                print(f"  warning step={step.index}: {warning}")
            for violation in step.violations:
                print(f"  violation step={step.index}: {violation}")
        if not plan.approved_by_safety:
            print("Safety rejected this chunk; stopping loop.")
            return 2

        human_approved = True
        if args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla") and not cfg.dry_run:
            already_approved = bool(
                args.mode == "pick-vla" and getattr(args, "_pick_run_approved", False)
            )
            human_approved = already_approved or _request_arm_test_approval(args, plan)
            if human_approved:
                if args.mode == "pick-vla":
                    args._pick_run_approved = True
                if not getattr(args, "_arm_prepared", False):
                    before_enable = robot.read_ee_pose()
                    move_mode = "J" if cfg.cartesian_execution_mode == "joint_ik" else "L"
                    print(robot.enable(speed_pct=cfg.speed_pct, move_mode=move_mode))
                    time.sleep(0.20)
                    after_enable = robot.read_ee_pose()
                    enable_drift = math.sqrt(
                        sum((a - b) ** 2 for a, b in zip(before_enable.xyz(), after_enable.xyz()))
                    )
                    print(
                        f"Enable hold check: before=({_fmt_xyz_values(before_enable.xyz())}), "
                        f"after=({_fmt_xyz_values(after_enable.xyz())}), drift={enable_drift:.6f}m"
                    )
                    if enable_drift > cfg.max_start_pose_drift_m:
                        stop_message = robot.pause_hold()
                        raise RuntimeError(
                            f"Arm drifted {enable_drift:.6f}m while enabling; {stop_message}"
                        )
                    args._arm_prepared = True
        if (
            not cfg.dry_run
            and cfg.cartesian_execution_mode == "joint_ik"
            and len(action_chunk) > 1
        ):
            result = _execute_live_rebased_chunk(
                action_chunk=action_chunk,
                action_mode=action_mode,
                args=args,
                cfg=cfg,
                robot=robot,
                checker=checker,
                executor=executor,
                loop_initial_pose=loop_initial_pose,
                human_approved=human_approved,
            )
        else:
            result = executor.execute(plan, human_approved=human_approved, dry_run=cfg.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if (
            args.mode in ("arm-test", "xy-vla", "xyz-vla", "pick-vla")
            and not cfg.dry_run
            and human_approved
        ):
            if result.get("ok"):
                print(robot.pause_hold())
            elif result.get("executed_steps", 0) and not result.get("stop_kind"):
                # Cover an unclassified post-command failure with measured-joint hold.
                print(robot.pause_hold())
        if not result.get("ok"):
            print("Execution failed or was blocked; stopping loop.")
            return 3
        if args.mode != "hybrid" and _maybe_run_pick_assist(args, cfg, robot, logger):
            return 0

        if args.cycle_sleep_s > 0:
            elapsed = time.time() - started
            time.sleep(max(0.0, float(args.cycle_sleep_s) - elapsed))
        return None


def _execute_live_rebased_chunk(
    *,
    action_chunk: np.ndarray,
    action_mode: str,
    args: argparse.Namespace,
    cfg: SafetyConfig,
    robot: RobotAdapter,
    checker: SafetyChecker,
    executor: PlanExecutor,
    loop_initial_pose: EEPose,
    human_approved: bool,
) -> dict[str, Any]:
    """Execute one policy prefix like repeated robosuite env.step calls.

    Every Cartesian delta is rebuilt from the live EE/joint feedback after the
    preceding 50 ms action period. The full nominal prefix was already checked
    for operator preview; this second check guards the actual rebased target.
    """
    aggregate: dict[str, Any] = {
        "ok": False,
        "dry_run": False,
        "executed_steps": 0,
        "max_tracking_error_m": 0.0,
        "stop_kind": None,
        "messages": [],
        "live_rebased": True,
    }
    for action_index, action in enumerate(np.asarray(action_chunk)):
        live_pose = robot.read_ee_pose()
        live_joints = robot.read_joint_state()
        if getattr(args, "mode", None) == "pick-vla":
            _, live_plan = _clamp_pick_descent_to_floor(
                checker=checker,
                current_pose=live_pose,
                current_joints=live_joints,
                action_chunk=np.asarray([action], dtype=np.float32),
                action_mode=action_mode,
            )
        else:
            live_plan = checker.build_plan(
                current_pose=live_pose,
                current_joints=live_joints,
                actions=[action.tolist()],
                action_mode=action_mode,
            )
        _enforce_loop_translation_limit(live_plan, loop_initial_pose, args.max_loop_translation_m)
        if args.no_gripper:
            _remove_gripper_commands(live_plan)
        if not live_plan.approved_by_safety:
            hold = robot.pause_hold()
            aggregate["stop_kind"] = "pause_hold"
            aggregate["messages"].append(
                f"Live-rebased action {action_index} rejected: {live_plan.summary}; {hold}"
            )
            return aggregate
        step_result = executor.execute(
            live_plan,
            human_approved=human_approved,
            dry_run=False,
        )
        aggregate["executed_steps"] += int(step_result.get("executed_steps", 0))
        aggregate["max_tracking_error_m"] = max(
            float(aggregate["max_tracking_error_m"]),
            float(step_result.get("max_tracking_error_m", 0.0)),
        )
        aggregate["messages"].extend(
            f"action {action_index}: {message}" for message in step_result.get("messages", [])
        )
        if not step_result.get("ok"):
            aggregate["stop_kind"] = step_result.get("stop_kind")
            return aggregate
    aggregate["ok"] = True
    return aggregate


def _jsonable(value: Any) -> Any:
    try:
        json.loads(response_to_json(value))
        return json.loads(response_to_json(value))
    except Exception:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def _fmt_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def _fmt_xyz_values(values: Sequence[float]) -> str:
    return ", ".join(f"{float(v):.6f}" for v in values)


def _clamp_pick_descent_to_floor(
    *,
    checker: SafetyChecker,
    current_pose: EEPose,
    current_joints: Optional[JointState],
    action_chunk: np.ndarray,
    action_mode: str,
) -> tuple[np.ndarray, Any]:
    """Clamp only floor-crossing negative Z; preserve VLA gripper and lift actions."""
    buffered_cfg = copy.deepcopy(checker.cfg)
    buffered_cfg.table_margin_m += PICK_FLOOR_EXECUTION_BUFFER_M
    buffered_cfg.workspace_floor_margin_m += PICK_FLOOR_EXECUTION_BUFFER_M
    buffered_checker = SafetyChecker(buffered_cfg)
    transformed: list[np.ndarray] = []
    clamped: list[tuple[int, float, float]] = []
    pose = current_pose
    joints = current_joints

    for index, raw_value in enumerate(np.asarray(action_chunk, dtype=np.float32)):
        raw = raw_value.copy()
        single = checker.build_plan(pose, [raw.tolist()], action_mode, joints)
        buffered_single = buffered_checker.build_plan(
            pose,
            [raw.tolist()],
            action_mode,
            joints,
        )
        step = single.steps[0]
        buffered_step = buffered_single.steps[0]
        violations = " ".join(buffered_step.violations).lower()
        floor_blocked = "tool point" in violations and (
            "table clearance" in violations or "four-corner floor" in violations
        )
        if not buffered_step.ok and floor_blocked and float(raw[2]) < 0.0:
            original_z = float(raw[2])
            zero_z = raw.copy()
            zero_z[2] = 0.0
            clamp_checker = buffered_checker
            zero_plan = clamp_checker.build_plan(
                pose,
                [zero_z.tolist()],
                action_mode,
                joints,
            )
            # Feedback can settle inside the 1 mm execution buffer while
            # remaining above the hard floor. In that case retain a zero-Z
            # VLA action under the hard checker instead of inventing an upward
            # correction that did not come from the policy.
            if not zero_plan.approved_by_safety:
                hard_zero_plan = checker.build_plan(
                    pose,
                    [zero_z.tolist()],
                    action_mode,
                    joints,
                )
                if hard_zero_plan.approved_by_safety:
                    clamp_checker = checker
                    zero_plan = hard_zero_plan
            uniform_xyz = False
            zero_candidate = zero_z
            zero_candidate_plan = zero_plan
            if not zero_plan.approved_by_safety:
                zero_xyz = raw.copy()
                zero_xyz[:3] = 0.0
                zero_xyz_plan = clamp_checker.build_plan(
                    pose,
                    [zero_xyz.tolist()],
                    action_mode,
                    joints,
                )
                if not zero_xyz_plan.approved_by_safety and clamp_checker is buffered_checker:
                    hard_zero_xyz_plan = checker.build_plan(
                        pose,
                        [zero_xyz.tolist()],
                        action_mode,
                        joints,
                    )
                    if hard_zero_xyz_plan.approved_by_safety:
                        clamp_checker = checker
                        zero_xyz_plan = hard_zero_xyz_plan
                if zero_xyz_plan.approved_by_safety:
                    uniform_xyz = True
                    zero_candidate = zero_xyz
                    zero_candidate_plan = zero_xyz_plan
            if zero_candidate_plan.approved_by_safety:
                low = 0.0
                high = 1.0
                best = zero_candidate
                best_plan = zero_candidate_plan
                for _ in range(24):
                    scale = (low + high) / 2.0
                    candidate = raw.copy()
                    if uniform_xyz:
                        candidate[:3] = np.asarray(raw[:3] * scale, dtype=np.float32)
                    else:
                        candidate[2] = np.float32(original_z * scale)
                    candidate_plan = clamp_checker.build_plan(
                        pose,
                        [candidate.tolist()],
                        action_mode,
                        joints,
                    )
                    if candidate_plan.approved_by_safety:
                        low = scale
                        best = candidate
                        best_plan = candidate_plan
                    else:
                        high = scale
                raw = best
                single = best_plan
                step = single.steps[0]
                clamped.append((index, original_z, float(raw[2])))

        transformed.append(raw)
        if step.target_pose is not None:
            pose = step.target_pose
        if step.target_joints is not None:
            joints = step.target_joints

    values = np.asarray(transformed, dtype=np.float32)
    plan = checker.build_plan(
        current_pose=current_pose,
        current_joints=current_joints,
        actions=values.tolist(),
        action_mode=action_mode,
    )
    for index, original_z, guarded_z in clamped:
        if index < len(plan.steps):
            plan.steps[index].warnings.append(
                "floor guard clamped VLA Z action "
                f"from {original_z:.6f} to {guarded_z:.6f}; gripper/action sequence retained"
            )
        print(
            f"pick-vla floor clamp step={index}: "
            f"raw_z={original_z:.6f} guarded_z={guarded_z:.6f}"
        )
    if clamped:
        plan.summary = checker._summary(plan.steps, plan.approved_by_safety)
    return values, plan


def _remove_gripper_commands(plan: Any) -> None:
    for step in plan.steps:
        step.gripper_m = None
        step.warnings.append("gripper command suppressed by --no-gripper")


def _enforce_loop_translation_limit(plan: Any, origin_pose: Any, limit_m: Optional[float]) -> None:
    if limit_m is None:
        return
    limit = float(limit_m)
    if limit <= 0:
        return
    for step in plan.steps:
        if step.target_pose is None:
            continue
        total = math.sqrt(
            (step.target_pose.x - origin_pose.x) ** 2
            + (step.target_pose.y - origin_pose.y) ** 2
            + (step.target_pose.z - origin_pose.z) ** 2
        )
        if total > limit:
            step.violations.append(f"loop translation too large: {total:.4f} > {limit:.4f}")
            plan.approved_by_safety = False
    if not plan.approved_by_safety:
        plan.summary = _summary_with_loop_limit(plan)


def _summary_with_loop_limit(plan: Any) -> str:
    violations = sum(len(step.violations) for step in plan.steps)
    warnings = sum(len(step.warnings) for step in plan.steps)
    return f"SAFETY REJECTED: {violations} violations, {warnings} warnings."


def _save_cycle_debug(
    save_dir_text: str,
    cycle: int,
    observation: dict[str, Any],
    response: Any,
    action_chunk: np.ndarray,
    plan: Any,
) -> None:
    save_dir = pathlib.Path(save_dir_text).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("observation/image", "overhead"),
        ("observation/wrist_image", "wrist"),
    ):
        image = observation.get(key)
        if image is not None:
            _save_rgb_image(save_dir / f"cycle_{cycle:04d}_{name}.png", np.asarray(image))
    payload = {
        "cycle": cycle,
        "prompt": observation.get("prompt"),
        "state": _jsonable(observation.get("observation/state")),
        "raw_response": _jsonable(response),
        "selected_actions": action_chunk.tolist(),
        "plan": plan.to_dict(),
    }
    with open(save_dir / f"cycle_{cycle:04d}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_rgb_image(path: pathlib.Path, image: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save observation debug images.") from exc
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _save_approval_snapshot(
    save_dir_text: str,
    cycle: int,
    observation: dict[str, Any],
    pose: Any,
    proposal: Any,
) -> None:
    save_dir = pathlib.Path(save_dir_text).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"approval_cycle_{cycle:04d}"
    for key, suffix in (("observation/image", "overhead"), ("observation/wrist_image", "wrist")):
        image = observation.get(key)
        if image is not None:
            _save_rgb_image(save_dir / f"{stem}_{suffix}.png", np.asarray(image))
    payload = {
        "phase": proposal.phase.value,
        "ee_pose": pose.as_list(),
        "xy_error_m": proposal.xy_error_m,
        "detection": _jsonable(proposal.detection.__dict__),
    }
    (save_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Approval evidence saved under: {save_dir / stem}")


def _maybe_run_pick_assist(
    args: argparse.Namespace,
    cfg: SafetyConfig,
    robot: RobotAdapter,
    logger: JsonlLogger,
) -> bool:
    if not args.pick_assist:
        return False
    pose = robot.read_ee_pose()
    if pose.z > float(args.pick_close_z_m):
        return False

    close_m = max(cfg.gripper_closed_m, min(cfg.gripper_open_m, float(args.pick_close_m)))
    lift_m = max(0.0, float(args.pick_lift_m))
    hold_s = max(0.0, float(args.pick_hold_s))
    target_z = min(cfg.workspace_z_m[1], pose.z + lift_m)
    lift_pose = pose.moved_by((0.0, 0.0, target_z - pose.z), (0.0, 0.0, 0.0))
    payload = {
        "trigger_pose": pose.as_list(),
        "close_z_m": float(args.pick_close_z_m),
        "close_m": close_m,
        "hold_s": hold_s,
        "lift_m": lift_m,
        "lift_pose": lift_pose.as_list(),
        "dry_run": cfg.dry_run,
    }
    logger.write("pick_assist", payload)
    print(
        "Pick assist triggered: "
        f"z={pose.z:.6f} <= {float(args.pick_close_z_m):.6f}, "
        f"close={close_m:.3f}m, lift={lift_m:.3f}m"
    )
    if cfg.dry_run:
        print("Dry run: pick assist did not command gripper or lift.")
        return True

    robot.command_gripper(close_m, effort_n_m=cfg.gripper_effort_n_m)
    time.sleep(hold_s)
    if lift_m > 0:
        robot.command_end_pose(lift_pose, speed_pct=cfg.speed_pct, move_mode="L")
        time.sleep(max(0.0, cfg.step_sleep_s))
        after_pose = robot.read_ee_pose()
        print(f"Pick assist lift after XYZ=({_fmt_xyz_values(after_pose.xyz())})")
    print("Pick assist complete; stopping VLA loop.")
    return True


def _parse_three_floats(text: str, flag_name: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{flag_name} must be three comma-separated numbers, e.g. 1,1,1")
    values = tuple(float(part) for part in parts)
    return values  # type: ignore[return-value]


def _require_ready_pose(robot: RobotAdapter, cfg: SafetyConfig, announce: bool = False) -> None:
    failures = _ready_pose_errors(robot, cfg)
    if failures:
        raise RuntimeError("Robot is outside the calibrated checkpoint ready pose: " + "; ".join(failures))
    if announce:
        joints = robot.read_joint_state()
        values = "unavailable" if joints is None else _fmt_list(joints.as_list())
        print(f"READY POSE VERIFIED: measured joints={values}; VLA execution unlocked.")


def _move_robot_to_ready(
    robot: RobotAdapter,
    cfg: SafetyConfig,
    calibration: PickCalibration,
    logger: Optional[JsonlLogger] = None,
    approval_word: str = "READY",
) -> None:
    """Follow a measured joint-waypoint route to the policy ready pose."""
    try:
        current, path, target = _preflight_ready_path(robot, cfg, calibration)
    except Exception as exc:
        raise ArmInitializationRefused(
            f"Automatic ready return preflight refused before enabling motors: {exc}"
        ) from exc

    print("\n=== AUTOMATIC CHECKPOINT-READY RETURN ===")
    print(f"Current joints: {_fmt_list(current.as_list())}")
    print(f"Target joints:  {_fmt_list(target)}")
    print(
        f"Waypoints={len(path)}, speed={cfg.ready_return_speed_pct}%, "
        f"max target step={cfg.ready_return_max_step_deg:.3f}deg, "
        f"max measured speed={cfg.ready_return_max_joint_speed_deg_s:.1f}deg/s."
    )
    if not sys.stdin.isatty():
        raise ArmInitializationRefused(
            "Interactive terminal is required for automatic ready-return approval; no command sent."
        )
    entered = input(
        f"Type {approval_word!r} to attach/enable and move through this checked ready path: "
    ).strip()
    if entered != approval_word:
        raise ArmInitializationRefused("Ready return cancelled by operator; no command sent.")

    if logger is not None:
        logger.write(
            "ready_return_started",
            {
                "mode": "calibrated_joint_waypoints",
                "current_joints": current.as_list(),
                "target_joints": list(target),
                "waypoints": [list(point) for point in path],
                "max_step_deg": cfg.ready_return_max_step_deg,
            },
        )

    print(robot.enable(speed_pct=cfg.ready_return_speed_pct, move_mode="J"))
    deadline = time.monotonic() + cfg.ready_return_total_timeout_s
    command_count = 0
    last_progress_print = 0.0
    try:
        streamed_path = _joint_path_samples(
            current.values_deg,
            path,
            cfg.ready_return_max_step_deg,
        )
        for waypoint_index, sample, sample_count, next_values in streamed_path:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"ready return timed out at waypoint {waypoint_index + 1}")
            waypoint = path[waypoint_index]
            status = robot.read_arm_status()
            if status.get("fault") or (cfg.require_status_available and not status.get("available", False)):
                raise RuntimeError(f"arm status is not healthy: {status}")
            measured = robot.read_joint_state()
            if measured is None:
                raise RuntimeError("joint feedback was lost")
            max_remaining = max(
                abs(waypoint[i] - measured.values_deg[i]) for i in range(6)
            )
            _require_joint_values_in_limits(next_values, cfg, "next streamed target")
            step_deadline = min(deadline, time.monotonic() + cfg.ready_return_step_timeout_s)
            previous = measured
            previous_time = time.monotonic()
            while time.monotonic() < step_deadline:
                robot.command_joints(JointState(next_values), speed_pct=cfg.ready_return_speed_pct)
                command_count += 1
                time.sleep(1.0 / cfg.control_hz)
                updated = robot.read_joint_state()
                if updated is None:
                    raise RuntimeError("joint feedback was lost while tracking ready substep")
                now = time.monotonic()
                elapsed = max(1e-6, now - previous_time)
                max_speed = max(
                    abs(a - b) / elapsed for a, b in zip(updated.values_deg, previous.values_deg)
                )
                if max_speed > cfg.ready_return_max_joint_speed_deg_s:
                    raise RuntimeError(
                        f"joint speed {max_speed:.2f} deg/s exceeded "
                        f"{cfg.ready_return_max_joint_speed_deg_s:.2f} deg/s; "
                        f"elapsed={elapsed:.4f}s, previous={_fmt_list(previous.as_list())}, "
                        f"updated={_fmt_list(updated.as_list())}"
                    )
                _require_joint_values_in_limits(updated.values_deg, cfg, "measured")
                pose_error = _ready_return_pose_error(robot.read_ee_pose(), cfg)
                if pose_error:
                    raise RuntimeError(pose_error)
                if max(abs(a - b) for a, b in zip(updated.values_deg, next_values)) <= (
                    cfg.ready_return_tracking_tolerance_deg
                ):
                    break
                previous = updated
                previous_time = now
            else:
                raise RuntimeError(
                    f"ready substep tracking timeout after {cfg.ready_return_step_timeout_s:.2f}s; "
                    f"target={_fmt_list(next_values)}"
                )

            now = time.monotonic()
            if now - last_progress_print >= 1.0:
                pose = robot.read_ee_pose()
                print(
                    f"Ready progress: waypoint={waypoint_index + 1}/{len(path)}, "
                    f"sample={sample}/{sample_count}, commands={command_count}, "
                    f"remaining={max_remaining:.2f}deg, "
                    f"EE=({_fmt_xyz_values(pose.xyz())})"
                )
                last_progress_print = now

        measured = robot.read_joint_state()
        if measured is None:
            raise RuntimeError("joint feedback was lost at ready completion")
        max_error = max(abs(a - b) for a, b in zip(measured.values_deg, target))
        print(f"READY RETURN COMPLETE: measured joints={_fmt_list(measured.as_list())}")
        if cfg.cartesian_execution_mode == "joint_ik":
            print(robot.hold_current_joints())
        else:
            print(robot.activate_cartesian_hold(speed_pct=cfg.speed_pct, move_mode="L"))
        if logger is not None:
            logger.write(
                "ready_return_complete",
                {"joints": measured.as_list(), "commands": command_count, "max_error_deg": max_error},
            )
        return
    except BaseException as exc:
        if logger is not None:
            logger.write(
                "ready_return_aborted",
                {"commands": command_count, "error_type": type(exc).__name__, "error": str(exc)},
            )
        # The run-loop exception boundary clears queued motion and refreshes a
        # measured-position hold. It never auto-latches E-stop because that can
        # remove gravity-holding torque on this arm.
        raise

    raise RuntimeError(f"ready return timed out after {cfg.ready_return_total_timeout_s:.1f}s")


def _preflight_ready_path(
    robot: RobotAdapter,
    cfg: SafetyConfig,
    calibration: PickCalibration,
) -> tuple[JointState, list[tuple[float, ...]], tuple[float, ...]]:
    calibration.require_ready_path()
    current = robot.read_joint_state()
    if current is None:
        raise RuntimeError("joint feedback is unavailable")
    path = [
        tuple(float(value) for value in point)
        for point in calibration.resolved_ready_path_joints_deg()
    ]
    target = path[-1]
    _require_joint_values_in_limits(current.values_deg, cfg, "current")
    for index, waypoint in enumerate(path):
        _require_joint_values_in_limits(waypoint, cfg, f"ready waypoint {index}")
    measured_pose = robot.read_ee_pose()
    pose_error = _ready_return_pose_error(measured_pose, cfg)
    if pose_error:
        raise RuntimeError(f"unsafe start EE pose: {pose_error}")
    _validate_ready_path_fk(current.values_deg, path, measured_pose, cfg)
    return current, path, target


def _require_joint_values_in_limits(values: Sequence[float], cfg: SafetyConfig, label: str) -> None:
    for index, value in enumerate(values, start=1):
        low, high = cfg.joint_limits_deg[f"j{index}"]
        if not low <= float(value) <= high:
            raise RuntimeError(f"{label} J{index}={float(value):.3f} outside [{low}, {high}]")


def _validate_ready_path_fk(
    current_joints: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    measured_pose: EEPose,
    cfg: SafetyConfig,
) -> None:
    """Sample the complete route with Piper SDK FK before enabling motors."""
    try:
        from piper_sdk import C_PiperForwardKinematics  # type: ignore
    except Exception as exc:  # pragma: no cover - installed for real execution
        raise RuntimeError("piper_sdk FK is required to validate automatic ready return") from exc

    fk = C_PiperForwardKinematics(cfg.dh_is_offset)

    def pose_for(joints_deg: Sequence[float]) -> EEPose:
        links = fk.CalFK([math.radians(float(value)) for value in joints_deg])
        x_mm, y_mm, z_mm, rx, ry, rz = links[-1]
        return EEPose(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, rx, ry, rz)

    fk_current = pose_for(current_joints)
    fk_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(fk_current.xyz(), measured_pose.xyz())))
    if fk_error > 0.020:
        raise RuntimeError(
            f"Piper FK disagrees with measured EE feedback by {fk_error:.4f}m; "
            "check dh_is_offset and feedback freshness before auto-ready"
        )

    sample_index = 0
    for waypoint_index, sample, count, joints in _joint_path_samples(
        current_joints,
        waypoints,
        cfg.ready_return_max_step_deg,
    ):
        _require_joint_values_in_limits(joints, cfg, f"ready path sample {sample_index}")
        pose = pose_for(joints)
        error = _ready_return_pose_error(pose, cfg)
        if error:
            raise RuntimeError(
                f"Ready path FK rejected at waypoint {waypoint_index}, sample {sample}/{count}: {error}"
            )
        sample_index += 1


def _joint_path_samples(
    current_joints: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    max_step_deg: float,
) -> list[tuple[int, int, int, tuple[float, ...]]]:
    """Return the exact proportional joint samples used by preflight and execution."""
    if max_step_deg <= 0.0:
        raise ValueError("max_step_deg must be positive")
    result: list[tuple[int, int, int, tuple[float, ...]]] = []
    previous = tuple(float(value) for value in current_joints)
    for waypoint_index, waypoint_raw in enumerate(waypoints):
        waypoint = tuple(float(value) for value in waypoint_raw)
        max_delta = max(abs(a - b) for a, b in zip(waypoint, previous))
        count = max(1, int(math.ceil(max_delta / max_step_deg)))
        for sample in range(1, count + 1):
            fraction = sample / count
            joints = tuple(
                previous[i] + fraction * (waypoint[i] - previous[i]) for i in range(6)
            )
            result.append((waypoint_index, sample, count, joints))
        previous = waypoint
    return result


def _ready_return_pose_error(pose: Any, cfg: SafetyConfig) -> Optional[str]:
    tolerance = cfg.ready_return_workspace_tolerance_m
    ready_x = cfg.ready_return_workspace_x_m or cfg.workspace_x_m
    ready_y = cfg.ready_return_workspace_y_m or cfg.workspace_y_m
    ready_z = cfg.ready_return_workspace_z_m or cfg.workspace_z_m
    for axis, value, bounds in (
        ("x", pose.x, ready_x),
        ("y", pose.y, ready_y),
        ("z", pose.z, ready_z),
    ):
        if not bounds[0] - tolerance <= value <= bounds[1] + tolerance:
            return f"{axis}={value:.6f} outside ready-return workspace {bounds} +/- {tolerance:.4f}"
    if pose.z < cfg.min_z_m - tolerance:
        return f"z={pose.z:.6f} below ready-return min_z {cfg.min_z_m:.6f}"
    from .safety import workspace_floor_error

    if cfg.ready_return_enforce_workspace_floor_polygon:
        floor_error = workspace_floor_error(pose.xyz(), cfg)
        if floor_error:
            return floor_error
    for plane in cfg.safety_planes:
        norm = math.sqrt(sum(value * value for value in plane.normal))
        if norm <= 1e-12:
            return f"safety plane {plane.name} has a zero normal"
        signed_distance = sum(
            plane.normal[index] * (pose.xyz()[index] - plane.point[index])
            for index in range(3)
        ) / norm
        if signed_distance < plane.margin_m - tolerance:
            return (
                f"safety plane {plane.name} distance {signed_distance:.6f} "
                f"below margin {plane.margin_m:.6f}"
            )
    if cfg.tool_points_m:
        from .safety import _rpy_rotation_matrix

        rot = _rpy_rotation_matrix(pose.rx, pose.ry, pose.rz)
        floor = cfg.table_z_m + cfg.table_margin_m
        for index, local in enumerate(cfg.tool_points_m):
            world = tuple(
                pose.xyz()[row] + sum(rot[row][column] * local[column] for column in range(3))
                for row in range(3)
            )
            world_z = world[2]
            if world_z < floor - tolerance:
                return f"tool point {index} z={world_z:.6f} below table clearance {floor:.6f}"
            if cfg.ready_return_enforce_workspace_floor_polygon:
                floor_error = workspace_floor_error(world, cfg)
                if floor_error:
                    return f"tool point {index} {floor_error}"
    return None


def _ready_pose_errors(robot: RobotAdapter, cfg: SafetyConfig) -> list[str]:
    joints = robot.read_joint_state()
    if joints is None:
        return ["joint feedback is unavailable"]
    failures = []
    for index, (actual, expected, tolerance) in enumerate(
        zip(joints.values_deg, cfg.expected_ready_joints_deg, cfg.ready_joint_tolerance_deg), start=1
    ):
        if abs(actual - expected) > tolerance:
            failures.append(
                f"J{index}={actual:.2f}deg, expected {expected:.2f}+/-{tolerance:.2f}deg"
            )
    return failures


def _sha256_file(path_text: str) -> str:
    path = pathlib.Path(path_text).expanduser()
    if not path.is_file():
        raise RuntimeError(f"Expected norm-stats file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pure_vla_gate(path_text: str) -> None:
    if not path_text:
        raise RuntimeError("pure-vla execution requires --pure-vla-gate-file")
    payload = json.loads(pathlib.Path(path_text).expanduser().read_text(encoding="utf-8"))
    failures = []
    if payload.get("dataset_repo_id") != EXPECTED_DATASET_REPO_ID:
        failures.append("dataset_repo_id mismatch")
    if int(payload.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        failures.append("checkpoint_step mismatch")
    if not payload.get("image_sensitivity_pass", False):
        failures.append("image sensitivity test has not passed")
    if int(payload.get("dry_run_trials", 0)) < 20:
        failures.append("fewer than 20 clean dry-run trials")
    if int(payload.get("hybrid_trials", 0)) < 10 or int(payload.get("hybrid_successes", 0)) < 8:
        failures.append("hybrid acceptance is below 8/10")
    if failures:
        raise RuntimeError("pure-vla gate rejected execution: " + "; ".join(failures))


def _request_grasp_approval(args: argparse.Namespace, cfg: SafetyConfig, pose: Any, proposal: Any) -> bool:
    print("\n=== GRASP APPROVAL REQUIRED ===")
    print(f"EE pose: {_fmt_list(pose.as_list())}")
    print(f"Cylinder base XY: {_fmt_list(proposal.detection.base_xy_m)}")
    print(f"XY error: {proposal.xy_error_m:.6f} m")
    print(
        f"Planned grasp width: "
        f"{cfg.cylinder_diameter_m - cfg.grasp_width_margin_m:.6f} m; "
        f"test lift={cfg.hybrid_test_lift_m:.3f} m; full lift={cfg.hybrid_total_lift_m:.3f} m"
    )
    if cfg.dry_run:
        print("Dry run: approval simulated; no gripper/lift commands will be sent.")
        return True
    if not sys.stdin.isatty():
        raise RuntimeError("interactive terminal is required for per-grasp manual approval")
    entered = input(f"Type {args.grasp_approval_word!r} to close the gripper: ").strip()
    return entered == args.grasp_approval_word


def _request_arm_test_approval(args: argparse.Namespace, plan: Any) -> bool:
    mode_label = {
        "arm-test": "CHECKPOINT-READY ARM TEST",
        "xy-vla": "XY VLA CHUNK",
        "xyz-vla": "XYZ VLA CHUNK",
        "pick-vla": "BOUNDED VLA PICK RUN",
    }[args.mode]
    print(f"\n=== {mode_label} APPROVAL ===")
    if args.mode == "pick-vla":
        print(
            "XYZ and gripper may move; RPY is suppressed. Approval covers the complete "
            f"run of up to {int(args.max_cycles)} camera-feedback cycles."
        )
    elif args.mode == "xyz-vla":
        print("XYZ may move; RPY and gripper are suppressed.")
    else:
        print("Only XY may move; Z, RPY, and gripper are suppressed.")
    print(
        f"Maximum checked XYZ step: "
        f"{float(args.pick_vla_max_xyz_m if args.mode == 'pick-vla' else args.arm_test_max_xy_m):.6f} m; "
        f"summary: {plan.summary}"
    )
    if not sys.stdin.isatty():
        raise RuntimeError("interactive terminal is required for VLA motion approval")
    axes = "XYZ+gripper" if args.mode == "pick-vla" else "XYZ" if args.mode == "xyz-vla" else "XY"
    approval_word = (
        args.pick_vla_approval_word if args.mode == "pick-vla" else args.arm_test_approval_word
    )
    entered = input(
        f"Type {approval_word!r} to execute "
        f"{len(plan.steps)} VLA {axes} action(s): "
    ).strip()
    return entered == approval_word


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenPI/pi0 action-chunk loop on Piper hardware using the same "
            "observation/action boundary as custom_scripts_piper/eval_piper_vla.py."
        )
    )
    parser.add_argument("--config", default="configs/safety.example.yaml")
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "--attach-enabled-can",
        action="store_true",
        help=(
            "Explicitly attach to an arm already held in CAN/MoveJ with all motors enabled. "
            "Seeds the measured joints as the first target and never sends reset/E-stop resume."
        ),
    )
    parser.add_argument(
        "--vendor-teaching-bootstrap",
        action="store_true",
        help=(
            "On real SDK execution only, explicitly run the first-start stop/damping, "
            "safe-joint-window, disable+resume, CAN/MoveJ, and enable sequence used by "
            "test_ctrlPiperJoint_can0_2.py. Requires an interactive approval and the arm descends."
        ),
    )
    parser.add_argument("--bootstrap-approval-word", default="BOOTSTRAP")
    parser.add_argument(
        "--bootstrap-timeout-s",
        type=float,
        default=60.0,
        help="Maximum wait for each vendor bootstrap transition.",
    )
    parser.add_argument(
        "--transport",
        choices=("ros", "sdk"),
        default="sdk",
        help="Real-hardware transport. SDK uses Piper CAN directly; ROS remains an optional bridge.",
    )
    parser.add_argument("--rosbridge-host", default="127.0.0.1")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default=EXPECTED_PROMPT)
    parser.add_argument(
        "--mode",
        choices=("observe", "arm-test", "xy-vla", "xyz-vla", "pick-vla", "hybrid", "pure-vla"),
        default="observe",
    )
    parser.add_argument("--calibration", default="")
    parser.add_argument("--expected-norm-stats", default=DEFAULT_NORM_STATS_PATH)
    parser.add_argument("--pure-vla-gate-file", default="")
    parser.add_argument("--grasp-approval-word", default="CLOSE")
    parser.add_argument("--arm-test-approval-word", default="MOVE")
    parser.add_argument("--pick-vla-approval-word", default="PICK")
    parser.add_argument(
        "--arm-test-max-xy-m",
        type=float,
        default=0.0005,
        help="Maximum VLA-derived XY step in arm-test; hard-limited to 0.003 m.",
    )
    parser.add_argument(
        "--arm-test-tracking-tolerance-m",
        type=float,
        default=0.00055,
        help="Arm-test Cartesian settle tolerance; hard-limited to 0.001 m.",
    )
    parser.add_argument(
        "--pick-vla-tracking-tolerance-m",
        type=float,
        default=0.002,
        help=(
            "Measured target tolerance for live-rebased pick-vla actions; "
            "hard-limited to 0.005 m. The measured pose is still checked "
            "against the workspace and safety floor after every action."
        ),
    )
    parser.add_argument(
        "--xyz-vla-max-z-m",
        type=float,
        default=0.002,
        help="Maximum VLA-derived Z step in xyz-vla; hard-limited to 0.003 m.",
    )
    parser.add_argument(
        "--pick-vla-max-xyz-m",
        type=float,
        default=0.05,
        help=(
            "Sanity ceiling for the checkpoint-scaled XYZ action in pick-vla; "
            "0.05 m leaves the observed policy output unclipped."
        ),
    )
    parser.add_argument("--ready-approval-word", default="READY")
    parser.add_argument(
        "--auto-ready",
        action="store_true",
        help=(
            "Before policy inference, move from the current measured pose through optional "
            "ready_path_joints_deg to the checkpoint ready pose. An empty path moves directly. Without --execute, "
            "validate the complete FK path only. Real motion requires --execute --yes."
        ),
    )
    parser.add_argument("--overhead-camera-source", default="")
    parser.add_argument("--wrist-camera-source", default="")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Execute only this prefix of the 20-action policy output before recapturing both cameras.",
    )
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--speed-pct", type=int, default=10)
    parser.add_argument("--step-sleep-s", type=float, default=None)
    parser.add_argument("--cycle-sleep-s", type=float, default=0.0)
    parser.add_argument("--camera-retries", type=int, default=5)
    parser.add_argument("--camera-retry-sleep-s", type=float, default=0.10)
    parser.add_argument("--camera-warmup-frames", type=int, default=5)
    parser.add_argument("--camera-max-identical-frames", type=int, default=2)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="FourCC requested from V4L2 cameras; MJPG reduces shared USB bandwidth.",
    )
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--camera-read-timeout-ms", type=int, default=3000)
    parser.add_argument("--max-step-xyz-m", type=float, default=None)
    parser.add_argument(
        "--max-loop-translation-m",
        type=float,
        default=0.15,
        help="Stop if a target moves farther than this from the loop start pose. Set <=0 to disable.",
    )
    parser.add_argument(
        "--action-xyz-signs",
        default="1,1,1",
        help="Comma-separated xyz signs applied to VLA actions before safety, e.g. -1,1,1.",
    )
    parser.add_argument(
        "--save-observation-dir",
        default="",
        help="Optional directory for per-cycle camera images, state, raw actions, and checked plans.",
    )
    parser.add_argument(
        "--pick-assist",
        action="store_true",
        help="Close the gripper and lift once the end effector descends below --pick-close-z-m.",
    )
    parser.add_argument("--pick-close-z-m", type=float, default=0.155)
    parser.add_argument("--pick-close-m", type=float, default=0.030)
    parser.add_argument("--pick-hold-s", type=float, default=1.0)
    parser.add_argument("--pick-lift-m", type=float, default=0.060)
    parser.add_argument("--no-gripper", action="store_true", help="Suppress gripper commands for arm-only debugging.")
    parser.add_argument("--execute", action="store_true", help="Send commands to real Piper. Default is dry-run.")
    parser.add_argument("--yes", action="store_true", help="Required together with --execute.")
    parser.add_argument("--enable", action="store_true", help="Enable arm / set MoveL even in dry-run.")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic mock robot adapter.")
    return parser.parse_args(argv)


def main() -> None:
    raise SystemExit(run_real_loop(parse_args()))


if __name__ == "__main__":
    main()
