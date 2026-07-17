from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any, Optional, Sequence

import numpy as np

from .camera_adapter import PersistentCamera
from .config import load_config
from .executor import PlanExecutor
from .logging_utils import JsonlLogger
from .piper_adapter import MockPiperAdapter, PiperSDKAdapter, RobotAdapter
from .policy_adapter import OpenPIPolicyClient, response_to_json
from .safety import SafetyChecker
from .types import GripperState, SafetyConfig


ACTION_MODE = "robosuite_osc_pose"


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
    qpos = max(0.0, min(cfg.robosuite_gripper_qpos_max_m, gripper.opening_m / 2.0))
    return [qpos, qpos]


def build_real_observation(
    robot: RobotAdapter,
    cfg: SafetyConfig,
    prompt: str,
    overhead_camera: PersistentCamera,
    wrist_camera: PersistentCamera,
) -> dict[str, Any]:
    """Build the same policy-facing observation shape used by the robosuite eval loop."""
    obs: dict[str, Any] = {
        "prompt": prompt,
        "observation/state": np.asarray(robot_state_vector(robot, cfg), dtype=np.float32),
    }
    overhead = overhead_camera.read_rgb()
    wrist = wrist_camera.read_rgb()
    if overhead is not None:
        obs["observation/image"] = overhead
    if wrist is not None:
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


def apply_action_xyz_signs(actions: np.ndarray, signs_text: str) -> np.ndarray:
    signs = _parse_three_floats(signs_text, "--action-xyz-signs")
    transformed = np.asarray(actions, dtype=np.float32).copy()
    transformed[:, 0] *= signs[0]
    transformed[:, 1] *= signs[1]
    transformed[:, 2] *= signs[2]
    return transformed


def run_real_loop(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.can_name = args.can
    cfg.speed_pct = int(args.speed_pct)
    if args.step_sleep_s is not None:
        cfg.step_sleep_s = float(args.step_sleep_s)
    cfg.dry_run = not bool(args.execute)
    if args.max_step_xyz_m is not None:
        step = float(args.max_step_xyz_m)
        cfg.max_step_xyz_m = (step, step, step)
    if args.max_horizon is not None:
        cfg.max_horizon = int(args.max_horizon)

    logger = JsonlLogger(cfg.log_dir)
    robot: RobotAdapter = MockPiperAdapter() if args.mock else PiperSDKAdapter(cfg)
    print(robot.connect())
    if args.execute or args.enable:
        print(robot.enable(speed_pct=cfg.speed_pct, move_mode="L"))

    if args.execute and not args.yes:
        raise RuntimeError("Refusing to send real Piper commands without --yes.")

    client = OpenPIPolicyClient(host=args.policy_host, port=args.policy_port)
    executor = PlanExecutor(robot, cfg, logger=logger)
    checker = SafetyChecker(cfg)
    overhead_camera = PersistentCamera(
        args.overhead_camera_source,
        retries=args.camera_retries,
        retry_sleep_s=args.camera_retry_sleep_s,
        label="overhead",
    )
    wrist_camera = PersistentCamera(
        args.wrist_camera_source,
        retries=args.camera_retries,
        retry_sleep_s=args.camera_retry_sleep_s,
        label="wrist",
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
            )
            if result_code is not None:
                return result_code
    finally:
        overhead_camera.close()
        wrist_camera.close()

    print("Finished Piper VLA loop.")
    return 0


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
        )
        response = client.client.infer(observation)
        action_chunk = select_action_chunk(response, args.chunk_size, cfg.max_horizon)
        if args.action_xyz_signs != "1,1,1":
            action_chunk = apply_action_xyz_signs(action_chunk, args.action_xyz_signs)
        plan = checker.build_plan(
            current_pose=pose,
            current_joints=joints,
            actions=action_chunk.tolist(),
            action_mode=ACTION_MODE,
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

        result = executor.execute(plan, human_approved=True, dry_run=cfg.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result.get("ok"):
            print("Execution failed or was blocked; stopping loop.")
            return 3
        if _maybe_run_pick_assist(args, cfg, robot, logger):
            return 0

        if args.cycle_sleep_s > 0:
            elapsed = time.time() - started
            time.sleep(max(0.0, float(args.cycle_sleep_s) - elapsed))
        return None


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenPI/pi0 action-chunk loop on Piper hardware using the same "
            "observation/action boundary as custom_scripts_piper/eval_piper_vla.py."
        )
    )
    parser.add_argument("--config", default="configs/safety.example.yaml")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the white cylinder.")
    parser.add_argument("--overhead-camera-source", default="")
    parser.add_argument("--wrist-camera-source", default="")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--speed-pct", type=int, default=10)
    parser.add_argument("--step-sleep-s", type=float, default=None)
    parser.add_argument("--cycle-sleep-s", type=float, default=0.0)
    parser.add_argument("--camera-retries", type=int, default=5)
    parser.add_argument("--camera-retry-sleep-s", type=float, default=0.10)
    parser.add_argument("--max-step-xyz-m", type=float, default=None)
    parser.add_argument(
        "--max-loop-translation-m",
        type=float,
        default=0.05,
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
