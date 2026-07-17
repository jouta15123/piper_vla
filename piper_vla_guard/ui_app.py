from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .actions import SUPPORTED_ACTION_MODES, parse_action_json
from .camera_adapter import capture_rgb_frame
from .config import load_config
from .executor import PlanExecutor
from .logging_utils import JsonlLogger
from .piper_adapter import MockPiperAdapter, PiperSDKAdapter, RobotAdapter
from .policy_adapter import OpenPIPolicyClient, response_to_json, actions_to_json, parse_state_json
from .safety import SafetyChecker
from .types import EEPose, GripperState, JointState, SafetyConfig, TrajectoryPlan


SAMPLE_ACTION_JSON = json.dumps(
    {
        "actions": [
            [0.001, 0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    },
    indent=2,
)


class AppContext:
    def __init__(self, cfg: SafetyConfig, real_hardware: bool = False):
        self.cfg = cfg
        self.real_hardware = real_hardware
        self.robot: Optional[RobotAdapter] = None
        self.logger = JsonlLogger(cfg.log_dir)

    def set_runtime(self, can_name: str, dry_run: bool, real_hardware: bool) -> None:
        self.cfg.can_name = can_name
        self.cfg.dry_run = bool(dry_run)
        self.real_hardware = bool(real_hardware)

    def ensure_robot(self) -> RobotAdapter:
        if self.robot is None:
            self.robot = PiperSDKAdapter(self.cfg) if self.real_hardware else MockPiperAdapter()
            self.robot.connect()
        return self.robot

    def reconnect(self) -> str:
        self.robot = PiperSDKAdapter(self.cfg) if self.real_hardware else MockPiperAdapter()
        msg = self.robot.connect()
        self.logger.write("connect", {"real_hardware": self.real_hardware, "can_name": self.cfg.can_name})
        return msg


def create_ui(
    cfg: SafetyConfig,
    real_hardware: bool = False,
    robot_camera_source_default: str = "",
    overhead_camera_source_default: str = "",
):
    import gradio as gr

    ctx = AppContext(cfg, real_hardware=real_hardware)

    def connect(can_name: str, dry_run: bool, use_real_hardware: bool):
        try:
            ctx.set_runtime(can_name, dry_run, use_real_hardware)
            msg = ctx.reconnect()
            pose_df, joint_df, status_md = read_state_tables(ctx.robot)
            return f"OK: {msg}", pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    def enable_arm(speed_pct: int):
        try:
            robot = ctx.ensure_robot()
            msg = robot.enable(speed_pct=int(speed_pct), move_mode="L")
            ctx.logger.write("enable", {"speed_pct": int(speed_pct)})
            return f"OK: {msg}"
        except Exception as exc:
            return _err(exc)

    def refresh_state():
        try:
            robot = ctx.ensure_robot()
            pose_df, joint_df, status_md = read_state_tables(robot)
            return "State refreshed.", pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    def query_openpi(host: str, port: int, prompt: str, image: Any, wrist_image: Any, state_json: str):
        try:
            state = parse_state_json(state_json)
            client = OpenPIPolicyClient(host=host, port=int(port))
            response = client.infer(prompt=prompt, image=image, wrist_image=wrist_image, state=state)
            if "actions" not in response:
                raise RuntimeError(f"OpenPI response has no actions key: {list(response.keys())}")
            text = actions_to_json(response["actions"])
            return text, f"OK: received actions with outer length {len(response['actions'])}", response_to_json(response)
        except Exception as exc:
            return SAMPLE_ACTION_JSON, _err(exc), ""

    def capture_robot_inputs(robot_camera_source: str, overhead_camera_source: str):
        try:
            robot = ctx.ensure_robot()
            pose = robot.read_ee_pose()
            joints = robot.read_joint_state()
            gripper = robot.read_gripper_state()
            state = state_json_from_robot(pose, joints, gripper, ctx.cfg)
            wrist = capture_rgb_frame(robot_camera_source)
            overhead = capture_rgb_frame(overhead_camera_source)
            ctx.logger.write(
                "capture_robot_inputs",
                {
                    "robot_camera_source": robot_camera_source,
                    "overhead_camera_source": overhead_camera_source,
                    "state": state,
                    "has_wrist_image": wrist is not None,
                    "has_overhead_image": overhead is not None,
                },
            )
            return overhead, wrist, state, "OK: captured robot inputs."
        except Exception as exc:
            return None, None, "", _err(exc)

    def check_trajectory(action_json: str, action_mode: str):
        try:
            robot = ctx.ensure_robot()
            current_pose = robot.read_ee_pose()
            current_joints = robot.read_joint_state()
            actions = parse_action_json(action_json, action_mode)
            checker = SafetyChecker(ctx.cfg)
            plan = checker.build_plan(
                current_pose=current_pose,
                current_joints=current_joints,
                actions=actions,
                action_mode=action_mode,
            )
            ctx.logger.write("plan_checked", {"plan": plan.to_dict()})
            return (
                plan_report_markdown(plan),
                plan_dataframe(plan),
                trajectory_plot(plan),
                plan.to_dict(),
            )
        except Exception as exc:
            return (_err(exc), pd.DataFrame(), blank_plot(str(exc)), None)

    def execute_plan(approved: bool, dry_run: bool, plan_data: Optional[Dict[str, Any]]):
        try:
            if plan_data is None:
                return "No checked plan available.", empty_pose_df(), empty_joint_df(), ""
            ctx.cfg.dry_run = bool(dry_run)
            robot = ctx.ensure_robot()
            plan = TrajectoryPlan.from_dict(plan_data)
            executor = PlanExecutor(robot, ctx.cfg, logger=ctx.logger)
            result = executor.execute(plan, human_approved=bool(approved), dry_run=bool(dry_run))
            pose_df, joint_df, status_md = read_state_tables(robot)
            return execution_markdown(result), pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    def estop():
        try:
            robot = ctx.ensure_robot()
            msg = robot.emergency_stop()
            ctx.logger.write("emergency_stop", {})
            pose_df, joint_df, status_md = read_state_tables(robot)
            return f"OK: {msg}", pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    def resume():
        try:
            robot = ctx.ensure_robot()
            msg = robot.resume()
            ctx.logger.write("resume", {})
            pose_df, joint_df, status_md = read_state_tables(robot)
            return f"OK: {msg}", pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    def disable():
        try:
            robot = ctx.ensure_robot()
            msg = robot.disable()
            ctx.logger.write("disable", {})
            pose_df, joint_df, status_md = read_state_tables(robot)
            return f"OK: {msg}", pose_df, joint_df, status_md
        except Exception as exc:
            return _err(exc), empty_pose_df(), empty_joint_df(), ""

    with gr.Blocks(title="Piper VLA Guard") as demo:
        plan_state = gr.State(None)
        gr.Markdown(
            "# Piper VLA Guard\n"
            "Human-in-the-loop trajectory check before sending VLA/OpenPI/pi0 actions to Piper. "
            "Start with dry-run enabled. This UI is a guard layer, not a certified safety controller."
        )

        with gr.Tab("1. Robot"):
            with gr.Row():
                can_name = gr.Textbox(value=cfg.can_name, label="CAN name")
                dry_run_box = gr.Checkbox(value=cfg.dry_run, label="Dry run: do not send robot commands")
                use_real = gr.Checkbox(value=real_hardware, label="Use real Piper hardware")
                speed_pct = gr.Slider(1, 50, value=cfg.speed_pct, step=1, label="Move speed percent")
            with gr.Row():
                connect_btn = gr.Button("Connect")
                enable_btn = gr.Button("Enable arm / set MoveL")
                refresh_btn = gr.Button("Read state")
                estop_btn = gr.Button("EMERGENCY STOP")
                resume_btn = gr.Button("Resume E-stop")
                disable_btn = gr.Button("Disable arm")
            robot_status = gr.Markdown("Not connected.")
            with gr.Row():
                pose_table = gr.Dataframe(value=empty_pose_df(), label="Current end-effector pose", interactive=False)
                joint_table = gr.Dataframe(value=empty_joint_df(), label="Current joints", interactive=False)
            arm_status_md = gr.Markdown("")

        with gr.Tab("2. Policy / Action chunk"):
            action_mode = gr.Dropdown(
                choices=SUPPORTED_ACTION_MODES,
                value="robosuite_osc_pose",
                label="Action interpretation",
            )
            action_json = gr.Textbox(value=SAMPLE_ACTION_JSON, lines=14, label="Action JSON")
            gr.Markdown("Optional: query an OpenPI policy server and fill Action JSON automatically.")
            with gr.Row():
                host = gr.Textbox(value="localhost", label="OpenPI host")
                port = gr.Number(value=8000, precision=0, label="OpenPI port")
            prompt = gr.Textbox(value="pick up the white cylinder.", label="Prompt")
            with gr.Row():
                robot_camera_source = gr.Textbox(value=robot_camera_source_default, label="Robot camera source")
                overhead_camera_source = gr.Textbox(value=overhead_camera_source_default, label="Overhead camera source")
                capture_inputs_btn = gr.Button("Capture robot inputs")
            with gr.Row():
                image = gr.Image(type="numpy", label="Observation / overhead image")
                wrist_image = gr.Image(type="numpy", label="Robot / wrist image")
            state_json = gr.Textbox(value="", lines=3, label="State JSON, optional")
            query_btn = gr.Button("Query OpenPI")
            openpi_status = gr.Markdown("")
            raw_openpi_response = gr.Textbox(value="", lines=10, label="Raw OpenPI response", interactive=False)

        with gr.Tab("3. Safety check"):
            check_btn = gr.Button("Check trajectory")
            report = gr.Markdown("No plan checked yet.")
            with gr.Row():
                plan_table = gr.Dataframe(value=pd.DataFrame(), label="Checked steps", interactive=False)
                plot = gr.Plot(label="Trajectory preview")

        with gr.Tab("4. Approve and execute"):
            approve_box = gr.Checkbox(value=False, label="I inspected the trajectory and approve this plan")
            execute_btn = gr.Button("Execute approved plan")
            execute_status = gr.Markdown("")
            gr.Markdown(
                "Execution is blocked if the plan has safety violations or approval is missing. "
                "When dry-run is checked, the UI logs the plan but sends no Piper command."
            )

        connect_btn.click(connect, [can_name, dry_run_box, use_real], [robot_status, pose_table, joint_table, arm_status_md])
        enable_btn.click(enable_arm, [speed_pct], [robot_status])
        refresh_btn.click(refresh_state, None, [robot_status, pose_table, joint_table, arm_status_md])
        estop_btn.click(estop, None, [robot_status, pose_table, joint_table, arm_status_md])
        resume_btn.click(resume, None, [robot_status, pose_table, joint_table, arm_status_md])
        disable_btn.click(disable, None, [robot_status, pose_table, joint_table, arm_status_md])
        capture_inputs_btn.click(
            capture_robot_inputs,
            [robot_camera_source, overhead_camera_source],
            [image, wrist_image, state_json, openpi_status],
        )
        query_btn.click(
            query_openpi,
            [host, port, prompt, image, wrist_image, state_json],
            [action_json, openpi_status, raw_openpi_response],
        )
        check_btn.click(check_trajectory, [action_json, action_mode], [report, plan_table, plot, plan_state])
        execute_btn.click(execute_plan, [approve_box, dry_run_box, plan_state], [execute_status, pose_table, joint_table, arm_status_md])

    return demo


def read_state_tables(robot: RobotAdapter) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    pose = robot.read_ee_pose()
    joints = robot.read_joint_state()
    status = robot.read_arm_status()
    return pose_df(pose), joint_df(joints), status_markdown(status)


def pose_df(pose: EEPose) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "x_m", "value": pose.x},
            {"field": "y_m", "value": pose.y},
            {"field": "z_m", "value": pose.z},
            {"field": "rx_deg", "value": pose.rx},
            {"field": "ry_deg", "value": pose.ry},
            {"field": "rz_deg", "value": pose.rz},
        ]
    )


def empty_pose_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["field", "value"])


def joint_df(joints: Optional[JointState]) -> pd.DataFrame:
    if joints is None:
        return empty_joint_df()
    return pd.DataFrame([{"joint": f"j{i}", "deg": v} for i, v in enumerate(joints.values_deg, start=1)])


def empty_joint_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["joint", "deg"])


def status_markdown(status: Dict[str, Any]) -> str:
    lines = ["### Arm status"]
    if not status.get("available", False):
        return "### Arm status\nStatus API not available."
    for key in (
        "ctrl_mode",
        "arm_status",
        "mode_feed",
        "motion_status",
        "trajectory_num",
        "enable_status",
        "err_status",
        "fault",
    ):
        lines.append(f"- `{key}`: `{status.get(key)}`")
    if status.get("fault"):
        lines.append("\n**Fault reported. Do not execute a plan until this is resolved.**")
    return "\n".join(lines)


def state_json_from_robot(
    pose: EEPose,
    joints: Optional[JointState],
    gripper: Optional[GripperState] = None,
    cfg: Optional[SafetyConfig] = None,
) -> str:
    openpi_state = None
    if joints is not None:
        # OpenPI PiPER training recorded robosuite qpos: 6 joint angles in
        # radians plus two per-finger gripper qpos values. Piper SDK reports
        # joints in degrees and one total gripper opening, so convert degrees
        # to radians and split the opening equally across the two fingers.
        grip_pair = robosuite_gripper_qpos_pair(gripper, cfg)
        openpi_state = [math.radians(v) for v in joints.values_deg] + grip_pair
    state = {
        "ee_pose_m_deg": pose.as_list(),
        "joints_deg": joints.as_list() if joints else None,
        "gripper_m": None if gripper is None else gripper.opening_m,
        "gripper_effort_n_m": None if gripper is None else gripper.effort_n_m,
        "state": openpi_state,
        "observation/state": openpi_state,
    }
    return json.dumps(state, indent=2)


def robosuite_gripper_qpos_pair(
    gripper: Optional[GripperState],
    cfg: Optional[SafetyConfig] = None,
) -> list[float]:
    if gripper is None:
        return [0.0, 0.0]
    qpos_max = 0.035 if cfg is None else cfg.robosuite_gripper_qpos_max_m
    qpos = max(0.0, min(qpos_max, gripper.opening_m / 2.0))
    return [qpos, qpos]


def plan_dataframe(plan: TrajectoryPlan) -> pd.DataFrame:
    rows = []
    for step in plan.steps:
        row: Dict[str, Any] = {
            "step": step.index,
            "ok": step.ok,
            "gripper_mm": None if step.gripper_m is None else step.gripper_m * 1000.0,
            "violations": "; ".join(step.violations),
            "warnings": "; ".join(step.warnings),
        }
        if step.target_pose is not None:
            row.update(
                {
                    "x_m": step.target_pose.x,
                    "y_m": step.target_pose.y,
                    "z_m": step.target_pose.z,
                    "rx_deg": step.target_pose.rx,
                    "ry_deg": step.target_pose.ry,
                    "rz_deg": step.target_pose.rz,
                }
            )
        if step.target_joints is not None:
            for i, value in enumerate(step.target_joints.values_deg, start=1):
                row[f"j{i}_deg"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plan_report_markdown(plan: TrajectoryPlan) -> str:
    prefix = "SAFE" if plan.approved_by_safety else "REJECTED"
    lines = [f"## {prefix}", plan.summary]
    for step in plan.steps:
        if step.violations:
            lines.append(f"\n### Step {step.index} violations")
            lines.extend([f"- {v}" for v in step.violations])
        if step.warnings:
            lines.append(f"\n### Step {step.index} warnings")
            lines.extend([f"- {w}" for w in step.warnings])
    return "\n".join(lines)


def trajectory_plot(plan: TrajectoryPlan):
    pose_steps = [s for s in plan.steps if s.target_pose is not None]
    if not pose_steps:
        return blank_plot("No Cartesian targets in this plan")
    xs = [plan.initial_pose.x] + [s.target_pose.x for s in pose_steps if s.target_pose]
    ys = [plan.initial_pose.y] + [s.target_pose.y for s in pose_steps if s.target_pose]
    zs = [plan.initial_pose.z] + [s.target_pose.z for s in pose_steps if s.target_pose]
    cfg = plan.config_snapshot
    wx = tuple(cfg.get("workspace_x_m", (0.0, 1.0)))
    wy = tuple(cfg.get("workspace_y_m", (-0.5, 0.5)))
    wz = tuple(cfg.get("workspace_z_m", (0.0, 0.5)))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, marker="o")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="s", s=60)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Target trajectory in Piper/base frame")
    _draw_workspace_box(ax, wx, wy, wz)
    fig.tight_layout()
    return fig


def _draw_workspace_box(ax: Any, wx: Tuple[float, float], wy: Tuple[float, float], wz: Tuple[float, float]) -> None:
    corners = [
        (wx[0], wy[0], wz[0]),
        (wx[1], wy[0], wz[0]),
        (wx[1], wy[1], wz[0]),
        (wx[0], wy[1], wz[0]),
        (wx[0], wy[0], wz[1]),
        (wx[1], wy[0], wz[1]),
        (wx[1], wy[1], wz[1]),
        (wx[0], wy[1], wz[1]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(
            [corners[a][0], corners[b][0]],
            [corners[a][1], corners[b][1]],
            [corners[a][2], corners[b][2]],
            linewidth=0.8,
            alpha=0.7,
        )


def blank_plot(text: str = ""):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text or "No plot", ha="center", va="center")
    ax.set_axis_off()
    return fig


def execution_markdown(result: Dict[str, Any]) -> str:
    status = "OK" if result.get("ok") else "BLOCKED/ABORTED"
    lines = [f"## {status}", f"Dry run: `{result.get('dry_run')}`", f"Executed steps: `{result.get('executed_steps')}`"]
    for msg in result.get("messages", []):
        lines.append(f"- {msg}")
    return "\n".join(lines)


def _err(exc: Exception) -> str:
    return "ERROR: " + str(exc) + "\n\n```text\n" + traceback.format_exc(limit=3) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper VLA Guard UI")
    parser.add_argument("--config", default="configs/safety.example.yaml")
    parser.add_argument("--can", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Force dry run on")
    parser.add_argument("--real-hardware", action="store_true", help="Use piper_sdk instead of mock adapter")
    parser.add_argument("--robot-camera-source", default=None, help="Default robot/wrist camera source")
    parser.add_argument("--overhead-camera-source", default=None, help="Default overhead/front camera source")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.can is not None:
        cfg.can_name = args.can
    if args.dry_run:
        cfg.dry_run = True

    robot_camera_source_default = args.robot_camera_source or os.getenv("PIPER_VLA_ROBOT_CAMERA_SOURCE", "")
    overhead_camera_source_default = args.overhead_camera_source or os.getenv("PIPER_VLA_OVERHEAD_CAMERA_SOURCE", "")

    demo = create_ui(
        cfg,
        real_hardware=args.real_hardware,
        robot_camera_source_default=robot_camera_source_default,
        overhead_camera_source_default=overhead_camera_source_default,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
