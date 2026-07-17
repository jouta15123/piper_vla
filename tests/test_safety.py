import math
import unittest
from types import SimpleNamespace

import numpy as np

from piper_vla_guard.actions import parse_action_json
from piper_vla_guard.config import config_from_dict
from piper_vla_guard.executor import PlanExecutor
from piper_vla_guard.kinematics import CartesianIKSolver
from piper_vla_guard.piper_adapter import MockPiperAdapter
from piper_vla_guard.real_loop import (
    _execute_live_rebased_chunk,
    apply_action_xyz_signs,
    parse_args,
    robot_state_vector,
    select_action_chunk,
)
from piper_vla_guard.safety import SafetyChecker, workspace_floor_error
from piper_vla_guard.types import EEPose, GripperState, JointState, SafetyConfig, SafetyPlane
from piper_vla_guard.ui_app import robosuite_gripper_qpos_pair


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SafetyConfig()
        self.pose = EEPose(0.30, 0.0, 0.20, 0.0, 0.0, 0.0)
        self.joints = JointState((0.0, 60.0, -80.0, 0.0, 20.0, 0.0))

    def test_safe_delta_plan(self):
        actions = [[0.001, 0.0, 0.0, 0.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        self.assertTrue(plan.approved_by_safety)
        self.assertAlmostEqual(plan.steps[0].target_pose.x, 0.301)

    def test_omitted_gripper_nan_is_allowed(self):
        rows = parse_action_json('{"actions": [[0, 0, 0.002]]}', "delta_base_m_deg")
        plan = SafetyChecker(self.cfg).build_plan(self.pose, rows, "delta_base_m_deg", self.joints)

        self.assertTrue(plan.approved_by_safety)
        self.assertIsNone(plan.steps[0].gripper_m)
        self.assertAlmostEqual(plan.steps[0].target_pose.z, 0.202)

    def test_robosuite_osc_pose_uses_training_controller_scale(self):
        actions = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
        cfg = SafetyConfig(reject_on_clip=False)
        cfg.max_step_xyz_m = (0.05, 0.05, 0.05)
        plan = SafetyChecker(cfg).build_plan(self.pose, actions, "robosuite_osc_pose", self.joints)

        self.assertTrue(plan.approved_by_safety)
        self.assertAlmostEqual(plan.steps[0].scaled_action[0], 0.05)
        self.assertAlmostEqual(plan.steps[0].target_pose.x, 0.35)
        self.assertAlmostEqual(
            plan.steps[0].gripper_m,
            min(
                cfg.gripper_open_m,
                cfg.robosuite_gripper_min_width_m + 2 * cfg.robosuite_gripper_qpos_max_m,
            ),
        )

    def test_robosuite_rotation_is_world_axis_angle_not_euler_addition(self):
        pose = EEPose(0.30, 0.0, 0.20, 0.0, 0.0, 90.0)
        cfg = SafetyConfig(reject_on_clip=False)
        cfg.max_step_rpy_deg = (3.0, 3.0, 3.0)

        plan = SafetyChecker(cfg).build_plan(
            pose,
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]],
            "robosuite_osc_pose",
            self.joints,
        )

        target = plan.steps[0].target_pose
        self.assertAlmostEqual(target.rx, 0.0, places=5)
        self.assertAlmostEqual(target.ry, -math.degrees(0.05), places=5)
        self.assertAlmostEqual(target.rz, 90.0, places=5)

    def test_robosuite_gripper_close_maps_to_piper_grasp_width(self):
        actions = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "robosuite_osc_pose", self.joints)

        self.assertAlmostEqual(plan.steps[0].gripper_m, self.cfg.robosuite_gripper_min_width_m)

    def test_robosuite_piper_partial_close_maps_to_object_width(self):
        actions = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "robosuite_osc_pose", self.joints)

        training_open = (
            self.cfg.robosuite_gripper_min_width_m + 2 * self.cfg.robosuite_gripper_qpos_max_m
        )
        expected = 0.1 * training_open + 0.9 * self.cfg.robosuite_gripper_min_width_m
        self.assertAlmostEqual(plan.steps[0].gripper_m, expected)

    def test_robot_gripper_opening_is_split_into_robosuite_qpos_pair(self):
        pair = robosuite_gripper_qpos_pair(GripperState(opening_m=0.070), self.cfg)

        self.assertEqual(pair, [0.025, 0.025])

    def test_workspace_reject(self):
        actions = [[0.2, 0.0, 0.0, 0.0]]
        cfg = SafetyConfig(reject_on_clip=False)
        cfg.max_step_xyz_m = (0.2, 0.2, 0.2)
        plan = SafetyChecker(cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        self.assertFalse(plan.approved_by_safety)
        self.assertTrue(any("x out" in v for v in plan.steps[0].violations))

    def test_safety_plane_rejects_unsafe_side(self):
        actions = [[0.0, 0.0, -0.02, 0.0]]
        cfg = SafetyConfig(reject_on_clip=False)
        cfg.max_step_xyz_m = (0.2, 0.2, 0.2)
        cfg.safety_planes = (
            SafetyPlane(
                name="table_clearance",
                normal=(0.0, 0.0, 1.0),
                point=(0.0, 0.0, 0.19),
            ),
        )

        plan = SafetyChecker(cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)

        self.assertFalse(plan.approved_by_safety)
        self.assertTrue(any("safety plane table_clearance violated" in v for v in plan.steps[0].violations))

    def test_config_loads_safety_planes(self):
        cfg = config_from_dict(
            {
                "safety_planes": [
                    {
                        "name": "front_fixture",
                        "normal": [0.0, -1.0, 0.0],
                        "point": [0.0, 0.12, 0.0],
                        "margin_m": 0.01,
                    }
                ]
            }
        )

        self.assertEqual(len(cfg.safety_planes), 1)
        self.assertEqual(cfg.safety_planes[0].name, "front_fixture")
        self.assertEqual(cfg.safety_planes[0].margin_m, 0.01)

    def test_four_corner_floor_rejects_outside_and_below(self):
        cfg = SafetyConfig()
        cfg.workspace_floor_margin_m = 0.01
        cfg.workspace_floor_corners_m = (
            (0.20, -0.10, 0.10),
            (0.40, -0.10, 0.11),
            (0.40, 0.10, 0.11),
            (0.20, 0.10, 0.10),
        )

        self.assertIsNone(workspace_floor_error((0.30, 0.0, 0.13), cfg))
        self.assertIn("outside four-corner workspace", workspace_floor_error((0.195, 0.0, 0.13), cfg))
        self.assertIn("below four-corner floor", workspace_floor_error((0.30, 0.0, 0.11), cfg))

    def test_cartesian_plan_attaches_seeded_ik_joint_target(self):
        class LinearFK:
            def pose(self, joints):
                values = [math.radians(v) for v in joints.values_deg]
                return EEPose(values[0], values[1], values[2], *joints.values_deg[3:])

        cfg = SafetyConfig()
        cfg.joint_limits_deg = {f"j{i}": (-180.0, 180.0) for i in range(1, 7)}
        cfg.max_joint_step_deg = (5.0,) * 6
        cfg.ik_position_tolerance_m = 1e-5
        seed = JointState((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        solver = CartesianIKSolver(cfg, fk=LinearFK())

        plan = SafetyChecker(cfg, ik_solver=solver).build_plan(
            EEPose(0.30, 0.0, 0.20, 0.0, 0.0, 0.0),
            [[0.001, 0.0, 0.0, float("nan")]],
            "delta_base_m_deg",
            seed,
        )

        self.assertTrue(plan.approved_by_safety)
        self.assertIsNotNone(plan.steps[0].target_joints)
        self.assertAlmostEqual(plan.steps[0].target_joints.values_deg[0], math.degrees(0.001), places=3)

    def test_clip_reject_default(self):
        actions = [[0.1, 0.0, 0.0, 0.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        self.assertFalse(plan.approved_by_safety)
        self.assertTrue(any("clipped" in v for v in plan.steps[0].violations))

    def test_joint_limit_reject(self):
        actions = [[0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        cfg = SafetyConfig(reject_on_clip=False)
        cfg.max_joint_step_deg = (300.0, 300.0, 300.0, 300.0, 300.0, 300.0)
        plan = SafetyChecker(cfg).build_plan(self.pose, actions, "joint_delta_deg", self.joints)
        self.assertFalse(plan.approved_by_safety)
        self.assertTrue(any("j2 out" in v for v in plan.steps[0].violations))

    def test_parser_dict(self):
        text = '{"actions": [{"dpos": [0.001, 0, 0], "gripper": 0.5}]}'
        rows = parse_action_json(text, "delta_base_m_deg")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 7)
        self.assertAlmostEqual(rows[0][-1], 0.5)

    def test_rejects_non_finite_action_value(self):
        rows = parse_action_json('{"actions": [[NaN, 0, 0, 0]]}', "delta_base_m_deg")
        with self.assertRaisesRegex(ValueError, "finite"):
            SafetyChecker(self.cfg).build_plan(self.pose, rows, "delta_base_m_deg", self.joints)

    def test_executor_blocks_if_pose_drifted_after_check(self):
        actions = [[0.001, 0.0, 0.0, 0.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        robot = MockPiperAdapter()
        robot.connect()
        robot.pose = EEPose(0.31, 0.0, 0.20, 0.0, 0.0, 0.0)

        result = PlanExecutor(robot, self.cfg).execute(plan, human_approved=True, dry_run=False)

        self.assertFalse(result["ok"])
        self.assertIn("Preflight pose check failed", result["messages"][0])

    def test_executor_blocks_if_status_unavailable(self):
        actions = [[0.001, 0.0, 0.0, 0.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        robot = MockPiperAdapter()
        robot.connect()
        robot.status["available"] = False

        result = PlanExecutor(robot, self.cfg).execute(plan, human_approved=True, dry_run=False)

        self.assertFalse(result["ok"])
        self.assertIn("status API is unavailable", result["messages"][0])

    def test_executor_runs_with_joint_preflight_when_current_joints_match(self):
        actions = [[0.001, 0.0, 0.0, 0.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "delta_base_m_deg", self.joints)
        robot = MockPiperAdapter()
        robot.connect()
        robot.pose = self.pose
        robot.joints = self.joints

        result = PlanExecutor(robot, self.cfg).execute(plan, human_approved=True, dry_run=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["executed_steps"], 1)

    def test_joint_target_is_interpolated_at_configured_step_size(self):
        class RecordingRobot(MockPiperAdapter):
            def __init__(self):
                super().__init__()
                self.commands = []

            def command_joints(self, joints, speed_pct=10):
                self.commands.append(joints)
                super().command_joints(joints, speed_pct)

        cfg = SafetyConfig()
        cfg.ik_path_sample_step_deg = 0.25
        robot = RecordingRobot()
        robot.connect()
        plan = SafetyChecker(cfg).build_plan(
            robot.read_ee_pose(),
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]],
            "joint_delta_deg",
            robot.read_joint_state(),
        )

        result = PlanExecutor(robot, cfg).execute(plan, human_approved=True, dry_run=False)

        self.assertTrue(result["ok"])
        self.assertEqual(len(robot.commands), 4)
        self.assertAlmostEqual(robot.commands[-1].values_deg[0], 1.0)

    def test_real_loop_state_matches_openpi_piper_shape(self):
        robot = MockPiperAdapter()
        robot.connect()

        state = robot_state_vector(robot, self.cfg)

        self.assertEqual(len(state), 8)
        self.assertAlmostEqual(state[1], math.radians(60.0))
        # Sim fingertip width includes a 20 mm mechanical offset:
        # q_each = (70 mm physical width - 20 mm offset) / 2.
        self.assertEqual(state[-2:], [0.025, 0.025])

    def test_real_loop_selects_safe_chunk_prefix(self):
        response = {"actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]] * 10}

        chunk = select_action_chunk(response, chunk_size=5, max_horizon=3)

        self.assertEqual(chunk.shape, (3, 7))

    def test_real_loop_default_chunk_matches_sim_eval(self):
        self.assertEqual(parse_args([]).chunk_size, 5)

    def test_live_chunk_rebases_each_delta_on_latest_feedback(self):
        cfg = SafetyConfig()
        cfg.require_manual_approval = False
        robot = MockPiperAdapter()
        robot.connect()

        class RecordingChecker:
            def __init__(self):
                self.starts = []

            def build_plan(self, current_pose, current_joints, actions, action_mode):
                self.starts.append(current_pose)
                return SafetyChecker(cfg).build_plan(
                    current_pose, actions, action_mode, current_joints
                )

        class InstantExecutor:
            def execute(self, plan, human_approved, dry_run):
                robot.pose = plan.steps[0].target_pose
                return {
                    "ok": True,
                    "executed_steps": 1,
                    "max_tracking_error_m": 0.0,
                    "messages": [],
                    "stop_kind": None,
                }

        checker = RecordingChecker()
        result = _execute_live_rebased_chunk(
            action_chunk=np.asarray(
                [
                    [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
                    [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
                ]
            ),
            action_mode="delta_base_m_deg",
            args=SimpleNamespace(max_loop_translation_m=0.15, no_gripper=False),
            cfg=cfg,
            robot=robot,
            checker=checker,  # type: ignore[arg-type]
            executor=InstantExecutor(),  # type: ignore[arg-type]
            loop_initial_pose=robot.pose,
            human_approved=True,
        )

        self.assertTrue(result["ok"])
        self.assertAlmostEqual(checker.starts[0].x, 0.300)
        self.assertAlmostEqual(checker.starts[1].x, 0.301)

    def test_real_loop_can_flip_action_axes_for_calibration(self):
        response = {"actions": [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]]}
        chunk = select_action_chunk(response, chunk_size=1, max_horizon=1)

        transformed = apply_action_xyz_signs(chunk, "-1,1,-1")

        self.assertAlmostEqual(transformed[0, 0], -0.1)
        self.assertAlmostEqual(transformed[0, 1], -0.2)
        self.assertAlmostEqual(transformed[0, 2], -0.3)
        self.assertAlmostEqual(transformed[0, 6], -1.0)


if __name__ == "__main__":
    unittest.main()
