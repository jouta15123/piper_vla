import unittest

from piper_vla_guard.actions import parse_action_json
from piper_vla_guard.config import config_from_dict
from piper_vla_guard.executor import PlanExecutor
from piper_vla_guard.piper_adapter import MockPiperAdapter
from piper_vla_guard.safety import SafetyChecker
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
        self.assertAlmostEqual(plan.steps[0].gripper_m, cfg.gripper_open_m)

    def test_robosuite_gripper_close_maps_to_closed_width(self):
        actions = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        plan = SafetyChecker(self.cfg).build_plan(self.pose, actions, "robosuite_osc_pose", self.joints)

        self.assertAlmostEqual(plan.steps[0].gripper_m, self.cfg.gripper_closed_m)

    def test_robot_gripper_opening_is_split_into_robosuite_qpos_pair(self):
        pair = robosuite_gripper_qpos_pair(GripperState(opening_m=0.070), self.cfg)

        self.assertEqual(pair, [0.035, 0.035])

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


if __name__ == "__main__":
    unittest.main()
