from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .types import EEPose, JointState, SafetyConfig


class ForwardKinematics(Protocol):
    def pose(self, joints: JointState) -> EEPose: ...


class PiperForwardKinematics:
    """Piper SDK FK with the units normalized to metres and degrees."""

    def __init__(self, dh_is_offset: int = 1) -> None:
        try:
            from piper_sdk import C_PiperForwardKinematics  # type: ignore
        except Exception as exc:  # pragma: no cover - hardware extra
            raise RuntimeError("piper_sdk forward kinematics is required for joint-IK execution") from exc
        self._fk = C_PiperForwardKinematics(int(dh_is_offset))

    def pose(self, joints: JointState) -> EEPose:
        links = self._fk.CalFK([math.radians(value) for value in joints.values_deg])
        if not links or len(links[-1]) < 6:
            raise RuntimeError("Piper FK returned an invalid link pose")
        x_mm, y_mm, z_mm, rx, ry, rz = links[-1][:6]
        values = (x_mm, y_mm, z_mm, rx, ry, rz)
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("Piper FK returned a non-finite pose")
        return EEPose(
            float(x_mm) / 1000.0,
            float(y_mm) / 1000.0,
            float(z_mm) / 1000.0,
            float(rx),
            float(ry),
            float(rz),
        )


@dataclass(frozen=True)
class IKResult:
    joints: JointState
    achieved_pose: EEPose
    iterations: int
    position_error_m: float
    rotation_error_deg: float
    min_singular_value: float


class IKError(RuntimeError):
    pass


class CartesianIKSolver:
    """Seeded damped-least-squares IK for small VLA Cartesian deltas.

    The current measured joint state is always the seed. This keeps the solver
    on the robot's present IK branch instead of selecting a distant solution.
    """

    def __init__(self, cfg: SafetyConfig, fk: ForwardKinematics | None = None) -> None:
        self.cfg = cfg
        self.fk = fk or PiperForwardKinematics(cfg.dh_is_offset)

    def solve_delta(
        self,
        seed: JointState,
        dxyz_m: Sequence[float],
        drpy_deg: Sequence[float],
    ) -> IKResult:
        if len(dxyz_m) != 3 or len(drpy_deg) != 3:
            raise IKError("IK delta requires three translation and three rotation values")
        if not all(math.isfinite(float(v)) for v in (*dxyz_m, *drpy_deg)):
            raise IKError("IK delta contains a non-finite value")
        start_pose = self.fk.pose(seed)
        target = apply_world_pose_delta(
            start_pose,
            dxyz_m,
            np.radians(np.asarray(drpy_deg, dtype=np.float64)),
        )
        return self.solve(seed, target)

    def solve_pose_delta(
        self,
        seed: JointState,
        reference_start: EEPose,
        reference_target: EEPose,
    ) -> IKResult:
        """Map a measured/base-frame pose delta onto SDK FK before solving."""
        dxyz, rotvec = world_pose_delta(reference_start, reference_target)
        target = apply_world_pose_delta(self.fk.pose(seed), dxyz, rotvec)
        return self.solve(seed, target)

    def solve(self, seed: JointState, target: EEPose) -> IKResult:
        q = np.radians(np.asarray(seed.values_deg, dtype=np.float64))
        lower = np.radians(
            np.asarray([self.cfg.joint_limits_deg[f"j{i}"][0] for i in range(1, 7)], dtype=np.float64)
        )
        upper = np.radians(
            np.asarray([self.cfg.joint_limits_deg[f"j{i}"][1] for i in range(1, 7)], dtype=np.float64)
        )
        if np.any(q < lower) or np.any(q > upper):
            raise IKError("IK seed is outside configured joint limits")

        damping = float(self.cfg.ik_damping)
        epsilon = float(self.cfg.ik_jacobian_delta_rad)
        max_update = math.radians(float(self.cfg.ik_max_update_deg))
        min_sv = 0.0
        for iteration in range(1, int(self.cfg.ik_max_iterations) + 1):
            pose = self.fk.pose(_joint_state_rad(q))
            error = _pose_error(pose, target)
            position_error = float(np.linalg.norm(error[:3]))
            rotation_error = math.degrees(float(np.linalg.norm(error[3:])))
            if (
                position_error <= self.cfg.ik_position_tolerance_m
                and rotation_error <= self.cfg.ik_rotation_tolerance_deg
            ):
                return IKResult(
                    joints=_joint_state_rad(q),
                    achieved_pose=pose,
                    iterations=iteration - 1,
                    position_error_m=position_error,
                    rotation_error_deg=rotation_error,
                    min_singular_value=min_sv,
                )

            jacobian = np.empty((6, 6), dtype=np.float64)
            for column in range(6):
                perturbed = q.copy()
                perturbed[column] = min(upper[column], q[column] + epsilon)
                actual_delta = perturbed[column] - q[column]
                if actual_delta < epsilon * 0.5:
                    perturbed[column] = max(lower[column], q[column] - epsilon)
                    actual_delta = perturbed[column] - q[column]
                if abs(actual_delta) < 1e-12:
                    raise IKError(f"IK cannot perturb joint {column + 1} inside its limits")
                perturbed_pose = self.fk.pose(_joint_state_rad(perturbed))
                jacobian[:, column] = _pose_delta(pose, perturbed_pose) / actual_delta

            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            min_sv = float(singular_values[-1])
            system = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
            try:
                dq = jacobian.T @ np.linalg.solve(system, error)
            except np.linalg.LinAlgError as exc:
                raise IKError("IK Jacobian solve failed") from exc
            if not np.all(np.isfinite(dq)):
                raise IKError("IK produced a non-finite joint update")
            largest = float(np.max(np.abs(dq)))
            if largest > max_update:
                dq *= max_update / largest
            q = np.clip(q + dq, lower, upper)

        pose = self.fk.pose(_joint_state_rad(q))
        error = _pose_error(pose, target)
        raise IKError(
            "IK did not converge: "
            f"position_error={np.linalg.norm(error[:3]):.6f}m, "
            f"rotation_error={math.degrees(np.linalg.norm(error[3:])):.3f}deg, "
            f"iterations={self.cfg.ik_max_iterations}"
        )


def _joint_state_rad(values_rad: np.ndarray) -> JointState:
    return JointState(tuple(float(value) for value in np.degrees(values_rad)))  # type: ignore[arg-type]


def _pose_error(current: EEPose, target: EEPose) -> np.ndarray:
    translation = np.asarray(target.xyz(), dtype=np.float64) - np.asarray(current.xyz(), dtype=np.float64)
    rotation = _rotation_vector(
        np.asarray(_rpy_rotation_matrix(*target.rpy()), dtype=np.float64)
        @ np.asarray(_rpy_rotation_matrix(*current.rpy()), dtype=np.float64).T
    )
    return np.concatenate((translation, rotation))


def _pose_delta(start: EEPose, end: EEPose) -> np.ndarray:
    translation = np.asarray(end.xyz(), dtype=np.float64) - np.asarray(start.xyz(), dtype=np.float64)
    rotation = _rotation_vector(
        np.asarray(_rpy_rotation_matrix(*end.rpy()), dtype=np.float64)
        @ np.asarray(_rpy_rotation_matrix(*start.rpy()), dtype=np.float64).T
    )
    return np.concatenate((translation, rotation))


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.asarray(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * vee
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        # VLA steps are constrained to small rotations, so reaching pi here is
        # an invalid branch jump rather than an expected command.
        raise IKError("IK encountered a 180-degree orientation branch jump")
    return angle / (2.0 * sine) * vee


def world_pose_delta(start: EEPose, target: EEPose) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(target.xyz(), dtype=np.float64) - np.asarray(start.xyz(), dtype=np.float64)
    rotation = (
        np.asarray(_rpy_rotation_matrix(*target.rpy()), dtype=np.float64)
        @ np.asarray(_rpy_rotation_matrix(*start.rpy()), dtype=np.float64).T
    )
    return translation, _rotation_vector(rotation)


def apply_world_pose_delta(
    start: EEPose,
    dxyz_m: Sequence[float],
    rotvec_rad: Sequence[float],
) -> EEPose:
    """Apply base/world translation and axis-angle rotation like robosuite OSC_POSE."""
    rotation_delta = _rotvec_matrix(rotvec_rad)
    rotation_start = np.asarray(_rpy_rotation_matrix(*start.rpy()), dtype=np.float64)
    rotation_target = rotation_delta @ rotation_start
    rx, ry, rz = _matrix_to_rpy_deg(rotation_target)
    xyz = np.asarray(start.xyz(), dtype=np.float64) + np.asarray(dxyz_m, dtype=np.float64)
    return EEPose(float(xyz[0]), float(xyz[1]), float(xyz[2]), rx, ry, rz)


def _rotvec_matrix(rotvec_rad: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotvec_rad, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    sine_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))  # type: ignore[return-value]


def _rpy_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[tuple[float, ...], ...]:
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
