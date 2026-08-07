"""Direction-constrained inverse kinematics for the seven-axis Nero arm."""

import xml.etree.ElementTree as ET

import numpy as np

from demo2.arm_geometry import unit
from demo2.arm_sides import validate_side


SIDE_MOUNT_COMPONENT_SIGNS = {
    "left": np.asarray([1.0, 1.0, -1.0]),
    "right": np.asarray([1.0, 1.0, 1.0]),
}


def side_mount_components(components, side):
    """Map person-relative directions into one mirrored Nero mount."""
    side = validate_side(side)
    return unit(
        np.asarray(components, dtype=float)
        * SIDE_MOUNT_COMPONENT_SIGNS[side]
    )


def rpy_matrix(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_matrix(axis, angle):
    axis = unit(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one = 1.0 - c
    return np.asarray([
        [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
        [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
        [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
    ])


def transform(rotation=None, translation=None):
    output = np.eye(4)
    if rotation is not None:
        output[:3, :3] = rotation
    if translation is not None:
        output[:3, 3] = translation
    return output


class NeroKinematics:
    """Small URDF FK implementation for the serial joint1..joint7 chain."""

    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        self.joints = []
        self.limits = []
        for index in range(1, 8):
            name = f"joint{index}"
            joint = by_name[name]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            limit = joint.find("limit")
            self.joints.append((xyz, rpy, axis))
            self.limits.append(
                [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
            )
        self.limits = np.asarray(self.limits)

    def joint_origins(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError("expected seven joint angles")
        current = np.eye(4)
        origins = []
        for angle, (xyz, rpy, axis) in zip(q, self.joints):
            current = current @ transform(rpy_matrix(rpy), xyz)
            origins.append(current[:3, 3].copy())
            current = current @ transform(axis_angle_matrix(axis, angle))
        return np.asarray(origins)

    def arm_directions(self, q):
        origins = self.joint_origins(q)
        # joint2 is the shoulder, joint4 is the elbow, joint5 is the wrist.
        upper = unit(origins[3] - origins[1])
        forearm = unit(origins[4] - origins[3])
        return upper, forearm


class DirectionIK:
    def __init__(
        self,
        kinematics,
        neutral=None,
        direction_weight=1.0,
        continuity_weight=0.08,
        neutral_weight=0.015,
        damping=0.04,
        max_iterations=20,
        max_step_rad=0.12,
    ):
        self.kinematics = kinematics
        self.neutral = np.zeros(7) if neutral is None else np.asarray(neutral, dtype=float)
        self.previous = np.clip(
            self.neutral.copy(),
            kinematics.limits[:, 0],
            kinematics.limits[:, 1],
        )
        self.direction_weight = float(direction_weight)
        self.continuity_weight = float(continuity_weight)
        self.neutral_weight = float(neutral_weight)
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.max_step_rad = float(max_step_rad)
        neutral_upper, _ = self.kinematics.arm_directions(self.previous)
        self.robot_basis = self._make_robot_basis(neutral_upper)

    @staticmethod
    def _make_robot_basis(outward):
        outward = unit(outward)
        up_hint = np.asarray([0.0, 0.0, 1.0])
        up = up_hint - outward * float(np.dot(up_hint, outward))
        if np.linalg.norm(up) < 1e-6:
            up_hint = np.asarray([1.0, 0.0, 0.0])
            up = up_hint - outward * float(np.dot(up_hint, outward))
        up = unit(up)
        forward = unit(np.cross(outward, up))
        return np.column_stack((outward, up, forward))

    def components_to_robot(self, components):
        return unit(self.robot_basis @ unit(components))

    def residual(self, q, upper_target, forearm_target):
        upper, forearm = self.kinematics.arm_directions(q)
        return self.direction_weight * np.concatenate(
            (upper_target - upper, forearm_target - forearm)
        )

    def numerical_jacobian(self, q, epsilon=1e-5):
        upper0, forearm0 = self.kinematics.arm_directions(q)
        base = np.concatenate((upper0, forearm0))
        jacobian = np.zeros((6, 7))
        for index in range(7):
            shifted = q.copy()
            shifted[index] += epsilon
            upper1, forearm1 = self.kinematics.arm_directions(shifted)
            jacobian[:, index] = (
                np.concatenate((upper1, forearm1)) - base
            ) / epsilon
        return self.direction_weight * jacobian

    def cost(self, q, upper_target, forearm_target, seed):
        task = self.residual(q, upper_target, forearm_target)
        continuity = self.continuity_weight * float(
            np.sum((q - seed) ** 2)
        )
        neutral = self.neutral_weight * float(
            np.sum((q - self.neutral) ** 2)
        )
        return float(task @ task) + continuity + neutral

    def solve(self, upper_components, forearm_components):
        upper_target = self.components_to_robot(upper_components)
        forearm_target = self.components_to_robot(forearm_components)
        seed = self.previous.copy()
        q = seed.copy()

        for _ in range(self.max_iterations):
            error = self.residual(q, upper_target, forearm_target)
            jacobian = self.numerical_jacobian(q)
            rows = [jacobian]
            values = [error]
            if self.continuity_weight > 0.0:
                weight = np.sqrt(self.continuity_weight)
                rows.append(weight * np.eye(7))
                values.append(weight * (seed - q))
            if self.neutral_weight > 0.0:
                weight = np.sqrt(self.neutral_weight)
                rows.append(weight * np.eye(7))
                values.append(weight * (self.neutral - q))
            rows.append(self.damping * np.eye(7))
            values.append(np.zeros(7))
            delta, _, _, _ = np.linalg.lstsq(
                np.vstack(rows), np.concatenate(values), rcond=None
            )
            norm = float(np.linalg.norm(delta))
            if norm < 1e-5:
                break
            if norm > self.max_step_rad:
                delta *= self.max_step_rad / norm

            old_cost = self.cost(q, upper_target, forearm_target, seed)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = np.clip(
                    q + scale * delta,
                    self.kinematics.limits[:, 0],
                    self.kinematics.limits[:, 1],
                )
                if self.cost(candidate, upper_target, forearm_target, seed) < old_cost:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break

        self.previous = q
        upper_actual, forearm_actual = self.kinematics.arm_directions(q)
        error_deg = np.rad2deg(
            [
                np.arccos(np.clip(np.dot(upper_actual, upper_target), -1.0, 1.0)),
                np.arccos(np.clip(np.dot(forearm_actual, forearm_target), -1.0, 1.0)),
            ]
        )
        return q.copy(), error_deg
