from pathlib import Path

import numpy as np

from demo2.nero_direction_ik import (
    DirectionIK,
    NeroKinematics,
    rpy_matrix,
    side_mount_components,
)


URDF = Path(__file__).resolve().parents[1] / "urdf/nero_description.urdf"


def angle_deg(first, second):
    cosine = np.clip(np.dot(first, second), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def test_neutral_human_directions_keep_neutral_robot_pose():
    solver = DirectionIK(NeroKinematics(URDF))

    joints, errors = solver.solve([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    np.testing.assert_allclose(joints, np.zeros(7), atol=1e-6)
    assert float(np.max(errors)) < 1e-3


def test_solver_reduces_direction_error_and_respects_limits():
    kinematics = NeroKinematics(URDF)
    solver = DirectionIK(
        kinematics,
        continuity_weight=0.01,
        neutral_weight=0.001,
        max_iterations=40,
    )
    upper_components = np.asarray([0.75, 0.45, 0.48])
    forearm_components = np.asarray([0.35, 0.80, 0.48])
    upper_target = solver.components_to_robot(upper_components)
    forearm_target = solver.components_to_robot(forearm_components)
    before_upper, before_forearm = kinematics.arm_directions(np.zeros(7))
    before = angle_deg(before_upper, upper_target) + angle_deg(
        before_forearm, forearm_target
    )

    joints, errors = solver.solve(upper_components, forearm_components)

    assert float(np.sum(errors)) < before
    assert np.all(joints >= kinematics.limits[:, 0])
    assert np.all(joints <= kinematics.limits[:, 1])


def test_left_side_mount_mirrors_sagittal_motion_only():
    """Both side-mounted arms move forward in the same world direction."""
    solver = DirectionIK(NeroKinematics(URDF))
    components = np.asarray([0.75, 0.30, 0.45])
    left_local = solver.components_to_robot(
        side_mount_components(components, "left")
    )
    right_local = solver.components_to_robot(
        side_mount_components(components, "right")
    )
    left_mount = rpy_matrix([0.0, -np.pi / 2.0, -np.pi / 2.0])
    right_mount = rpy_matrix([0.0, -np.pi / 2.0, np.pi / 2.0])
    mirror_right_to_left = np.diag([1.0, -1.0, 1.0])

    np.testing.assert_allclose(
        left_mount @ left_local,
        mirror_right_to_left @ right_mount @ right_local,
        atol=1e-6,
    )


def test_repeated_warm_start_converges_after_large_direction_change():
    """An initially rejected direction becomes valid on the next frame."""
    solver = DirectionIK(NeroKinematics(URDF))
    upper = [-0.6457061, -0.4379615, -0.6255025]
    forearm = [-0.1784647, -0.9621566, 0.2059249]

    _, first_errors = solver.solve(upper, forearm)
    _, second_errors = solver.solve(upper, forearm)

    assert float(np.max(first_errors)) > 25.0
    assert float(np.max(second_errors)) < 25.0
