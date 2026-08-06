from pathlib import Path

import numpy as np

from demo2.nero_direction_ik import DirectionIK, NeroKinematics


URDF = Path(__file__).resolve().parents[1] / "urdf/nero_description.urdf"


def angle_deg(first, second):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


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
