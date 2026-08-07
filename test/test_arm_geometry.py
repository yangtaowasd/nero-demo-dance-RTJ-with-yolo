"""Tests for the 3-D geometry retained by the RealSense controller."""

import numpy as np

from demo2.arm_geometry import arm_direction_components, torso_basis


def standing_pose():
    """Return a front-facing pose with straight horizontal arms."""
    return np.asarray([
        [0.20, 0.00, 2.00],
        [0.50, 0.00, 2.00],
        [0.80, 0.00, 2.00],
        [0.15, 0.50, 2.00],
        [-0.20, 0.00, 2.00],
        [-0.50, 0.00, 2.00],
        [-0.80, 0.00, 2.00],
        [-0.15, 0.50, 2.00],
    ])


def test_torso_basis_is_orthonormal():
    """Torso axes form a proper orthonormal coordinate basis."""
    basis = np.column_stack(torso_basis(standing_pose()))

    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1e-8)


def test_straight_arms_point_along_each_outward_axis():
    """Anatomical left and right use their matching outward direction."""
    points = standing_pose()

    for side in ("left", "right"):
        upper, forearm = arm_direction_components(points, side)
        np.testing.assert_allclose(upper, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(forearm, [1.0, 0.0, 0.0])
