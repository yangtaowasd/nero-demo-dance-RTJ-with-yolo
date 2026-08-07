"""Minimal 3-D geometry used by RealSense dual-arm control."""

import numpy as np


def unit(vector, epsilon=1e-9):
    """Return a unit vector and reject degenerate geometry."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


def torso_basis(points):
    """Return the person's right, up, and backward axes."""
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError("expected eight 3-D landmarks")
    shoulder_center = 0.5 * (points[0] + points[4])
    hip_center = 0.5 * (points[3] + points[7])
    right = unit(points[4] - points[0])
    down_hint = unit(hip_center - shoulder_center)
    backward = unit(-np.cross(right, down_hint))
    down = unit(np.cross(right, backward))
    return right, -down, backward


def arm_direction_components(points, side):
    """Express one upper arm and forearm in its torso-relative basis."""
    points = np.asarray(points, dtype=float)
    right, up, backward = torso_basis(points)
    if side == "left":
        shoulder, elbow, wrist = points[0], points[1], points[2]
        outward = -right
    elif side == "right":
        shoulder, elbow, wrist = points[4], points[5], points[6]
        outward = right
    else:
        raise ValueError(f"unknown side: {side}")

    basis = np.column_stack((outward, up, backward))
    upper = basis.T @ unit(elbow - shoulder)
    forearm = basis.T @ unit(wrist - elbow)
    return unit(upper), unit(forearm)
