"""Canonical anatomical left/right mappings for RGB-D arm control."""

import numpy as np


SIDES = ("left", "right")
TORSO_LANDMARK_INDICES = (0, 3, 4, 7)
ARM_LANDMARK_INDICES = {
    "left": (0, 1, 2),
    "right": (4, 5, 6),
}
REQUIRED_LANDMARK_INDICES = {
    side: tuple(dict.fromkeys(TORSO_LANDMARK_INDICES + arm_indices))
    for side, arm_indices in ARM_LANDMARK_INDICES.items()
}
DEFAULT_JOINT_STATE_TOPICS = {
    "left": "/left/joint_states",
    "right": "/right/joint_states",
}
DEFAULT_COMMAND_TOPICS = {
    "left": "/left/neroarm/command_joints",
    "right": "/right/neroarm/command_joints",
}


def validate_side(side):
    """Return a canonical side name or raise for an unknown side."""
    side = str(side).strip().lower()
    if side not in SIDES:
        raise ValueError(f"unknown arm side: {side}")
    return side


def side_landmarks_valid(
    points,
    confidence,
    depth_valid,
    side,
    min_confidence,
):
    """Return whether one arm and the torso are independently usable."""
    side = validate_side(side)
    points = np.asarray(points, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    depth_valid = np.asarray(depth_valid, dtype=bool)
    if points.shape != (8, 3):
        raise ValueError("points must have shape 8x3")
    if confidence.shape != (8,) or depth_valid.shape != (8,):
        raise ValueError("confidence and depth_valid must have shape 8")
    indices = list(REQUIRED_LANDMARK_INDICES[side])
    return bool(
        np.all(np.isfinite(points[indices]))
        and np.all(np.isfinite(confidence[indices]))
        and np.all(confidence[indices] >= float(min_confidence))
        and np.all(depth_valid[indices])
    )
