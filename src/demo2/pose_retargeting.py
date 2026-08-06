from dataclasses import dataclass

import numpy as np


LEFT_SHOULDER = 0
LEFT_ELBOW = 1
LEFT_WRIST = 2
LEFT_HIP = 3
RIGHT_SHOULDER = 4
RIGHT_ELBOW = 5
RIGHT_WRIST = 6
RIGHT_HIP = 7


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError("cannot normalize a near-zero body vector")
    return vector / norm


def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def vector_angle(first, second):
    first = unit(first)
    second = unit(second)
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


@dataclass(frozen=True)
class ArmFeatures:
    forward: float
    elevation: float
    elbow_flex: float


def torso_frame(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError(f"expected eight 3-D landmarks, got {points.shape}")

    shoulder_center = 0.5 * (
        points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]
    )
    hip_center = 0.5 * (points[LEFT_HIP] + points[RIGHT_HIP])
    right = unit(points[RIGHT_SHOULDER] - points[LEFT_SHOULDER])
    down = unit(hip_center - shoulder_center)
    # MediaPipe uses negative Z toward the camera. Keep the torso frame
    # consistent with that convention so "arm forward" means toward camera.
    forward = unit(-np.cross(right, down))
    up = unit(np.cross(forward, right))
    return right, up, forward


def arm_features(points, side):
    points = np.asarray(points, dtype=float)
    right, up, forward = torso_frame(points)
    if side == "left":
        shoulder, elbow, wrist = (
            points[LEFT_SHOULDER],
            points[LEFT_ELBOW],
            points[LEFT_WRIST],
        )
        outward = -right
    elif side == "right":
        shoulder, elbow, wrist = (
            points[RIGHT_SHOULDER],
            points[RIGHT_ELBOW],
            points[RIGHT_WRIST],
        )
        outward = right
    else:
        raise ValueError(f"unknown arm side: {side}")

    upper = unit(elbow - shoulder)
    forearm = unit(wrist - elbow)
    outward_component = float(np.dot(upper, outward))
    forward_component = float(np.dot(upper, forward))
    up_component = float(np.dot(upper, up))
    return ArmFeatures(
        forward=float(np.arctan2(forward_component, outward_component)),
        elevation=float(
            np.arctan2(
                up_component,
                max(np.hypot(outward_component, forward_component), 1e-6),
            )
        ),
        elbow_flex=vector_angle(upper, forearm),
    )


def retarget_arm(side, current, baseline):
    side_sign = -1.0 if side == "left" else 1.0
    joints = np.zeros(7, dtype=float)
    joints[0] = side_sign * normalize_angle(
        current.forward - baseline.forward
    )
    joints[1] = -side_sign * normalize_angle(
        current.elevation - baseline.elevation
    )
    joints[2] = 0.0
    joints[3] = max(current.elbow_flex - baseline.elbow_flex, 0.0)
    return joints
