"""Geometry primitives for heterogeneous pinhole/fisheye stereo arm tracking.

The stereo rig convention is::

    X_upper = R_upper_from_lower @ X_lower + t_upper_from_lower

All triangulated points are returned in the lower-camera coordinate frame.
This module has no ROS or MediaPipe dependency and can be tested offline.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


LANDMARK_NAMES = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
)


def unit(vector, epsilon=1e-9):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


@dataclass(frozen=True)
class CameraModel:
    model: str
    camera_matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self):
        model = self.model.strip().lower()
        if model not in ("pinhole", "fisheye"):
            raise ValueError(f"unsupported camera model: {self.model}")
        camera_matrix = np.asarray(self.camera_matrix, dtype=float)
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if model == "fisheye" and distortion.size != 4:
            raise ValueError("fisheye distortion must contain four coefficients")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "distortion", distortion)

    def pixel_to_ray(self, pixel):
        pixel = np.asarray(pixel, dtype=float).reshape(1, 1, 2)
        if self.model == "fisheye":
            normalized = cv2.fisheye.undistortPoints(
                pixel,
                self.camera_matrix,
                self.distortion.reshape(4, 1),
            )
        else:
            normalized = cv2.undistortPoints(
                pixel,
                self.camera_matrix,
                self.distortion,
            )
        x, y = normalized.reshape(2)
        return unit([x, y, 1.0])

    def project(self, point):
        point = np.asarray(point, dtype=float).reshape(1, 1, 3)
        if float(point[0, 0, 2]) <= 0.0:
            return None
        if self.model == "fisheye":
            pixels, _ = cv2.fisheye.projectPoints(
                point,
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.distortion.reshape(4, 1),
            )
        else:
            pixels, _ = cv2.projectPoints(
                point,
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.distortion,
            )
        return pixels.reshape(2)


@dataclass(frozen=True)
class TriangulationResult:
    point: np.ndarray
    ray_gap_m: float
    ray_angle_deg: float
    reprojection_error_px: float


class StereoRig:
    def __init__(
        self,
        lower,
        upper,
        rotation_upper_from_lower,
        translation_upper_from_lower,
    ):
        self.lower = lower
        self.upper = upper
        self.rotation = np.asarray(rotation_upper_from_lower, dtype=float)
        self.translation = np.asarray(
            translation_upper_from_lower, dtype=float
        ).reshape(3)
        if self.rotation.shape != (3, 3):
            raise ValueError("stereo rotation must have shape (3, 3)")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-4):
            raise ValueError("stereo rotation is not orthonormal")

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if bool(config.get("calibration_is_example", False)):
            raise ValueError(
                "the selected stereo calibration is a template; run "
                "calibrate_vertical_stereo.py and use its output"
            )

        def camera(section):
            values = config[section]
            return CameraModel(
                values["model"],
                values["camera_matrix"],
                values.get("distortion", []),
            )

        extrinsics = config["upper_from_lower"]
        return cls(
            camera("lower_camera"),
            camera("upper_camera"),
            extrinsics["rotation"],
            extrinsics["translation_m"],
        )

    def triangulate(self, lower_pixel, upper_pixel):
        lower_origin = np.zeros(3)
        lower_direction = self.lower.pixel_to_ray(lower_pixel)

        # Transform the upper-camera ray into the lower-camera frame.
        upper_origin = -self.rotation.T @ self.translation
        upper_direction = unit(
            self.rotation.T @ self.upper.pixel_to_ray(upper_pixel)
        )

        matrix = np.column_stack((lower_direction, -upper_direction))
        scales, _, _, _ = np.linalg.lstsq(
            matrix, upper_origin - lower_origin, rcond=None
        )
        lower_closest = lower_origin + scales[0] * lower_direction
        upper_closest = upper_origin + scales[1] * upper_direction
        point = 0.5 * (lower_closest + upper_closest)
        gap = float(np.linalg.norm(lower_closest - upper_closest))
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        abs(float(np.dot(lower_direction, upper_direction))),
                        0.0,
                        1.0,
                    )
                )
            )
        )

        lower_projection = self.lower.project(point)
        upper_point = self.rotation @ point + self.translation
        upper_projection = self.upper.project(upper_point)
        invalid_projection = any((
            scales[0] <= 0.0,
            scales[1] <= 0.0,
            lower_projection is None,
            upper_projection is None,
        ))
        if invalid_projection:
            reprojection = float("inf")
        else:
            lower_error = float(
                np.linalg.norm(lower_projection - lower_pixel)
            )
            upper_error = float(
                np.linalg.norm(upper_projection - upper_pixel)
            )
            reprojection = 0.5 * (lower_error + upper_error)
        return TriangulationResult(point, gap, angle, reprojection)

    def triangulate_landmarks(
        self,
        lower_pixels,
        upper_pixels,
        max_ray_gap_m=0.08,
        min_ray_angle_deg=1.5,
        max_reprojection_error_px=8.0,
    ):
        lower_pixels = np.asarray(lower_pixels, dtype=float)
        upper_pixels = np.asarray(upper_pixels, dtype=float)
        if lower_pixels.shape != (8, 2) or upper_pixels.shape != (8, 2):
            raise ValueError("expected eight 2-D landmarks from each camera")

        points = []
        qualities = []
        for lower_pixel, upper_pixel in zip(lower_pixels, upper_pixels):
            result = self.triangulate(lower_pixel, upper_pixel)
            invalid = any((
                result.ray_gap_m > max_ray_gap_m,
                result.ray_angle_deg < min_ray_angle_deg,
                result.reprojection_error_px > max_reprojection_error_px,
                not np.all(np.isfinite(result.point)),
            ))
            if invalid:
                return None, None
            points.append(result.point)
            qualities.append(
                [
                    result.ray_gap_m,
                    result.ray_angle_deg,
                    result.reprojection_error_px,
                ]
            )
        return np.asarray(points), np.asarray(qualities)


def torso_basis(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError("expected eight 3-D landmarks")
    shoulder_center = 0.5 * (points[0] + points[4])
    hip_center = 0.5 * (points[3] + points[7])
    right = unit(points[4] - points[0])
    down_hint = unit(hip_center - shoulder_center)
    forward = unit(-np.cross(right, down_hint))
    down = unit(np.cross(right, forward))
    up = -down
    return right, up, forward


def arm_direction_components(points, side):
    points = np.asarray(points, dtype=float)
    right, up, forward = torso_basis(points)
    if side == "left":
        shoulder, elbow, wrist = points[0], points[1], points[2]
        outward = -right
    elif side == "right":
        shoulder, elbow, wrist = points[4], points[5], points[6]
        outward = right
    else:
        raise ValueError(f"unknown side: {side}")

    basis = np.column_stack((outward, up, forward))
    upper = basis.T @ unit(elbow - shoulder)
    forearm = basis.T @ unit(wrist - elbow)
    return unit(upper), unit(forearm)


def align_vector_rotation(source, target):
    """Return a proper rotation that maps one unit vector onto another."""
    source = unit(source)
    target = unit(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        helper = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(source, helper))) > 0.9:
            helper = np.asarray([0.0, 1.0, 0.0])
        axis = unit(np.cross(source, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
