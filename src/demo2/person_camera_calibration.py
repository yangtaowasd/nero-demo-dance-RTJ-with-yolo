"""Camera-to-person reference calibration for RGB-D arm retargeting."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from demo2.arm_geometry import torso_basis


def torso_pose(points):
    """Return torso origin and person-basis rotation in camera coordinates."""
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError("expected eight 3-D landmark slots")
    torso_indices = (0, 3, 4, 7)
    if not np.all(np.isfinite(points[list(torso_indices)])):
        raise ValueError("shoulder and hip landmarks must be finite")
    shoulder_center = 0.5 * (points[0] + points[4])
    hip_center = 0.5 * (points[3] + points[7])
    origin = 0.5 * (shoulder_center + hip_center)
    right, up, backward = torso_basis(points)
    # A camera-like person frame stays right-handed while exposing the actual
    # facing direction: X right, Y down, Z out of the chest.
    basis = np.column_stack((right, -up, -backward))
    return origin, basis


def project_to_rotation(matrix):
    """Project a near-rotation matrix onto SO(3)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation candidate must be a finite 3x3 matrix")
    left, _, right = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def average_rotations(rotations):
    """Average rotation matrices and return a proper rotation."""
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape Nx3x3")
    if len(rotations) == 0:
        raise ValueError("at least one rotation is required")
    return project_to_rotation(np.mean(rotations, axis=0))


def relative_person_pose(
    reference_origin, reference_basis, current_origin, current_basis
):
    """Return current torso translation/rotation in the reference frame."""
    reference_origin = np.asarray(reference_origin, dtype=float).reshape(3)
    reference_basis = project_to_rotation(reference_basis)
    current_origin = np.asarray(current_origin, dtype=float).reshape(3)
    current_basis = project_to_rotation(current_basis)
    translation = reference_basis.T @ (current_origin - reference_origin)
    rotation = reference_basis.T @ current_basis
    return translation, project_to_rotation(rotation)


def rotation_angle_deg(rotation):
    """Return the unsigned angle represented by a rotation matrix."""
    rotation = project_to_rotation(rotation)
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def rotation_to_quaternion(rotation):
    """Return an `(x, y, z, w)` unit quaternion."""
    rotation = project_to_rotation(rotation)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        first = (index + 1) % 3
        second = (index + 2) % 3
        scale = 2.0 * np.sqrt(
            max(
                1.0 + rotation[index, index]
                - rotation[first, first]
                - rotation[second, second],
                0.0,
            )
        )
        quaternion = np.zeros(4)
        quaternion[index] = 0.25 * scale
        if scale > 1e-9:
            quaternion[first] = (
                rotation[first, index] + rotation[index, first]
            ) / scale
            quaternion[second] = (
                rotation[second, index] + rotation[index, second]
            ) / scale
            quaternion[3] = (
                rotation[second, first] - rotation[first, second]
            ) / scale
        else:
            quaternion[3] = 1.0
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("rotation produced a zero quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


@dataclass(frozen=True)
class StablePoseUpdate:
    """Result of adding one pose to a stable calibration cluster."""

    accepted: bool
    restarted: bool
    translation_error_m: float
    rotation_error_deg: float


class StablePoseSamples:
    """Accumulate a stable pose while ignoring isolated detector outliers."""

    def __init__(
        self,
        max_translation_m,
        max_rotation_deg,
        max_consecutive_outliers=3,
    ):
        """Configure the inlier thresholds and restart policy."""
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_consecutive_outliers = max(
            int(max_consecutive_outliers), 1
        )
        self.reset()

    def reset(self):
        """Clear all accepted samples and timing state."""
        self.origins = []
        self.bases = []
        self.start_time = None
        self.last_accepted_time = None
        self.consecutive_outliers = 0

    @property
    def sample_count(self):
        """Return the number of accepted inlier samples."""
        return len(self.origins)

    def elapsed(self):
        """Return elapsed time between the first and latest inlier."""
        if self.start_time is None or self.last_accepted_time is None:
            return 0.0
        return max(self.last_accepted_time - self.start_time, 0.0)

    def center(self):
        """Return the robust center of the current sample cluster."""
        if not self.origins:
            raise ValueError("pose sample cluster is empty")
        origin = np.median(np.asarray(self.origins), axis=0)
        basis = average_rotations(self.bases)
        return origin, basis

    def add(self, origin, basis, timestamp):
        """Add an inlier, ignore one outlier, or start a new cluster."""
        origin = np.asarray(origin, dtype=float).reshape(3)
        basis = project_to_rotation(basis)
        timestamp = float(timestamp)
        if not np.all(np.isfinite(origin)) or not np.isfinite(timestamp):
            raise ValueError("pose sample must be finite")

        restarted = False
        translation_error = 0.0
        rotation_error = 0.0
        if self.origins:
            center_origin, center_basis = self.center()
            translation_error = float(
                np.linalg.norm(origin - center_origin)
            )
            rotation_error = rotation_angle_deg(center_basis.T @ basis)
            is_outlier = (
                translation_error > self.max_translation_m
                or rotation_error > self.max_rotation_deg
            )
            if is_outlier:
                self.consecutive_outliers += 1
                if (
                    self.consecutive_outliers
                    < self.max_consecutive_outliers
                ):
                    return StablePoseUpdate(
                        False,
                        False,
                        translation_error,
                        rotation_error,
                    )
                self.reset()
                restarted = True

        if not self.origins:
            self.start_time = timestamp
        self.origins.append(origin)
        self.bases.append(basis)
        self.last_accepted_time = timestamp
        self.consecutive_outliers = 0
        return StablePoseUpdate(
            True,
            restarted,
            translation_error,
            rotation_error,
        )


@dataclass(frozen=True)
class PersonCameraReference:
    """Persisted neutral person pose and arm calibration."""

    frame_id: str
    camera_id: str
    origin_camera_m: np.ndarray
    basis_camera: np.ndarray
    left_correction: np.ndarray
    right_correction: np.ndarray
    bone_lengths_m: np.ndarray

    def __post_init__(self):
        """Normalize and validate persisted reference fields."""
        camera_id = str(self.camera_id).strip()
        if not camera_id:
            raise ValueError("calibration camera identity must not be empty")
        origin = np.asarray(self.origin_camera_m, dtype=float).reshape(3)
        basis = project_to_rotation(self.basis_camera)
        left = project_to_rotation(self.left_correction)
        right = project_to_rotation(self.right_correction)
        if self.bone_lengths_m is None:
            bones = np.full(4, np.nan, dtype=float)
        else:
            bones = np.asarray(self.bone_lengths_m, dtype=float).reshape(4)
        if not np.all(np.isfinite(origin)):
            raise ValueError("reference origin must be finite")
        bones_available = np.all(np.isfinite(bones))
        if bones_available and np.any(bones <= 0.0):
            raise ValueError("reference bone lengths must be positive")
        if not bones_available and not np.all(np.isnan(bones)):
            raise ValueError(
                "reference bone lengths must be complete or absent"
            )
        object.__setattr__(self, "origin_camera_m", origin)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "basis_camera", basis)
        object.__setattr__(self, "left_correction", left)
        object.__setattr__(self, "right_correction", right)
        object.__setattr__(self, "bone_lengths_m", bones)

    def to_dict(self):
        """Return a JSON-serializable calibration record."""
        return {
            "version": 4,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "origin_camera_m": self.origin_camera_m.tolist(),
            "basis_camera": self.basis_camera.tolist(),
            "left_correction": self.left_correction.tolist(),
            "right_correction": self.right_correction.tolist(),
            "bone_lengths_m": (
                self.bone_lengths_m.tolist()
                if np.all(np.isfinite(self.bone_lengths_m))
                else None
            ),
        }

    @classmethod
    def from_dict(cls, values):
        """Validate and construct a reference from serialized values."""
        version = int(values.get("version", 0))
        if version not in (3, 4):
            raise ValueError("unsupported person calibration version")
        return cls(
            str(values["frame_id"]),
            str(values["camera_id"]),
            values["origin_camera_m"],
            values["basis_camera"],
            values["left_correction"],
            values["right_correction"],
            values.get("bone_lengths_m") if version >= 4 else None,
        )

    def save(self, path):
        """Atomically save the calibration as JSON."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        """Load and validate a JSON calibration file."""
        path = Path(path).expanduser()
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(values)
