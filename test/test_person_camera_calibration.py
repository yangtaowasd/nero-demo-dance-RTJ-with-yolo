"""Tests for camera-to-person neutral reference geometry."""

import numpy as np
import pytest

from demo2.person_camera_calibration import (
    PersonCameraReference,
    StablePoseSamples,
    average_rotations,
    relative_person_pose,
    rotation_angle_deg,
    rotation_to_quaternion,
    rotation_to_rpy,
    torso_pose,
)


def front_facing_pose():
    """Return a natural standing pose in a color optical frame."""
    return np.asarray([
        [0.20, 0.00, 2.00],
        [0.25, 0.25, 2.00],
        [0.28, 0.50, 2.00],
        [0.15, 0.50, 2.00],
        [-0.20, 0.00, 2.00],
        [-0.25, 0.25, 2.00],
        [-0.28, 0.50, 2.00],
        [-0.15, 0.50, 2.00],
    ])


def z_rotation(angle):
    """Return a Z-axis rotation matrix."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_torso_pose_recovers_human_origin_and_axes():
    """The natural pose produces the documented person coordinate axes."""
    origin, basis = torso_pose(front_facing_pose())

    np.testing.assert_allclose(origin, [0.0, 0.25, 2.0])
    np.testing.assert_allclose(basis[:, 0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(basis[:, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(basis[:, 2], [0.0, 0.0, -1.0])


def test_relative_pose_recovers_motion_in_reference_axes():
    """Relative pose reports motion in the saved person frame."""
    points = front_facing_pose()
    reference_origin, reference_basis = torso_pose(points)
    rotation = z_rotation(np.deg2rad(20.0))
    translation_camera = reference_basis @ np.asarray([0.3, -0.1, 0.2])
    moved = (rotation @ (points - reference_origin).T).T
    moved += reference_origin + translation_camera
    current_origin, current_basis = torso_pose(moved)

    translation, relative_rotation = relative_person_pose(
        reference_origin,
        reference_basis,
        current_origin,
        current_basis,
    )

    np.testing.assert_allclose(translation, [0.3, -0.1, 0.2], atol=1e-7)
    assert rotation_angle_deg(relative_rotation) == pytest.approx(20.0)


def test_rotation_outputs_are_normalized_and_consistent():
    """Rotation matrix conversions preserve the requested rotation."""
    rotation = z_rotation(np.deg2rad(30.0))

    quaternion = rotation_to_quaternion(rotation)
    rpy = rotation_to_rpy(rotation)

    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    np.testing.assert_allclose(rpy, [0.0, 0.0, np.deg2rad(30.0)])


def test_rotation_average():
    """Rotation averaging returns the midpoint orientation."""
    average = average_rotations([
        z_rotation(np.deg2rad(9.0)),
        z_rotation(np.deg2rad(11.0)),
    ])

    assert rotation_angle_deg(average) == pytest.approx(10.0)


def test_reference_round_trip(tmp_path):
    """A complete reference survives JSON serialization."""
    origin, basis = torso_pose(front_facing_pose())
    reference = PersonCameraReference(
        "camera_color_optical_frame",
        origin,
        basis,
        np.eye(3),
        np.eye(3),
        [0.30, 0.30, 0.30, 0.30],
    )
    path = tmp_path / "person.json"

    reference.save(path)
    loaded = PersonCameraReference.load(path)

    assert loaded.frame_id == reference.frame_id
    np.testing.assert_allclose(loaded.origin_camera_m, origin)
    np.testing.assert_allclose(loaded.basis_camera, basis)


def test_torso_pose_does_not_require_arm_landmarks():
    """Shoulders and hips alone are sufficient for torso calibration."""
    points = front_facing_pose()
    points[[1, 2, 5, 6]] = np.nan

    origin, basis = torso_pose(points)

    np.testing.assert_allclose(origin, [0.0, 0.25, 2.0])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-7)


def test_reference_can_be_saved_before_arm_lengths_are_available(tmp_path):
    """Torso calibration persists before any safe full-arm frame arrives."""
    origin, basis = torso_pose(front_facing_pose())
    reference = PersonCameraReference(
        "camera_color_optical_frame",
        origin,
        basis,
        np.eye(3),
        np.eye(3),
        None,
    )
    path = tmp_path / "torso_only.json"

    reference.save(path)
    loaded = PersonCameraReference.load(path)

    assert np.all(np.isnan(loaded.bone_lengths_m))


def test_stable_samples_ignore_one_detector_outlier():
    """One false torso estimate does not erase valid natural-pose samples."""
    origin, basis = torso_pose(front_facing_pose())
    samples = StablePoseSamples(0.08, 12.0, 3)

    assert samples.add(origin, basis, 0.0).accepted
    rejected = samples.add(origin + [1.0, 0.0, 0.0], basis, 1.0)
    accepted = samples.add(origin + [0.01, 0.0, 0.0], basis, 3.1)

    assert not rejected.accepted
    assert accepted.accepted
    assert samples.sample_count == 2
    assert samples.elapsed() == pytest.approx(3.1)


def test_stable_samples_restart_only_after_sustained_motion():
    """Several consecutive outliers establish a deliberately moved pose."""
    origin, basis = torso_pose(front_facing_pose())
    samples = StablePoseSamples(0.08, 12.0, 3)
    moved = origin + [0.5, 0.0, 0.0]

    samples.add(origin, basis, 0.0)
    first = samples.add(moved, basis, 1.0)
    second = samples.add(moved, basis, 2.0)
    third = samples.add(moved, basis, 3.0)

    assert not first.accepted
    assert not second.accepted
    assert third.accepted and third.restarted
    assert samples.sample_count == 1
    assert samples.elapsed() == 0.0
