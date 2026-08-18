"""Tests for the C++ detector to Python controller topic contract."""

import numpy as np
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32, PointCloud
from std_msgs.msg import Header

from demo2.arm_sides import (
    ARM_LANDMARK_INDICES,
    DEFAULT_COMMAND_TOPICS,
    DEFAULT_JOINT_STATE_TOPICS,
    side_landmarks_valid,
)
from demo2.depth_arm_controller import DepthArmController


def point_cloud(points, confidence):
    """Build the standard eight-landmark PointCloud contract."""
    points = np.asarray(points, dtype=float)
    message = PointCloud()
    message.header = Header(frame_id="camera_color_optical_frame")
    message.points = [
        Point32(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        for point in points
    ]
    message.channels = [
        ChannelFloat32(
            name="confidence",
            values=[float(value) for value in confidence],
        ),
        ChannelFloat32(
            name="landmark_id",
            values=[float(index) for index in range(8)],
        ),
        ChannelFloat32(
            name="depth_valid",
            values=[
                float(np.all(np.isfinite(point))) for point in points
            ],
        ),
    ]
    return message


def test_controller_decodes_cpp_pose_message():
    """The C++ landmark state is accepted by the controller."""
    points = np.arange(24, dtype=float).reshape(8, 3) * 0.1
    controller = object.__new__(DepthArmController)

    decoded, confidence, depth_valid, error = controller.decode_landmark_state(
        point_cloud(points, np.full(8, 0.8))
    )

    assert error is None
    np.testing.assert_allclose(decoded, points)
    np.testing.assert_allclose(confidence, 0.8)
    assert np.all(depth_valid)


def test_controller_rejects_malformed_confidence_channel():
    """The detector/controller confidence contract stays explicit."""
    points = np.arange(24, dtype=float).reshape(8, 3) * 0.1
    message = point_cloud(points, np.full(8, 0.8))
    message.channels[0].values.pop()
    controller = object.__new__(DepthArmController)

    decoded, confidence, depth_valid, error = (
        controller.decode_landmark_state(message)
    )

    assert decoded is None
    assert confidence is None
    assert depth_valid is None
    assert error == "invalid landmark confidence channel"


def test_torso_calibration_does_not_require_arm_depth():
    """Shoulders and hips remain usable when arm depth is unavailable."""
    points = np.ones((8, 3), dtype=float)
    points[[1, 2, 5, 6]] = np.nan
    confidence = np.asarray([0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9])
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.45
    controller.torso_hold_sec = 0.25
    controller.reference_origin = None
    controller.cached_torso_points = None
    controller.cached_torso_confidence = None
    controller.cached_torso_time = None

    decoded, decoded_confidence, depth_valid, error = (
        controller.decode_landmark_state(
            point_cloud(points, confidence)
        )
    )
    prepared, _, _, cached, torso_error = controller.prepare_torso_state(
        decoded, decoded_confidence, depth_valid, now=1.0
    )

    assert error is None
    assert torso_error is None
    assert not cached
    np.testing.assert_allclose(prepared[[0, 3, 4, 7]], 1.0)


def test_short_torso_dropout_uses_cached_calibrated_torso():
    """One torso depth hole cannot immediately stop both arms."""
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.45
    controller.torso_hold_sec = 0.25
    controller.reference_origin = np.zeros(3)
    controller.cached_torso_points = None
    controller.cached_torso_confidence = None
    controller.cached_torso_time = None
    points = np.ones((8, 3), dtype=float)
    confidence = np.full(8, 0.9)
    depth_valid = np.ones(8, dtype=bool)
    controller.prepare_torso_state(
        points, confidence, depth_valid, now=1.0
    )

    dropped = points.copy()
    dropped[3] = np.nan
    depth_valid[3] = False
    prepared, _, restored_depth, cached, error = (
        controller.prepare_torso_state(
            dropped, confidence, depth_valid, now=1.1
        )
    )

    assert error is None
    assert cached
    assert restored_depth[3]
    np.testing.assert_allclose(prepared[3], points[3])


def test_expired_torso_cache_is_rejected():
    """A stale torso is never used to keep hardware control alive."""
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.45
    controller.torso_hold_sec = 0.25
    controller.reference_origin = np.zeros(3)
    controller.cached_torso_points = None
    controller.cached_torso_confidence = None
    controller.cached_torso_time = None
    points = np.ones((8, 3), dtype=float)
    confidence = np.full(8, 0.9)
    depth_valid = np.ones(8, dtype=bool)
    controller.prepare_torso_state(
        points, confidence, depth_valid, now=1.0
    )
    points[0] = np.nan
    depth_valid[0] = False

    prepared, _, _, cached, error = controller.prepare_torso_state(
        points, confidence, depth_valid, now=1.3
    )

    assert prepared is None
    assert not cached
    assert error == "missing shoulder/hip depth"


def test_low_confidence_is_reported_before_missing_depth():
    """A rejected keypoint is not misdiagnosed as a depth-camera hole."""
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.45
    controller.torso_hold_sec = 0.25
    controller.reference_origin = None
    controller.cached_torso_points = None
    controller.cached_torso_confidence = None
    controller.cached_torso_time = None
    points = np.ones((8, 3), dtype=float)
    points[3] = np.nan
    confidence = np.full(8, 0.9)
    confidence[3] = 0.02
    depth_valid = np.ones(8, dtype=bool)
    depth_valid[3] = False

    prepared, _, _, cached, error = controller.prepare_torso_state(
        points, confidence, depth_valid, now=1.0
    )

    assert prepared is None
    assert not cached
    assert error == "low-confidence shoulder/hip landmark"


def test_anatomical_sides_use_matching_topics():
    """Human left/right landmarks map to the same ROS namespace."""
    assert ARM_LANDMARK_INDICES == {
        "left": (0, 1, 2),
        "right": (4, 5, 6),
    }
    assert DEFAULT_JOINT_STATE_TOPICS == {
        "left": "/demo2_display/left/joint_states",
        "right": "/demo2_display/right/joint_states",
    }
    assert DEFAULT_COMMAND_TOPICS == {
        "left": "/left/neroarm/command_joints",
        "right": "/right/neroarm/command_joints",
    }


def test_opposite_wrist_dropout_does_not_stop_left_arm():
    """A right wrist dropout cannot invalidate the anatomical left arm."""
    points = np.ones((8, 3), dtype=float)
    points[6] = np.nan
    confidence = np.full(8, 0.9)
    depth_valid = np.all(np.isfinite(points), axis=1)

    assert side_landmarks_valid(
        points, confidence, depth_valid, "left", 0.45
    )
    assert not side_landmarks_valid(
        points, confidence, depth_valid, "right", 0.45
    )
