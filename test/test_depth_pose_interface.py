import numpy as np
from std_msgs.msg import Header

from demo2.depth_arm_controller import DepthArmController
from demo2.depth_pose_detector import DepthPoseDetector


def header():
    return Header(frame_id="camera_color_optical_frame")


def test_pose_message_uses_eight_points_and_confidence_channel():
    points = np.arange(24, dtype=float).reshape(8, 3) * 0.1
    confidence = np.linspace(0.5, 0.9, 8)

    message = DepthPoseDetector.pose_message(header(), points, confidence)

    assert len(message.points) == 8
    assert message.points[2].x == points[2, 0]
    assert message.points[2].y == points[2, 1]
    assert message.points[2].z == points[2, 2]
    assert message.channels[0].name == "confidence"
    np.testing.assert_allclose(message.channels[0].values, confidence)
    assert message.channels[1].name == "landmark_id"
    assert list(message.channels[1].values) == list(range(8))
    assert message.channels[2].name == "depth_valid"
    assert list(message.channels[2].values) == [1.0] * 8


def test_invalid_pose_is_an_empty_point_cloud():
    message = DepthPoseDetector.pose_message(header())

    assert not message.points
    assert not message.channels


def test_debug_landmarks_keep_valid_points_when_one_depth_is_missing():
    points = np.ones((8, 3), dtype=float)
    points[2] = np.nan
    confidence = np.full(8, 0.8)

    message = DepthPoseDetector.landmarks_message(
        header(), points, confidence
    )
    safe_message = DepthPoseDetector.pose_message(
        header(), points, confidence
    )

    assert len(message.points) == 8
    assert np.isnan(message.points[2].z)
    assert list(message.channels[2].values) == [
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    assert not safe_message.points


def test_debug_image_is_published_as_contiguous_bgr():
    frame = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    message = DepthPoseDetector.debug_image_message(header(), frame)

    assert message.width == 2
    assert message.height == 1
    assert message.encoding == "bgr8"
    assert message.step == 6
    assert bytes(message.data) == frame.tobytes()


def test_controller_decodes_portable_pose_message():
    points = np.arange(24, dtype=float).reshape(8, 3) * 0.1
    confidence = np.full(8, 0.8)
    message = DepthPoseDetector.pose_message(header(), points, confidence)
    controller = object.__new__(DepthArmController)
    controller.min_landmark_confidence = 0.45

    decoded, error = controller.decode_pose(message)

    assert error is None
    np.testing.assert_allclose(decoded, points)


def test_controller_rejects_low_confidence_pose():
    points = np.arange(24, dtype=float).reshape(8, 3) * 0.1
    confidence = np.full(8, 0.8)
    confidence[3] = 0.2
    message = DepthPoseDetector.pose_message(header(), points, confidence)
    controller = object.__new__(DepthArmController)
    controller.min_landmark_confidence = 0.45

    decoded, error = controller.decode_pose(message)

    assert decoded is None
    assert error == "low-confidence RGB-D landmark"


def test_camera_calibration_uses_torso_without_requiring_arm_pose():
    points = np.ones((8, 3), dtype=float)
    points[[1, 2, 5, 6]] = np.nan
    confidence = np.asarray([0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9])
    message = DepthPoseDetector.landmarks_message(
        header(), points, confidence
    )
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.55

    decoded, error = controller.decode_torso_landmarks(message)

    assert error is None
    np.testing.assert_allclose(decoded[[0, 3, 4, 7]], 1.0)


def test_camera_calibration_rejects_missing_torso_depth():
    points = np.ones((8, 3), dtype=float)
    points[3] = np.nan
    confidence = np.full(8, 0.9)
    message = DepthPoseDetector.landmarks_message(
        header(), points, confidence
    )
    controller = object.__new__(DepthArmController)
    controller.min_torso_confidence = 0.55

    decoded, error = controller.decode_torso_landmarks(message)

    assert decoded is None
    assert error == "missing shoulder/hip depth for calibration"
