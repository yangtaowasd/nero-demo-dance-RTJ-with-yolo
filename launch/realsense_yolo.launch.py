"""Start the C++ RealSense camera and standalone YOLO RGB-D recognition."""

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable

from demo2.launch_common import (
    CAMERA_DEFAULTS,
    camera_topic,
    configured_arguments,
    declare_arguments,
    detector_defaults,
    include_launch,
)


CAMERA_TOPIC_ARGUMENTS = (
    "color_topic",
    "aligned_depth_topic",
    "camera_info_topic",
)


def generate_launch_description():
    """Build the camera plus YOLO recognition launch description."""
    detector = detector_defaults()
    for name in CAMERA_TOPIC_ARGUMENTS:
        detector.pop(name)

    defaults = {**CAMERA_DEFAULTS, **detector}
    detector_arguments = configured_arguments(detector)
    detector_arguments.update({
        "color_topic": camera_topic("/color/image_raw"),
        "aligned_depth_topic": camera_topic(
            "/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": camera_topic("/color/camera_info"),
    })
    return LaunchDescription([
        SetEnvironmentVariable(name="ROS_LOCALHOST_ONLY", value="1"),
        *declare_arguments(defaults),
        include_launch(
            "realsense_camera.launch.py",
            configured_arguments(CAMERA_DEFAULTS),
        ),
        include_launch("depth_pose_detector.launch.py", detector_arguments),
    ])
