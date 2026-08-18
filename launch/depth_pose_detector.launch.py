"""Start only RGB-D landmark recognition for camera or rosbag debugging."""

from launch import LaunchDescription
from launch_ros.actions import Node

from demo2.launch_common import (
    DETECTOR_PARAMETER_TYPES,
    configured_parameters,
    declare_arguments,
    detector_defaults,
)


def generate_launch_description():
    """Build the recognition-only launch description."""
    defaults = detector_defaults()
    return LaunchDescription([
        *declare_arguments(defaults),
        Node(
            package="demo2",
            executable="depth_pose_detector_cpp",
            name="depth_pose_detector",
            parameters=[configured_parameters(DETECTOR_PARAMETER_TYPES)],
            output="screen",
        ),
    ])
