"""Start the single-process C++ RealSense RGB, depth, and fusion viewer."""

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

from demo2.launch_common import (
    CAMERA_DEFAULTS,
    CAMERA_PARAMETER_TYPES,
    configured_parameters,
    declare_arguments,
)


VIEWER_DEFAULTS = {
    "min_depth_m": "0.15",
    "max_depth_m": "8.0",
    "fusion_alpha": "0.45",
    "show_gui": "true",
    "publish_composite": "true",
    "output_topic": "/realsense/rgbd_view",
}

VIEWER_PARAMETER_TYPES = {
    "min_depth_m": float,
    "max_depth_m": float,
    "fusion_alpha": float,
    "show_gui": bool,
    "publish_composite": bool,
    "output_topic": None,
}


def generate_launch_description():
    """Build the C++ camera plus three-panel viewer launch."""
    defaults = {**CAMERA_DEFAULTS, **VIEWER_DEFAULTS}
    parameter_types = {
        **CAMERA_PARAMETER_TYPES,
        **VIEWER_PARAMETER_TYPES,
    }
    return LaunchDescription([
        SetEnvironmentVariable(name="ROS_LOCALHOST_ONLY", value="1"),
        *declare_arguments(defaults),
        Node(
            package="demo2",
            executable="realsense_rgbd_viewer_cpp",
            name="realsense_rgbd_viewer_cpp",
            parameters=[configured_parameters(parameter_types)],
            output="screen",
        ),
    ])
