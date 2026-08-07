"""Start the single-process C++ RealSense RGB, depth, and fusion viewer."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, value_type):
    """Return a typed launch parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    """Build the C++ camera plus three-panel viewer launch."""
    defaults = {
        "camera_namespace": "usb_realsense",
        "camera_name": "d435i",
        "serial_no": "'108322074190'",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
        "min_depth_m": "0.15",
        "max_depth_m": "5.0",
        "fusion_alpha": "0.45",
        "show_gui": "true",
        "publish_composite": "true",
        "output_topic": "/realsense/rgbd_view",
    }
    parameters = {
        "camera_namespace": LaunchConfiguration("camera_namespace"),
        "camera_name": LaunchConfiguration("camera_name"),
        "serial_no": LaunchConfiguration("serial_no"),
        "color_profile": LaunchConfiguration("color_profile"),
        "depth_profile": LaunchConfiguration("depth_profile"),
        "initial_reset": typed("initial_reset", bool),
        "spatial_filter_enabled": typed(
            "spatial_filter_enabled", bool
        ),
        "temporal_filter_enabled": typed(
            "temporal_filter_enabled", bool
        ),
        "min_depth_m": typed("min_depth_m", float),
        "max_depth_m": typed("max_depth_m", float),
        "fusion_alpha": typed("fusion_alpha", float),
        "show_gui": typed("show_gui", bool),
        "publish_composite": typed("publish_composite", bool),
        "output_topic": LaunchConfiguration("output_topic"),
    }
    return LaunchDescription([
        SetEnvironmentVariable(
            name="ROS_LOCALHOST_ONLY", value="1"
        ),
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        Node(
            package="demo2",
            executable="realsense_rgbd_viewer_cpp",
            name="realsense_rgbd_viewer_cpp",
            parameters=[parameters],
            output="screen",
        ),
    ])
