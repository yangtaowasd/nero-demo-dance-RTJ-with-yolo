"""Start RealSense RGB-D topics with an automatic local-driver fallback."""

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, value_type):
    """Return a typed launch parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def camera_driver_action():
    """Use the official driver or the local librealsense bridge."""
    try:
        driver_share = get_package_share_directory("realsense2_camera")
    except PackageNotFoundError:
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
            "show_gui": False,
            "publish_composite": False,
        }
        return GroupAction(actions=[
            LogInfo(
                msg=(
                    "realsense2_camera is not installed; using demo2's "
                    "C++ librealsense bridge"
                )
            ),
            Node(
                package="demo2",
                executable="realsense_rgbd_viewer_cpp",
                name="realsense_cpp_camera",
                parameters=[parameters],
                output="screen",
            ),
        ])

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{driver_share}/launch/rs_launch.py"
        ),
        launch_arguments={
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "serial_no": LaunchConfiguration("serial_no"),
            "initial_reset": LaunchConfiguration("initial_reset"),
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "enable_rgbd": "false",
            "pointcloud.enable": "false",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration(
                "depth_profile"
            ),
            "spatial_filter.enable": LaunchConfiguration(
                "spatial_filter_enabled"
            ),
            "temporal_filter.enable": LaunchConfiguration(
                "temporal_filter_enabled"
            ),
        }.items(),
    )
    return GroupAction(scoped=True, actions=[driver])


def generate_launch_description():
    """Build the camera-only launch description."""
    defaults = {
        "camera_namespace": "usb_realsense",
        "camera_name": "d435i",
        "serial_no": "'108322074190'",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
    }
    return LaunchDescription([
        SetEnvironmentVariable(
            name="ROS_LOCALHOST_ONLY", value="1"
        ),
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        camera_driver_action(),
    ])
