"""Start an Intel RealSense camera and the depth-based Nero follower."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def camera_topic(suffix):
    return [
        "/",
        LaunchConfiguration("camera_namespace"),
        "/",
        LaunchConfiguration("camera_name"),
        suffix,
    ]


def generate_launch_description():
    defaults = {
        "camera_namespace": "camera",
        "camera_name": "camera",
        "serial_no": "''",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
        "model_complexity": "0",
        "show_gui": "true",
        "start_rviz": "true",
        "command_output_enabled": "false",
    }

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
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
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            "spatial_filter.enable": LaunchConfiguration(
                "spatial_filter_enabled"
            ),
            "temporal_filter.enable": LaunchConfiguration(
                "temporal_filter_enabled"
            ),
        }.items(),
    )

    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("demo2"),
                "launch",
                "depth_camera_arm.launch.py",
            ])
        ),
        launch_arguments={
            "color_topic": camera_topic("/color/image_raw"),
            "aligned_depth_topic": camera_topic(
                "/aligned_depth_to_color/image_raw"
            ),
            "camera_info_topic": camera_topic("/color/camera_info"),
            "model_complexity": LaunchConfiguration("model_complexity"),
            "show_gui": LaunchConfiguration("show_gui"),
            "start_rviz": LaunchConfiguration("start_rviz"),
            "command_output_enabled": LaunchConfiguration(
                "command_output_enabled"
            ),
        }.items(),
    )

    # Scoping prevents the RealSense launch from warning about follower-only
    # arguments when it validates its supported parameter list.
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        GroupAction(scoped=True, actions=[realsense_launch]),
        GroupAction(scoped=True, actions=[follower_launch]),
    ])
