"""Start the C++ RealSense camera and standalone YOLO RGB-D recognition."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package_file, arguments):
    """Include a launch file with locally scoped arguments."""
    return GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("demo2"), "launch", package_file
                    ])
                ),
                launch_arguments=arguments.items(),
            )
        ],
    )


def camera_topic(suffix):
    """Build a topic from the configurable camera namespace and name."""
    return [
        "/",
        LaunchConfiguration("camera_namespace"),
        "/",
        LaunchConfiguration("camera_name"),
        suffix,
    ]


def generate_launch_description():
    """Build the camera plus YOLO recognition launch description."""
    defaults = {
        "camera_namespace": "camera",
        "camera_name": "camera",
        "serial_no": "''",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
        "model_path": PathJoinSubstitution([
            FindPackageShare("demo2"), "model", "yolo26n-pose.pt"
        ]),
        "person_confidence": "0.45",
        "min_landmark_confidence": "0.45",
        "depth_window_radius": "4",
        "min_valid_depth_pixels": "4",
        "depth_cluster_tolerance_m": "0.08",
        "min_depth_m": "0.15",
        "max_depth_m": "5.0",
        "sync_tolerance_sec": "0.05",
        "pose_topic": "/realsense/arm_pose_3d",
        "landmarks_topic": "/realsense/landmarks_3d",
        "debug_image_topic": "/realsense/arm_pose_debug",
        "publish_debug_image": "true",
        "show_gui": "true",
    }
    camera_names = (
        "camera_namespace",
        "camera_name",
        "serial_no",
        "color_profile",
        "depth_profile",
        "initial_reset",
        "spatial_filter_enabled",
        "temporal_filter_enabled",
    )
    detector_names = (
        "model_path",
        "person_confidence",
        "min_landmark_confidence",
        "depth_window_radius",
        "min_valid_depth_pixels",
        "depth_cluster_tolerance_m",
        "min_depth_m",
        "max_depth_m",
        "sync_tolerance_sec",
        "pose_topic",
        "landmarks_topic",
        "debug_image_topic",
        "publish_debug_image",
        "show_gui",
    )
    detector_arguments = {
        "color_topic": camera_topic("/color/image_raw"),
        "aligned_depth_topic": camera_topic(
            "/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": camera_topic("/color/camera_info"),
        "detector_backend": "yolo",
    }
    detector_arguments.update({
        name: LaunchConfiguration(name) for name in detector_names
    })
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        include(
            "realsense_camera.launch.py",
            {name: LaunchConfiguration(name) for name in camera_names},
        ),
        include("depth_pose_detector.launch.py", detector_arguments),
    ])
