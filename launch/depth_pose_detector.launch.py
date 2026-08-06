"""Start only RGB-D landmark recognition for camera or rosbag debugging."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def typed(name, value_type):
    """Return a typed launch parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    """Build the recognition-only launch description."""
    defaults = {
        "color_topic": "/camera/camera/color/image_raw",
        "aligned_depth_topic": (
            "/camera/camera/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": "/camera/camera/color/camera_info",
        "pose_topic": "/realsense/arm_pose_3d",
        "landmarks_topic": "/realsense/landmarks_3d",
        "debug_image_topic": "/realsense/arm_pose_debug",
        "detector_backend": "yolo",
        "model_path": PathJoinSubstitution([
            FindPackageShare("demo2"), "model", "yolo26n-pose.pt"
        ]),
        "person_confidence": "0.45",
        "min_landmark_confidence": "0.45",
        "model_complexity": "0",
        "depth_uint16_scale": "0.001",
        "depth_window_radius": "4",
        "min_valid_depth_pixels": "4",
        "depth_cluster_tolerance_m": "0.08",
        "min_depth_m": "0.15",
        "max_depth_m": "5.0",
        "sync_tolerance_sec": "0.05",
        "publish_debug_image": "true",
        "show_gui": "true",
    }
    parameters = {
        "color_topic": LaunchConfiguration("color_topic"),
        "aligned_depth_topic": LaunchConfiguration("aligned_depth_topic"),
        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
        "pose_topic": LaunchConfiguration("pose_topic"),
        "landmarks_topic": LaunchConfiguration("landmarks_topic"),
        "debug_image_topic": LaunchConfiguration("debug_image_topic"),
        "detector_backend": LaunchConfiguration("detector_backend"),
        "model_path": LaunchConfiguration("model_path"),
        "person_confidence": typed("person_confidence", float),
        "min_landmark_confidence": typed(
            "min_landmark_confidence", float
        ),
        "model_complexity": typed("model_complexity", int),
        "depth_uint16_scale": typed("depth_uint16_scale", float),
        "depth_window_radius": typed("depth_window_radius", int),
        "min_valid_depth_pixels": typed("min_valid_depth_pixels", int),
        "depth_cluster_tolerance_m": typed(
            "depth_cluster_tolerance_m", float
        ),
        "min_depth_m": typed("min_depth_m", float),
        "max_depth_m": typed("max_depth_m", float),
        "sync_tolerance_sec": typed("sync_tolerance_sec", float),
        "publish_debug_image": typed("publish_debug_image", bool),
        "show_gui": typed("show_gui", bool),
    }
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        Node(
            package="demo2",
            executable="depth_pose_detector.py",
            name="depth_pose_detector",
            parameters=[parameters],
            output="screen",
        ),
    ])
