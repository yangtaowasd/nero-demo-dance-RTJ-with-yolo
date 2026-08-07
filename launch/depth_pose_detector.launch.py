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
        "landmarks_topic": "/realsense/landmarks_3d",
        "debug_image_topic": "/realsense/arm_pose_debug",
        "model_path": PathJoinSubstitution([
            FindPackageShare("demo2"),
            "model",
            "yolo26n-pose.torchscript",
        ]),
        "inference_size": "384",
        "torch_threads": "1",
        "person_confidence": "0.30",
        "min_landmark_confidence": "0.35",
        "target_lock_enabled": "true",
        "target_lock_min_iou": "0.05",
        "target_lock_max_center_distance_ratio": "1.25",
        "target_lock_max_missed_frames": "8",
        "keypoint_smoothing_alpha": "0.55",
        "depth_uint16_scale": "0.001",
        "depth_window_radius": "4",
        "min_valid_depth_pixels": "4",
        "depth_cluster_tolerance_m": "0.08",
        "min_depth_m": "0.15",
        "max_depth_m": "5.0",
        "sync_tolerance_sec": "0.02",
        "sync_wait_sec": "0.02",
        "publish_debug_image": "true",
        "show_gui": "true",
    }
    parameters = {
        "color_topic": LaunchConfiguration("color_topic"),
        "aligned_depth_topic": LaunchConfiguration("aligned_depth_topic"),
        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
        "landmarks_topic": LaunchConfiguration("landmarks_topic"),
        "debug_image_topic": LaunchConfiguration("debug_image_topic"),
        "model_path": LaunchConfiguration("model_path"),
        "person_confidence": typed("person_confidence", float),
        "min_landmark_confidence": typed(
            "min_landmark_confidence", float
        ),
        "target_lock_enabled": typed("target_lock_enabled", bool),
        "target_lock_min_iou": typed("target_lock_min_iou", float),
        "target_lock_max_center_distance_ratio": typed(
            "target_lock_max_center_distance_ratio", float
        ),
        "target_lock_max_missed_frames": typed(
            "target_lock_max_missed_frames", int
        ),
        "keypoint_smoothing_alpha": typed(
            "keypoint_smoothing_alpha", float
        ),
        "depth_uint16_scale": typed("depth_uint16_scale", float),
        "depth_window_radius": typed("depth_window_radius", int),
        "min_valid_depth_pixels": typed("min_valid_depth_pixels", int),
        "depth_cluster_tolerance_m": typed(
            "depth_cluster_tolerance_m", float
        ),
        "min_depth_m": typed("min_depth_m", float),
        "max_depth_m": typed("max_depth_m", float),
        "sync_tolerance_sec": typed("sync_tolerance_sec", float),
        "sync_wait_sec": typed("sync_wait_sec", float),
        "publish_debug_image": typed("publish_debug_image", bool),
        "show_gui": typed("show_gui", bool),
    }
    parameters.update({
        "inference_size": typed("inference_size", int),
        "torch_threads": typed("torch_threads", int),
    })
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        Node(
            package="demo2",
            executable="depth_pose_detector_cpp",
            name="depth_pose_detector",
            parameters=[parameters],
            output="screen",
        ),
    ])
