"""Compose RGB-D recognition and Nero control for any aligned-depth camera."""

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
    """Include one launch file from this package with scoped arguments."""
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


def generate_launch_description():
    """Build the complete camera-agnostic RGB-D arm pipeline."""
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
        "min_torso_confidence": "0.55",
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
        "point_smoothing_alpha": "0.45",
        "max_point_jump_m": "0.25",
        "bone_length_tolerance_ratio": "0.30",
        "neutral_calibration_sec": "3.0",
        "calibration_min_samples": "8",
        "calibration_file": (
            "~/.ros/demo2/realsense_person_calibration.json"
        ),
        "load_calibration_on_start": "true",
        "calibration_max_translation_step_m": "0.08",
        "calibration_max_rotation_step_deg": "12.0",
        "calibration_max_consecutive_outliers": "3",
        "max_person_translation_m": "1.0",
        "max_person_rotation_deg": "100.0",
        "max_direction_error_deg": "25.0",
        "max_joint_speed_deg_sec": "120.0",
        "pose_timeout_sec": "0.35",
        "publish_joint_states_enabled": "true",
        "command_output_enabled": "false",
        "person_camera_pose_topic": "/realsense/person_camera_pose",
        "person_relative_pose_topic": "/realsense/person_relative_pose",
        "calibration_status_topic": "/realsense/calibration_status",
        "start_rviz": "true",
    }
    detector_names = (
        "color_topic",
        "aligned_depth_topic",
        "camera_info_topic",
        "pose_topic",
        "landmarks_topic",
        "debug_image_topic",
        "detector_backend",
        "model_path",
        "person_confidence",
        "min_landmark_confidence",
        "model_complexity",
        "depth_uint16_scale",
        "depth_window_radius",
        "min_valid_depth_pixels",
        "depth_cluster_tolerance_m",
        "min_depth_m",
        "max_depth_m",
        "sync_tolerance_sec",
        "publish_debug_image",
        "show_gui",
    )
    controller_names = (
        "pose_topic",
        "landmarks_topic",
        "min_landmark_confidence",
        "min_torso_confidence",
        "point_smoothing_alpha",
        "max_point_jump_m",
        "bone_length_tolerance_ratio",
        "neutral_calibration_sec",
        "calibration_min_samples",
        "calibration_file",
        "load_calibration_on_start",
        "calibration_max_translation_step_m",
        "calibration_max_rotation_step_deg",
        "calibration_max_consecutive_outliers",
        "max_person_translation_m",
        "max_person_rotation_deg",
        "max_direction_error_deg",
        "max_joint_speed_deg_sec",
        "pose_timeout_sec",
        "publish_joint_states_enabled",
        "command_output_enabled",
        "person_camera_pose_topic",
        "person_relative_pose_topic",
        "calibration_status_topic",
        "start_rviz",
    )
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        include(
            "depth_pose_detector.launch.py",
            {name: LaunchConfiguration(name) for name in detector_names},
        ),
        include(
            "depth_arm_control.launch.py",
            {name: LaunchConfiguration(name) for name in controller_names},
        ),
    ])
