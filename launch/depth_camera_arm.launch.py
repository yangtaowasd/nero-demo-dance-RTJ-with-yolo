"""Compose RGB-D recognition and Nero control for any aligned-depth camera."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package_file, arguments, condition=None):
    """Include one launch file from this package with scoped arguments."""
    return GroupAction(
        scoped=True,
        condition=condition,
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
        "min_torso_confidence": "0.45",
        "torso_hold_sec": "0.25",
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
        "max_depth_m": "8.0",
        "sync_tolerance_sec": "0.02",
        "sync_wait_sec": "0.02",
        "publish_debug_image": "true",
        "show_gui": "true",
        "point_smoothing_alpha": "0.30",
        "point_median_window": "3",
        "max_point_jump_m": "0.25",
        "bone_length_tolerance_ratio": "0.30",
        "neutral_calibration_sec": "3.0",
        "calibration_min_samples": "8",
        "calibration_file": (
            "~/.ros/demo2/realsense_person_calibration.json"
        ),
        "calibration_camera_id": "unspecified",
        "load_calibration_on_start": "true",
        "calibration_max_translation_step_m": "0.08",
        "calibration_max_rotation_step_deg": "12.0",
        "calibration_max_consecutive_outliers": "3",
        "max_person_translation_m": "1.0",
        "max_person_rotation_deg": "100.0",
        "max_direction_error_deg": "25.0",
        "ik_max_iterations": "8",
        "control_rate_hz": "20.0",
        "max_joint_speed_deg_sec": "120.0",
        "joint_smoothing_tau_sec": "0.20",
        "joint_deadband_deg": "0.35",
        "pose_timeout_sec": "0.35",
        "publish_joint_states_enabled": "true",
        "command_output_enabled": "false",
        "person_camera_pose_topic": "/realsense/person_camera_pose",
        "person_relative_pose_topic": "/realsense/person_relative_pose",
        "calibration_status_topic": "/realsense/calibration_status",
        "left_tracking_status_topic": "/left/tracking_status",
        "right_tracking_status_topic": "/right/tracking_status",
        "start_rviz": "true",
        "start_hardware": "false",
        "left_hardware_enabled": "true",
        "right_hardware_enabled": "true",
        "left_can_interface": "can0",
        "right_can_interface": "can1",
        "left_firmware": "v111",
        "right_firmware": "v111",
        "hardware_execute_motion": "false",
        "disable_on_shutdown": "false",
        "hardware_rate_hz": "20.0",
        "hardware_command_timeout_sec": "0.35",
        "hardware_feedback_timeout_sec": "0.50",
        "hardware_connect_timeout_sec": "2.0",
        "hardware_reconnect_interval_sec": "2.0",
        "hardware_enable_timeout_sec": "5.0",
        "hardware_max_command_speed_deg_sec": "30.0",
        "hardware_speed_percent": "20",
    }
    detector_names = (
        "color_topic",
        "aligned_depth_topic",
        "camera_info_topic",
        "landmarks_topic",
        "debug_image_topic",
        "model_path",
        "inference_size",
        "torch_threads",
        "person_confidence",
        "min_landmark_confidence",
        "target_lock_enabled",
        "target_lock_min_iou",
        "target_lock_max_center_distance_ratio",
        "target_lock_max_missed_frames",
        "keypoint_smoothing_alpha",
        "depth_uint16_scale",
        "depth_window_radius",
        "min_valid_depth_pixels",
        "depth_cluster_tolerance_m",
        "min_depth_m",
        "max_depth_m",
        "sync_tolerance_sec",
        "sync_wait_sec",
        "publish_debug_image",
        "show_gui",
    )
    controller_names = (
        "landmarks_topic",
        "min_landmark_confidence",
        "min_torso_confidence",
        "torso_hold_sec",
        "point_smoothing_alpha",
        "point_median_window",
        "max_point_jump_m",
        "bone_length_tolerance_ratio",
        "neutral_calibration_sec",
        "calibration_min_samples",
        "calibration_file",
        "calibration_camera_id",
        "load_calibration_on_start",
        "calibration_max_translation_step_m",
        "calibration_max_rotation_step_deg",
        "calibration_max_consecutive_outliers",
        "max_person_translation_m",
        "max_person_rotation_deg",
        "max_direction_error_deg",
        "ik_max_iterations",
        "control_rate_hz",
        "max_joint_speed_deg_sec",
        "joint_smoothing_tau_sec",
        "joint_deadband_deg",
        "pose_timeout_sec",
        "publish_joint_states_enabled",
        "command_output_enabled",
        "person_camera_pose_topic",
        "person_relative_pose_topic",
        "calibration_status_topic",
        "left_tracking_status_topic",
        "right_tracking_status_topic",
        "start_rviz",
    )
    hardware_names = (
        "left_hardware_enabled",
        "right_hardware_enabled",
        "left_can_interface",
        "right_can_interface",
        "left_firmware",
        "right_firmware",
        "hardware_execute_motion",
        "disable_on_shutdown",
        "hardware_rate_hz",
        "hardware_command_timeout_sec",
        "hardware_feedback_timeout_sec",
        "hardware_connect_timeout_sec",
        "hardware_reconnect_interval_sec",
        "hardware_enable_timeout_sec",
        "hardware_max_command_speed_deg_sec",
        "hardware_speed_percent",
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
        include(
            "dual_nero_pyagxarm.launch.py",
            {name: LaunchConfiguration(name) for name in hardware_names},
            condition=IfCondition(LaunchConfiguration("start_hardware")),
        ),
    ])
