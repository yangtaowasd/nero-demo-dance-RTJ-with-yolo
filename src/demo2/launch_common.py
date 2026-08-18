"""Shared defaults and helpers for the demo2 ROS 2 launch files."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from demo2.arm_sides import (
    DEFAULT_JOINT_STATE_TOPICS,
    DISPLAY_NAMESPACE,
)


CAMERA_DEFAULTS = {
    "camera_namespace": "usb_realsense",
    "camera_name": "d435i",
    "serial_no": "'108322074190'",
    "color_profile": "640x480x30",
    "depth_profile": "640x480x30",
    "initial_reset": "false",
    "spatial_filter_enabled": "false",
    "temporal_filter_enabled": "true",
}

CAMERA_PARAMETER_TYPES = {
    "camera_namespace": None,
    "camera_name": None,
    "serial_no": None,
    "color_profile": None,
    "depth_profile": None,
    "initial_reset": bool,
    "spatial_filter_enabled": bool,
    "temporal_filter_enabled": bool,
}

DETECTOR_PARAMETER_TYPES = {
    "color_topic": None,
    "aligned_depth_topic": None,
    "camera_info_topic": None,
    "landmarks_topic": None,
    "debug_image_topic": None,
    "model_path": None,
    "inference_size": int,
    "torch_threads": int,
    "person_confidence": float,
    "min_landmark_confidence": float,
    "target_lock_enabled": bool,
    "target_lock_min_iou": float,
    "target_lock_max_center_distance_ratio": float,
    "target_lock_max_missed_frames": int,
    "keypoint_smoothing_alpha": float,
    "kalman_tracking_enabled": bool,
    "kalman_prediction_timeout_sec": float,
    "kalman_process_noise_mps2": float,
    "kalman_measurement_noise_m": float,
    "kalman_max_velocity_mps": float,
    "depth_uint16_scale": float,
    "depth_window_radius": int,
    "min_valid_depth_pixels": int,
    "depth_cluster_tolerance_m": float,
    "min_depth_m": float,
    "max_depth_m": float,
    "sync_tolerance_sec": float,
    "sync_wait_sec": float,
    "publish_debug_image": bool,
    "show_gui": bool,
}

CONTROLLER_PARAMETER_TYPES = {
    "landmarks_topic": None,
    "urdf_file": None,
    "min_landmark_confidence": float,
    "min_torso_confidence": float,
    "torso_hold_sec": float,
    "point_smoothing_alpha": float,
    "point_median_window": int,
    "adaptive_point_filter_enabled": bool,
    "point_fast_smoothing_alpha": float,
    "point_motion_start_m": float,
    "point_motion_full_m": float,
    "max_point_jump_m": float,
    "bone_length_tolerance_ratio": float,
    "neutral_calibration_sec": float,
    "calibration_min_samples": int,
    "calibration_file": None,
    "calibration_camera_id": str,
    "load_calibration_on_start": bool,
    "calibration_max_translation_step_m": float,
    "calibration_max_rotation_step_deg": float,
    "calibration_max_consecutive_outliers": int,
    "max_person_translation_m": float,
    "max_person_rotation_deg": float,
    "max_direction_error_deg": float,
    "ik_max_iterations": int,
    "control_rate_hz": float,
    "max_joint_speed_deg_sec": float,
    "joint_smoothing_tau_sec": float,
    "adaptive_joint_smoothing_enabled": bool,
    "joint_fast_smoothing_tau_sec": float,
    "joint_motion_start_deg": float,
    "joint_motion_full_deg": float,
    "joint_deadband_deg": float,
    "pose_timeout_sec": float,
    "publish_joint_states_enabled": bool,
    "command_output_enabled": bool,
    "exit_if_parent_changes": bool,
    "left_joint_state_topic": None,
    "right_joint_state_topic": None,
    "left_command_topic": None,
    "right_command_topic": None,
    "left_tracking_status_topic": None,
    "right_tracking_status_topic": None,
    "person_camera_pose_topic": None,
    "person_relative_pose_topic": None,
    "calibration_status_topic": None,
}

HARDWARE_PARAMETER_TYPES = {
    "left_hardware_enabled": bool,
    "right_hardware_enabled": bool,
    "left_can_interface": None,
    "right_can_interface": None,
    "left_firmware": None,
    "right_firmware": None,
    "urdf_file": None,
    "hardware_execute_motion": bool,
    "hardware_auto_enable": bool,
    "hardware_reset_emergency_stop_on_start": bool,
    "hardware_emergency_reset_timeout_sec": float,
    "hardware_require_command_before_enable": bool,
    "hardware_motion_start_delay_sec": float,
    "return_to_home_on_shutdown": bool,
    "shutdown_return_timeout_sec": float,
    "shutdown_position_tolerance_deg": float,
    "hardware_shutdown_sigterm_timeout_sec": float,
    "disable_on_shutdown": bool,
    "hardware_rate_hz": float,
    "hardware_command_timeout_sec": float,
    "hardware_feedback_timeout_sec": float,
    "hardware_connect_timeout_sec": float,
    "hardware_probe_reconnect_delay_sec": float,
    "hardware_reconnect_interval_sec": float,
    "hardware_enable_timeout_sec": float,
    "hardware_max_command_speed_deg_sec": float,
    "hardware_speed_percent": int,
    "exit_if_parent_changes": bool,
}


def package_path(*parts):
    """Return a deferred path inside this package's share directory."""
    return PathJoinSubstitution([FindPackageShare("demo2"), *parts])


def detector_defaults():
    """Return launch defaults for the C++ RGB-D pose detector."""
    return {
        "color_topic": "/camera/camera/color/image_raw",
        "aligned_depth_topic": (
            "/camera/camera/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": "/camera/camera/color/camera_info",
        "landmarks_topic": "/realsense/landmarks_3d",
        "debug_image_topic": "/realsense/arm_pose_debug",
        "model_path": package_path("model", "yolo26n-pose.torchscript"),
        "inference_size": "384",
        "torch_threads": "1",
        "person_confidence": "0.30",
        "min_landmark_confidence": "0.35",
        "target_lock_enabled": "true",
        "target_lock_min_iou": "0.05",
        "target_lock_max_center_distance_ratio": "1.25",
        "target_lock_max_missed_frames": "8",
        "keypoint_smoothing_alpha": "0.55",
        "kalman_tracking_enabled": "true",
        "kalman_prediction_timeout_sec": "0.35",
        "kalman_process_noise_mps2": "5.0",
        "kalman_measurement_noise_m": "0.025",
        "kalman_max_velocity_mps": "3.0",
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
    }


def controller_defaults():
    """Return launch defaults for pose filtering and Nero IK/control."""
    return {
        "landmarks_topic": "/realsense/landmarks_3d",
        "urdf_file": package_path("urdf", "nero_description.urdf"),
        "min_landmark_confidence": "0.35",
        "min_torso_confidence": "0.45",
        "torso_hold_sec": "0.25",
        "point_smoothing_alpha": "0.30",
        "point_median_window": "3",
        "adaptive_point_filter_enabled": "true",
        "point_fast_smoothing_alpha": "0.85",
        "point_motion_start_m": "0.015",
        "point_motion_full_m": "0.060",
        "max_point_jump_m": "0.25",
        "bone_length_tolerance_ratio": "0.30",
        "neutral_calibration_sec": "2.0",
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
        "adaptive_joint_smoothing_enabled": "true",
        "joint_fast_smoothing_tau_sec": "0.04",
        "joint_motion_start_deg": "1.0",
        "joint_motion_full_deg": "8.0",
        "joint_deadband_deg": "0.35",
        "pose_timeout_sec": "0.35",
        "publish_joint_states_enabled": "true",
        "command_output_enabled": "false",
        "exit_if_parent_changes": "true",
        "left_joint_state_topic": DEFAULT_JOINT_STATE_TOPICS["left"],
        "right_joint_state_topic": DEFAULT_JOINT_STATE_TOPICS["right"],
        "left_command_topic": "/left/neroarm/command_joints",
        "right_command_topic": "/right/neroarm/command_joints",
        "left_tracking_status_topic": "/left/tracking_status",
        "right_tracking_status_topic": "/right/tracking_status",
        "person_camera_pose_topic": "/realsense/person_camera_pose",
        "person_relative_pose_topic": "/realsense/person_relative_pose",
        "calibration_status_topic": "/realsense/calibration_status",
        "start_rviz": "true",
    }


def hardware_defaults():
    """Return safe-by-default launch settings for the two arm drivers."""
    return {
        "left_hardware_enabled": "true",
        "right_hardware_enabled": "true",
        "left_can_interface": "can1",
        "right_can_interface": "can0",
        "left_firmware": "v111",
        "right_firmware": "v111",
        "urdf_file": package_path("urdf", "nero_description.urdf"),
        "hardware_execute_motion": "false",
        "hardware_auto_enable": "true",
        "hardware_reset_emergency_stop_on_start": "true",
        "hardware_emergency_reset_timeout_sec": "5.0",
        "hardware_require_command_before_enable": "false",
        "hardware_motion_start_delay_sec": "5.0",
        "return_to_home_on_shutdown": "true",
        "shutdown_return_timeout_sec": "8.0",
        "shutdown_position_tolerance_deg": "1.5",
        "hardware_shutdown_sigterm_timeout_sec": "12.0",
        "disable_on_shutdown": "false",
        "hardware_rate_hz": "20.0",
        "hardware_command_timeout_sec": "0.35",
        "hardware_feedback_timeout_sec": "0.50",
        "hardware_connect_timeout_sec": "2.0",
        "hardware_probe_reconnect_delay_sec": "0.5",
        "hardware_reconnect_interval_sec": "2.0",
        "hardware_enable_timeout_sec": "5.0",
        "hardware_max_command_speed_deg_sec": "30.0",
        "hardware_speed_percent": "20",
        "exit_if_parent_changes": "true",
    }


def declare_arguments(defaults):
    """Create declarations for every item in a defaults mapping."""
    return [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in defaults.items()
    ]


def configured_arguments(names):
    """Map launch argument names to their deferred values."""
    return {name: LaunchConfiguration(name) for name in names}


def configured_parameters(parameter_types):
    """Build a ROS parameter mapping with explicit scalar types."""
    return {
        name: (
            LaunchConfiguration(name)
            if value_type is None
            else ParameterValue(
                LaunchConfiguration(name), value_type=value_type
            )
        )
        for name, value_type in parameter_types.items()
    }


def include_launch(filename, arguments, condition=None):
    """Include one launch file from this package with scoped arguments."""
    return GroupAction(
        scoped=True,
        condition=condition,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    package_path("launch", filename)
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


def robot_description():
    """Read the package-local plain URDF."""
    model = (
        Path(get_package_share_directory("demo2"))
        / "urdf/nero_description.urdf"
    )
    return {"robot_description": model.read_text(encoding="utf-8")}


def arm_state_publishers(side, y_offset, pitch, yaw, name_suffix):
    """Create TF and robot-state publishers for one arm."""
    namespace = f"{DISPLAY_NAMESPACE}/{side}"
    prefix = f"{namespace}/"
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            namespace=DISPLAY_NAMESPACE,
            name=f"{side}_{name_suffix}",
            arguments=[
                "--x", "0.0", "--y", str(y_offset), "--z", "0.70",
                "--roll", "0.0", "--pitch", str(pitch), "--yaw", str(yaw),
                "--frame-id", "world", "--child-frame-id", f"{prefix}world",
            ],
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[robot_description(), {"frame_prefix": prefix}],
            remappings=[
                ("joint_states", DEFAULT_JOINT_STATE_TOPICS[side])
            ],
            output="screen",
        ),
    ]
