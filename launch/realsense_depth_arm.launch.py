"""Compose the RealSense driver, portable recognition, and Nero control."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def camera_topic(suffix):
    """Build a topic from the configurable RealSense namespace/name."""
    return [
        "/",
        LaunchConfiguration("camera_namespace"),
        "/",
        LaunchConfiguration("camera_name"),
        suffix,
    ]


def package_launch(filename, arguments):
    """Include a scoped launch file from this package."""
    return GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("demo2"), "launch", filename
                    ])
                ),
                launch_arguments=arguments.items(),
            )
        ],
    )


def generate_launch_description():
    """Build the one-command RealSense arm pipeline."""
    defaults = {
        "camera_namespace": "camera",
        "camera_name": "camera",
        "serial_no": "''",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
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
        "show_gui": "true",
        "publish_debug_image": "true",
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
        "start_rviz": "true",
        "command_output_enabled": "false",
        "person_camera_pose_topic": "/realsense/person_camera_pose",
        "person_relative_pose_topic": "/realsense/person_relative_pose",
        "calibration_status_topic": "/realsense/calibration_status",
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
    pipeline_arguments = {
        "color_topic": camera_topic("/color/image_raw"),
        "aligned_depth_topic": camera_topic(
            "/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": camera_topic("/color/camera_info"),
    }
    pipeline_names = tuple(
        name for name in defaults if name not in camera_names
    )
    pipeline_arguments.update({
        name: LaunchConfiguration(name) for name in pipeline_names
    })
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        package_launch(
            "realsense_camera.launch.py",
            {name: LaunchConfiguration(name) for name in camera_names},
        ),
        package_launch("depth_camera_arm.launch.py", pipeline_arguments),
    ])
