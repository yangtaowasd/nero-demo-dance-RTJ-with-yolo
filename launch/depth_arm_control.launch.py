"""Start only Nero IK/control for an external RGB-D landmark topic."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_description():
    """Read the plain local URDF without requiring the xacro package."""
    model = (
        Path(get_package_share_directory("demo2"))
        / "urdf/nero_description.urdf"
    )
    return {"robot_description": model.read_text(encoding="utf-8")}


def state_publishers(namespace, y_offset, pitch, yaw):
    """Create TF and robot-state publishers for one arm."""
    prefix = f"{namespace}/"
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{namespace}_depth_mount_tf",
            arguments=[
                "--x", "0.0", "--y", str(y_offset), "--z", "0.70",
                "--roll", "0.0", "--pitch", str(pitch), "--yaw", str(yaw),
                "--frame-id", "world", "--child-frame-id", f"{prefix}world",
            ],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[robot_description(), {"frame_prefix": prefix}],
            remappings=[("joint_states", f"/{namespace}/joint_states")],
            output="screen",
        ),
    ]


def typed(name, value_type):
    """Return a typed launch parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    """Build the robot-control-only launch description."""
    share = FindPackageShare("demo2")
    defaults = {
        "pose_topic": "/realsense/arm_pose_3d",
        "landmarks_topic": "/realsense/landmarks_3d",
        "urdf_file": PathJoinSubstitution(
            [share, "urdf", "nero_description.urdf"]
        ),
        "min_landmark_confidence": "0.45",
        "min_torso_confidence": "0.55",
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
        "left_joint_state_topic": "/left/joint_states",
        "right_joint_state_topic": "/right/joint_states",
        "left_command_topic": "/left/neroarm/command_joints",
        "right_command_topic": "/right/neroarm/command_joints",
        "person_camera_pose_topic": "/realsense/person_camera_pose",
        "person_relative_pose_topic": "/realsense/person_relative_pose",
        "calibration_status_topic": "/realsense/calibration_status",
        "start_rviz": "true",
    }
    parameters = {
        "pose_topic": LaunchConfiguration("pose_topic"),
        "landmarks_topic": LaunchConfiguration("landmarks_topic"),
        "urdf_file": LaunchConfiguration("urdf_file"),
        "min_landmark_confidence": typed(
            "min_landmark_confidence", float
        ),
        "min_torso_confidence": typed("min_torso_confidence", float),
        "point_smoothing_alpha": typed("point_smoothing_alpha", float),
        "max_point_jump_m": typed("max_point_jump_m", float),
        "bone_length_tolerance_ratio": typed(
            "bone_length_tolerance_ratio", float
        ),
        "neutral_calibration_sec": typed("neutral_calibration_sec", float),
        "calibration_min_samples": typed(
            "calibration_min_samples", int
        ),
        "calibration_file": LaunchConfiguration("calibration_file"),
        "load_calibration_on_start": typed(
            "load_calibration_on_start", bool
        ),
        "calibration_max_translation_step_m": typed(
            "calibration_max_translation_step_m", float
        ),
        "calibration_max_rotation_step_deg": typed(
            "calibration_max_rotation_step_deg", float
        ),
        "calibration_max_consecutive_outliers": typed(
            "calibration_max_consecutive_outliers", int
        ),
        "max_person_translation_m": typed(
            "max_person_translation_m", float
        ),
        "max_person_rotation_deg": typed(
            "max_person_rotation_deg", float
        ),
        "max_direction_error_deg": typed("max_direction_error_deg", float),
        "max_joint_speed_deg_sec": typed(
            "max_joint_speed_deg_sec", float
        ),
        "pose_timeout_sec": typed("pose_timeout_sec", float),
        "publish_joint_states_enabled": typed(
            "publish_joint_states_enabled", bool
        ),
        "command_output_enabled": typed("command_output_enabled", bool),
        "left_joint_state_topic": LaunchConfiguration(
            "left_joint_state_topic"
        ),
        "right_joint_state_topic": LaunchConfiguration(
            "right_joint_state_topic"
        ),
        "left_command_topic": LaunchConfiguration("left_command_topic"),
        "right_command_topic": LaunchConfiguration("right_command_topic"),
        "person_camera_pose_topic": LaunchConfiguration(
            "person_camera_pose_topic"
        ),
        "person_relative_pose_topic": LaunchConfiguration(
            "person_relative_pose_topic"
        ),
        "calibration_status_topic": LaunchConfiguration(
            "calibration_status_topic"
        ),
    }
    rviz_config = PathJoinSubstitution([share, "config", "dual_nero.rviz"])
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        *state_publishers("left", 0.35, -1.5707963, -1.5707963),
        *state_publishers("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="demo2",
            executable="depth_arm_controller.py",
            name="depth_arm_controller",
            parameters=[parameters],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ])
