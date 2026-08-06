from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


ARG_DEFAULTS = {
    "pose_backend": "mediapipe_3d",
    "pose_3d_topic": "/arm_pose_3d",
    "start_pose": "true",
    "yolo_model_path": PathJoinSubstitution([
        FindPackageShare("demo2"),
        "model",
        "yolo26s-pose.pt",
    ]),
    "camera_id": "0",
    "camera_width": "640",
    "camera_height": "480",
    "camera_fps": "30.0",
    "camera_fourcc": "MJPG",
    "start_yolo": "true",
    "mediapipe_model_complexity": "1",
    "pose_3d_calibration_duration": "2.0",
    "pose_3d_still_threshold_deg": "8.0",
    "retarget_j1_gain": "0.8",
    "retarget_j2_gain": "0.8",
    "retarget_j4_gain": "1.0",
    "pose_timeout_sec": "0.5",
    "show_gui": "true",
    "center_roi_enabled": "true",
    "center_roi_fraction": "0.80",
    "require_all_arm_points_in_roi": "true",
    "image_points_topic": "/arm_pose_v2/image_points",
    "stable_pose_required": "true",
    "stable_pose_duration": "1.0",
    "pose_read_duration": "2.0",
    "stable_motion_threshold_px": "15.0",
    "stable_range_threshold_px": "20.0",
    "show_initial_pose_guide": "true",
    "start_real_driver": "false",
    "execute_motion": "false",
    "connect_on_start": "true",
    "auto_enable": "true",
    "speed_percent": "20",
    "max_command_delta": "0.10",
    "home_on_shutdown": "true",
    "shutdown_home_timeout": "8.0",
    "shutdown_home_tolerance": "0.03",
    "safety_enabled": "true",
    "left_can_interface": "can0",
    "right_can_interface": "can1",
    "left_status_topic": "/left/neroarm/status_joints",
    "right_status_topic": "/right/neroarm/status_joints",
    "command_output_enabled": "false",
    "left_command_topic": "/left/neroarm/command_joints",
    "right_command_topic": "/right/neroarm/command_joints",
    "yolo_calibration_duration": "2.0",
    "yolo_still_threshold_px": "15.0",
    "yolo_j1_home_deg": "0.0",
    "yolo_j2_home_deg": "0.0",
    "yolo_j1_forward_max_deg": "70.0",
    "yolo_j1_forward_full_ratio": "0.45",
    "yolo_j2_gain": "1.0",
    "yolo_j4_gain": "1.0",
    "joint_smoothing_tau": "0.15",
    "joint_deadband_deg": "0.8",
    "yolo_left_j2_sign": "1.0",
    "yolo_right_j2_sign": "-1.0",
    "lock_j1_enabled": "false",
    "lock_j2_enabled": "false",
    "lock_j5_j6_j7_enabled": "true",
    "j1_j2_only": "false",
}


def robot_description():
    model = (
        Path(get_package_share_directory("demo2"))
        / "urdf/nero_description.urdf"
    )
    return {"robot_description": model.read_text(encoding="utf-8")}


def state_publisher(namespace, y_offset, pitch, yaw):
    prefix = f"{namespace}/"
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{namespace}_mount_tf",
            arguments=[
                "--x", "0.0",
                "--y", str(y_offset),
                "--z", "0.70",
                "--roll", "0.0",
                "--pitch", str(pitch),
                "--yaw", str(yaw),
                "--frame-id", "world",
                "--child-frame-id", f"{prefix}world",
            ],
            output="screen",
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


def typed_config(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def real_driver(namespace):
    bool_params = (
        "execute_motion",
        "connect_on_start",
        "auto_enable",
        "home_on_shutdown",
        "safety_enabled",
    )
    float_params = (
        "max_command_delta",
        "shutdown_home_timeout",
        "shutdown_home_tolerance",
    )
    params = {
        "side": namespace,
        "command_topic": LaunchConfiguration(f"{namespace}_command_topic"),
        "status_topic": LaunchConfiguration(f"{namespace}_status_topic"),
        "joint_state_topic": f"/{namespace}/joint_states",
        "can_interface": LaunchConfiguration(f"{namespace}_can_interface"),
        "speed_percent": typed_config("speed_percent", int),
    }
    params.update({name: typed_config(name, bool) for name in bool_params})
    params.update({name: typed_config(name, float) for name in float_params})

    return Node(
        package="neroarm_control",
        executable="neroarm_driver",
        name=f"{namespace}_neroarm_driver",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_real_driver")),
        parameters=[params],
    )


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare("demo2"),
        "config",
        "dual_nero.rviz",
    ])
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in ARG_DEFAULTS.items()
        ],
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        real_driver("left"),
        real_driver("right"),
        Node(
            package="demo2",
            executable="arm_mediapipe3d.py",
            name="arm_mediapipe3d",
            output="screen",
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration("start_pose"),
                "'.lower() in ('true', '1', 'yes', 'on') and '",
                LaunchConfiguration("pose_backend"),
                "' == 'mediapipe_3d'",
            ])),
            parameters=[{
                "camera_id": typed_config("camera_id", int),
                "camera_width": typed_config("camera_width", int),
                "camera_height": typed_config("camera_height", int),
                "camera_fps": typed_config("camera_fps", float),
                "camera_fourcc": LaunchConfiguration("camera_fourcc"),
                "output_topic": LaunchConfiguration("pose_3d_topic"),
                "model_complexity": typed_config(
                    "mediapipe_model_complexity", int
                ),
                "show_gui": typed_config("show_gui", bool),
            }],
        ),
        Node(
            package="demo2",
            executable="arm_yolo26s_v2.py",
            name="arm_yolo26s_v2",
            output="screen",
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration("start_pose"),
                "'.lower() in ('true', '1', 'yes', 'on') and '",
                LaunchConfiguration("start_yolo"),
                "'.lower() in ('true', '1', 'yes', 'on') and '",
                LaunchConfiguration("pose_backend"),
                "' == 'yolo_2d'",
            ])),
            parameters=[{
                "model_path": LaunchConfiguration("yolo_model_path"),
                "camera_id": typed_config("camera_id", int),
                "camera_width": typed_config("camera_width", int),
                "camera_height": typed_config("camera_height", int),
                "camera_fps": typed_config("camera_fps", float),
                "camera_fourcc": LaunchConfiguration("camera_fourcc"),
                "output_topic": "/arm_pose_v2",
                "image_points_topic": LaunchConfiguration("image_points_topic"),
                "show_gui": typed_config("show_gui", bool),
                "center_roi_enabled": typed_config("center_roi_enabled", bool),
                "center_roi_fraction": typed_config("center_roi_fraction", float),
                "require_all_arm_points_in_roi": typed_config(
                    "require_all_arm_points_in_roi", bool
                ),
                "stable_pose_required": typed_config("stable_pose_required", bool),
                "stable_pose_duration": typed_config("stable_pose_duration", float),
                "pose_read_duration": typed_config("pose_read_duration", float),
                "stable_motion_threshold_px": typed_config(
                    "stable_motion_threshold_px", float
                ),
                "stable_range_threshold_px": typed_config(
                    "stable_range_threshold_px", float
                ),
                "show_initial_pose_guide": typed_config("show_initial_pose_guide", bool),
            }],
        ),
        Node(
            package="demo2",
            executable="position_to_angle_v2.py",
            name="position_to_angle_v2",
            output="screen",
            parameters=[{
                "input_mode": LaunchConfiguration("pose_backend"),
                "pose_3d_topic": LaunchConfiguration("pose_3d_topic"),
                "image_points_topic": LaunchConfiguration("image_points_topic"),
                "left_joint_state_topic": "/left/joint_states",
                "right_joint_state_topic": "/right/joint_states",
                "publish_joint_states_enabled": ParameterValue(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("start_real_driver"),
                        "'.lower() not in ('true', '1', 'yes', 'on')",
                    ]),
                    value_type=bool,
                ),
                "command_output_enabled": typed_config("command_output_enabled", bool),
                "left_command_topic": LaunchConfiguration("left_command_topic"),
                "right_command_topic": LaunchConfiguration("right_command_topic"),
                "lock_j1_enabled": typed_config("lock_j1_enabled", bool),
                "lock_j2_enabled": typed_config("lock_j2_enabled", bool),
                "lock_j5_j6_j7_enabled": typed_config("lock_j5_j6_j7_enabled", bool),
                "lock_j1_deg": 0.0,
                "lock_j2_deg": 0.0,
                "j1_j2_only": typed_config("j1_j2_only", bool),
                "yolo_calibration_duration": typed_config(
                    "yolo_calibration_duration", float
                ),
                "yolo_still_threshold_px": typed_config("yolo_still_threshold_px", float),
                "yolo_j1_home_deg": typed_config("yolo_j1_home_deg", float),
                "yolo_j2_home_deg": typed_config("yolo_j2_home_deg", float),
                "yolo_j1_forward_max_deg": typed_config(
                    "yolo_j1_forward_max_deg", float
                ),
                "yolo_j1_forward_full_ratio": typed_config(
                    "yolo_j1_forward_full_ratio", float
                ),
                "yolo_j2_gain": typed_config("yolo_j2_gain", float),
                "yolo_j4_gain": typed_config("yolo_j4_gain", float),
                "yolo_left_j2_sign": typed_config("yolo_left_j2_sign", float),
                "yolo_right_j2_sign": typed_config("yolo_right_j2_sign", float),
                "joint_smoothing_tau": typed_config(
                    "joint_smoothing_tau", float
                ),
                "joint_deadband_deg": typed_config(
                    "joint_deadband_deg", float
                ),
                "pose_3d_calibration_duration": typed_config(
                    "pose_3d_calibration_duration", float
                ),
                "pose_3d_still_threshold_deg": typed_config(
                    "pose_3d_still_threshold_deg", float
                ),
                "retarget_j1_gain": typed_config(
                    "retarget_j1_gain", float
                ),
                "retarget_j2_gain": typed_config(
                    "retarget_j2_gain", float
                ),
                "retarget_j4_gain": typed_config(
                    "retarget_j4_gain", float
                ),
                "pose_timeout_sec": typed_config("pose_timeout_sec", float),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_yolo_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
