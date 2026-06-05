from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_description():
    model = PathJoinSubstitution([
        FindPackageShare("demo2v2"),
        "urdf",
        "nero_description.urdf",
    ])
    return {
        "robot_description": ParameterValue(
            Command(["xacro ", model]),
            value_type=str,
        )
    }


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
            parameters=[
                robot_description(),
                {"frame_prefix": prefix},
            ],
            remappings=[("joint_states", f"/{namespace}/joint_states")],
            output="screen",
        ),
    ]


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare("demo2v2"),
        "config",
        "dual_nero.rviz",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("camera_id", default_value="0"),
        DeclareLaunchArgument("show_gui", default_value="true"),
        DeclareLaunchArgument("mediapipe_world_scale", default_value="1.0"),
        DeclareLaunchArgument("mediapipe_z_sign", default_value="-1.0"),
        DeclareLaunchArgument("normalize_limb_lengths", default_value="true"),
        DeclareLaunchArgument("upper_arm_length", default_value="0.30"),
        DeclareLaunchArgument("forearm_length", default_value="0.26"),
        DeclareLaunchArgument("center_roi_enabled", default_value="true"),
        DeclareLaunchArgument("center_roi_fraction", default_value="0.67"),
        DeclareLaunchArgument("motion_scale", default_value="1.0"),
        DeclareLaunchArgument("max_joint_step", default_value="0.08"),
        DeclareLaunchArgument("debug_nero_points", default_value="true"),
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="demo2v2",
            executable="arm_mediapipe",
            name="arm_mediapipe_node",
            output="screen",
            parameters=[{
                "camera_id": LaunchConfiguration("camera_id"),
                "use_camera_topic": False,
                "arm_pose": "/arm_pose",
                "show_gui": LaunchConfiguration("show_gui"),
                "model_complexity": 1,
                "world_scale": ParameterValue(
                    LaunchConfiguration("mediapipe_world_scale"),
                    value_type=float,
                ),
                "z_sign": ParameterValue(
                    LaunchConfiguration("mediapipe_z_sign"),
                    value_type=float,
                ),
                "normalize_limb_lengths": ParameterValue(
                    LaunchConfiguration("normalize_limb_lengths"),
                    value_type=bool,
                ),
                "upper_arm_length": ParameterValue(
                    LaunchConfiguration("upper_arm_length"),
                    value_type=float,
                ),
                "forearm_length": ParameterValue(
                    LaunchConfiguration("forearm_length"),
                    value_type=float,
                ),
                "center_roi_enabled": ParameterValue(
                    LaunchConfiguration("center_roi_enabled"),
                    value_type=bool,
                ),
                "center_roi_fraction": ParameterValue(
                    LaunchConfiguration("center_roi_fraction"),
                    value_type=float,
                ),
            }],
        ),
        Node(
            package="demo2v2",
            executable="position_to_angle",
            name="position_to_angle",
            output="screen",
            parameters=[{
                "backend": "rviz_joint_state",
                "input_topic": "/arm_pose",
                "motion_scale": LaunchConfiguration("motion_scale"),
                "max_joint_step": LaunchConfiguration("max_joint_step"),
                "debug_nero_points": ParameterValue(
                    LaunchConfiguration("debug_nero_points"),
                    value_type=bool,
                ),
                "left_joint_state_topic": "/left/joint_states",
                "right_joint_state_topic": "/right/joint_states",
                "left_camera_to_robot_matrix": [
                    0.0, 0.0, -1.0,
                    -1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                ],
                "right_camera_to_robot_matrix": [
                    0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                ],
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
