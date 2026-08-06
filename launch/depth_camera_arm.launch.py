"""RViz dry-run launch for the generic aligned-depth arm follower."""

import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_description():
    model = PathJoinSubstitution(
        [FindPackageShare("demo2"), "urdf", "nero_description.urdf"]
    )
    return {
        "robot_description": ParameterValue(
            Command(["xacro ", model]), value_type=str
        )
    }


def state_publishers(namespace, y_offset, pitch, yaw):
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


def ros_parameter(name):
    return [f"{name}:=", LaunchConfiguration(name)]


def generate_launch_description():
    share = FindPackageShare("demo2")
    defaults = {
        "color_topic": "/camera/camera/color/image_raw",
        "aligned_depth_topic": (
            "/camera/camera/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": "/camera/camera/color/camera_info",
        "urdf_file": PathJoinSubstitution(
            [share, "urdf", "nero_description.urdf"]
        ),
        "model_complexity": "0",
        "show_gui": "true",
        "command_output_enabled": "false",
        "start_rviz": "true",
    }
    follower_parameters = tuple(
        name for name in defaults if name != "start_rviz"
    )
    follower_arguments = [
        sys.executable,
        "-m",
        "demo2.depth_arm_follower",
        "--ros-args",
    ]
    for name in follower_parameters:
        follower_arguments.extend(["-p", ros_parameter(name)])

    rviz_config = PathJoinSubstitution([share, "config", "dual_nero.rviz"])
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        *state_publishers("left", 0.35, -1.5707963, -1.5707963),
        *state_publishers("right", -0.35, -1.5707963, 1.5707963),
        ExecuteProcess(cmd=follower_arguments, output="screen"),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ])
