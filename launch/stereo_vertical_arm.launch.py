"""RViz dry-run launch for the independent vertical-stereo follower."""

import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_description():
    model = PathJoinSubstitution(
        [FindPackageShare("demo2"), "urdf", "nero_description.urdf"]
    )
    return {
        "robot_description": ParameterValue(Command(["xacro ", model]), value_type=str)
    }


def state_publishers(namespace, y_offset, pitch, yaw):
    prefix = f"{namespace}/"
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{namespace}_stereo_mount_tf",
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
        "lower_camera_id": "0",
        "upper_camera_id": "2",
        "camera_width": "640",
        "camera_height": "480",
        "camera_fps": "30.0",
        "calibration_file": PathJoinSubstitution(
            [share, "config", "stereo_vertical_example.yaml"]
        ),
        "urdf_file": PathJoinSubstitution([share, "urdf", "nero_description.urdf"]),
        "model_complexity": "0",
        "show_gui": "true",
        "command_output_enabled": "false",
    }
    follower_arguments = [sys.executable, "-m", "demo2.stereo_arm_follower", "--ros-args"]
    for name in defaults:
        follower_arguments.extend(["-p", ros_parameter(name)])

    rviz_config = PathJoinSubstitution([share, "config", "dual_nero.rviz"])
    return LaunchDescription(
        [
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
                output="screen",
            ),
        ]
    )
