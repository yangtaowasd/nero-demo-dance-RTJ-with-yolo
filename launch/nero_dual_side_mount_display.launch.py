from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
        FindPackageShare("demo2"),
        "config",
        "dual_nero.rviz",
    ])

    pose_guard = Node(
        package="demo2",
        executable="dual_joint_state_publisher.py",
        name="dual_joint_state_publisher",
        output="screen",
    )
    display_nodes = [
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_rviz",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("start_rviz", default_value="true"),
        pose_guard,
        RegisterEventHandler(
            OnProcessExit(
                target_action=pose_guard,
                on_exit=[
                    EmitEvent(event=Shutdown(
                        reason="dual display singleton guard stopped"
                    ))
                ],
            )
        ),
        TimerAction(period=1.0, actions=display_nodes),
    ])
