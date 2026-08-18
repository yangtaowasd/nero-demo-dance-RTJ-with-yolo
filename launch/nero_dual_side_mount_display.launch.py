"""Display two side-mounted Nero arms in RViz."""

from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from demo2.instance_guard import require_instance_available
from demo2.launch_common import (
    arm_state_publishers,
    declare_arguments,
    package_path,
)


def generate_launch_description():
    """Build the static dual-arm display launch description."""
    pose_guard = Node(
        package="demo2",
        executable="dual_joint_state_publisher.py",
        name="dual_joint_state_publisher",
        output="screen",
    )
    display_nodes = [
        *arm_state_publishers(
            "left", 0.35, -1.5707963, -1.5707963, "mount_tf"
        ),
        *arm_state_publishers(
            "right", -0.35, -1.5707963, 1.5707963, "mount_tf"
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_rviz",
            arguments=["-d", package_path("config", "dual_nero.rviz")],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ]
    return LaunchDescription([
        *declare_arguments({"start_rviz": "true"}),
        OpaqueFunction(function=require_instance_available),
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
