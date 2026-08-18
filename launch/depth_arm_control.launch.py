"""Start only Nero IK/control for an external RGB-D landmark topic."""

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
    CONTROLLER_PARAMETER_TYPES,
    arm_state_publishers,
    configured_parameters,
    controller_defaults,
    declare_arguments,
    package_path,
)


def generate_launch_description():
    """Build the robot-control-only launch description."""
    defaults = controller_defaults()
    controller = Node(
        package="demo2",
        executable="depth_arm_controller.py",
        name="depth_arm_controller",
        parameters=[configured_parameters(CONTROLLER_PARAMETER_TYPES)],
        output="screen",
    )
    display_nodes = [
        *arm_state_publishers(
            "left", 0.35, -1.5707963, -1.5707963, "depth_mount_tf"
        ),
        *arm_state_publishers(
            "right", -0.35, -1.5707963, 1.5707963, "depth_mount_tf"
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", package_path("config", "dual_nero.rviz")],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ]
    return LaunchDescription([
        *declare_arguments(defaults),
        OpaqueFunction(function=require_instance_available),
        controller,
        RegisterEventHandler(
            OnProcessExit(
                target_action=controller,
                on_exit=[
                    EmitEvent(event=Shutdown(
                        reason="depth arm controller guard stopped"
                    ))
                ],
            )
        ),
        TimerAction(period=1.0, actions=display_nodes),
    ])
