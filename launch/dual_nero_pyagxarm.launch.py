"""Connect two independent Nero arms through pyAgxArm and SocketCAN."""

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from demo2.launch_common import (
    HARDWARE_PARAMETER_TYPES,
    configured_parameters,
    declare_arguments,
    hardware_defaults,
)


def arm_driver(side, can_argument, firmware_argument, condition_argument):
    """Create one side-isolated pyAgxArm driver node."""
    values = configured_parameters(HARDWARE_PARAMETER_TYPES)
    parameters = {
        "side": side,
        "can_interface": LaunchConfiguration(can_argument),
        "firmware": LaunchConfiguration(firmware_argument),
        "urdf_file": values["urdf_file"],
        "command_topic": f"/{side}/neroarm/command_joints",
        "feedback_topic": f"/{side}/neroarm/measured_joint_states",
        "status_topic": f"/{side}/neroarm/hardware_status",
        "execute_motion": values["hardware_execute_motion"],
        "auto_enable": values["hardware_auto_enable"],
        "reset_emergency_stop_on_start": values[
            "hardware_reset_emergency_stop_on_start"
        ],
        "emergency_reset_timeout_sec": values[
            "hardware_emergency_reset_timeout_sec"
        ],
        "require_command_before_enable": values[
            "hardware_require_command_before_enable"
        ],
        "motion_start_delay_sec": values[
            "hardware_motion_start_delay_sec"
        ],
        "return_to_home_on_shutdown": values[
            "return_to_home_on_shutdown"
        ],
        "shutdown_return_timeout_sec": values[
            "shutdown_return_timeout_sec"
        ],
        "shutdown_position_tolerance_deg": values[
            "shutdown_position_tolerance_deg"
        ],
        "disable_on_shutdown": values["disable_on_shutdown"],
        "feedback_rate_hz": values["hardware_rate_hz"],
        "command_timeout_sec": values["hardware_command_timeout_sec"],
        "feedback_timeout_sec": values["hardware_feedback_timeout_sec"],
        "connect_timeout_sec": values["hardware_connect_timeout_sec"],
        "probe_reconnect_delay_sec": values[
            "hardware_probe_reconnect_delay_sec"
        ],
        "reconnect_interval_sec": values[
            "hardware_reconnect_interval_sec"
        ],
        "enable_timeout_sec": values["hardware_enable_timeout_sec"],
        "max_command_speed_deg_sec": values[
            "hardware_max_command_speed_deg_sec"
        ],
        "speed_percent": values["hardware_speed_percent"],
        "exit_if_parent_changes": values["exit_if_parent_changes"],
    }

    return Node(
        package="demo2",
        executable="pyagxarm_driver.py",
        namespace=side,
        name="nero_pyagxarm_driver",
        condition=IfCondition(LaunchConfiguration(condition_argument)),
        parameters=[parameters],
        output="screen",
        sigterm_timeout=LaunchConfiguration(
            "hardware_shutdown_sigterm_timeout_sec"
        ),
    )


def generate_launch_description():
    """Build the read-only-by-default dual-Nero hardware bridge."""
    defaults = hardware_defaults()
    return LaunchDescription([
        *declare_arguments(defaults),
        arm_driver(
            "left",
            "left_can_interface",
            "left_firmware",
            "left_hardware_enabled",
        ),
        arm_driver(
            "right",
            "right_can_interface",
            "right_firmware",
            "right_hardware_enabled",
        ),
    ])
