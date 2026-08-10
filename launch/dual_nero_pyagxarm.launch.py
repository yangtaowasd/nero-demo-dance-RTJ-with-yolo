"""Connect two independent Nero arms through pyAgxArm and SocketCAN."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def typed(name, value_type):
    """Return a typed launch parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def arm_driver(side, can_argument, firmware_argument, condition_argument):
    """Create one side-isolated pyAgxArm driver node."""
    prefix = f"/{side}/neroarm"
    return Node(
        package="demo2",
        executable="pyagxarm_driver.py",
        namespace=side,
        name="nero_pyagxarm_driver",
        condition=IfCondition(LaunchConfiguration(condition_argument)),
        parameters=[{
            "side": side,
            "can_interface": LaunchConfiguration(can_argument),
            "firmware": LaunchConfiguration(firmware_argument),
            "urdf_file": LaunchConfiguration("urdf_file"),
            "command_topic": f"{prefix}/command_joints",
            "feedback_topic": f"{prefix}/measured_joint_states",
            "status_topic": f"{prefix}/hardware_status",
            "execute_motion": typed("hardware_execute_motion", bool),
            "auto_enable": typed("hardware_auto_enable", bool),
            "require_command_before_enable": typed(
                "hardware_require_command_before_enable", bool
            ),
            "motion_start_delay_sec": typed(
                "hardware_motion_start_delay_sec", float
            ),
            "return_to_home_on_shutdown": typed(
                "return_to_home_on_shutdown", bool
            ),
            "shutdown_return_timeout_sec": typed(
                "shutdown_return_timeout_sec", float
            ),
            "shutdown_position_tolerance_deg": typed(
                "shutdown_position_tolerance_deg", float
            ),
            "disable_on_shutdown": typed("disable_on_shutdown", bool),
            "feedback_rate_hz": typed("hardware_rate_hz", float),
            "command_timeout_sec": typed(
                "hardware_command_timeout_sec", float
            ),
            "feedback_timeout_sec": typed(
                "hardware_feedback_timeout_sec", float
            ),
            "connect_timeout_sec": typed(
                "hardware_connect_timeout_sec", float
            ),
            "reconnect_interval_sec": typed(
                "hardware_reconnect_interval_sec", float
            ),
            "enable_timeout_sec": typed("hardware_enable_timeout_sec", float),
            "max_command_speed_deg_sec": typed(
                "hardware_max_command_speed_deg_sec", float
            ),
            "speed_percent": typed("hardware_speed_percent", int),
            "exit_if_parent_changes": typed(
                "exit_if_parent_changes", bool
            ),
        }],
        output="screen",
        sigterm_timeout=LaunchConfiguration(
            "hardware_shutdown_sigterm_timeout_sec"
        ),
    )


def generate_launch_description():
    """Build the read-only-by-default dual-Nero hardware bridge."""
    defaults = {
        "left_hardware_enabled": "true",
        "right_hardware_enabled": "true",
        "left_can_interface": "can1",
        "right_can_interface": "can0",
        "left_firmware": "v111",
        "right_firmware": "v111",
        "urdf_file": PathJoinSubstitution([
            FindPackageShare("demo2"),
            "urdf",
            "nero_description.urdf",
        ]),
        "hardware_execute_motion": "false",
        "hardware_auto_enable": "true",
        "hardware_require_command_before_enable": "false",
        "hardware_motion_start_delay_sec": "10.0",
        "return_to_home_on_shutdown": "true",
        "shutdown_return_timeout_sec": "8.0",
        "shutdown_position_tolerance_deg": "1.5",
        "hardware_shutdown_sigterm_timeout_sec": "12.0",
        "disable_on_shutdown": "false",
        "hardware_rate_hz": "20.0",
        "hardware_command_timeout_sec": "0.35",
        "hardware_feedback_timeout_sec": "0.50",
        "hardware_connect_timeout_sec": "2.0",
        "hardware_reconnect_interval_sec": "2.0",
        "hardware_enable_timeout_sec": "5.0",
        "hardware_max_command_speed_deg_sec": "30.0",
        "hardware_speed_percent": "20",
        "exit_if_parent_changes": "true",
    }
    return LaunchDescription([
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
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
