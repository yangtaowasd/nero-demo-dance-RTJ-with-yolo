"""Compose RGB-D recognition and Nero control for any aligned-depth camera."""

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from demo2.instance_guard import require_instance_available
from demo2.launch_common import (
    DETECTOR_PARAMETER_TYPES,
    HARDWARE_PARAMETER_TYPES,
    configured_arguments,
    controller_defaults,
    declare_arguments,
    detector_defaults,
    hardware_defaults,
    include_launch,
)


def generate_launch_description():
    """Build the complete camera-agnostic RGB-D arm pipeline."""
    detector = detector_defaults()
    controller = controller_defaults()
    hardware = hardware_defaults()
    defaults = {
        **detector,
        **controller,
        **hardware,
        "start_hardware": "false",
    }
    return LaunchDescription([
        *declare_arguments(defaults),
        OpaqueFunction(function=require_instance_available),
        include_launch(
            "depth_pose_detector.launch.py",
            configured_arguments(DETECTOR_PARAMETER_TYPES),
        ),
        include_launch(
            "depth_arm_control.launch.py",
            configured_arguments(controller),
        ),
        include_launch(
            "dual_nero_pyagxarm.launch.py",
            configured_arguments(HARDWARE_PARAMETER_TYPES),
            condition=IfCondition(LaunchConfiguration("start_hardware")),
        ),
    ])
