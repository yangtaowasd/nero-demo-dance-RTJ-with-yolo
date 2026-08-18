"""Compose the RealSense driver, recognition, and Nero control."""

from launch import LaunchDescription
from launch.actions import OpaqueFunction, SetEnvironmentVariable

from demo2.instance_guard import require_instance_available
from demo2.launch_common import (
    CAMERA_DEFAULTS,
    camera_topic,
    configured_arguments,
    controller_defaults,
    declare_arguments,
    detector_defaults,
    hardware_defaults,
    include_launch,
)


CAMERA_TOPIC_ARGUMENTS = (
    "color_topic",
    "aligned_depth_topic",
    "camera_info_topic",
)


def generate_launch_description():
    """Build the one-command RealSense arm pipeline."""
    pipeline = {
        **detector_defaults(),
        **controller_defaults(),
        **hardware_defaults(),
        "start_hardware": "false",
    }
    pipeline.update({
        "calibration_file": (
            "~/.ros/demo2/person_calibration_108322074190.json"
        ),
        "calibration_camera_id": "108322074190",
    })
    for name in CAMERA_TOPIC_ARGUMENTS:
        pipeline.pop(name)

    defaults = {**CAMERA_DEFAULTS, **pipeline}
    pipeline_arguments = configured_arguments(pipeline)
    pipeline_arguments.update({
        "color_topic": camera_topic("/color/image_raw"),
        "aligned_depth_topic": camera_topic(
            "/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": camera_topic("/color/camera_info"),
    })
    return LaunchDescription([
        SetEnvironmentVariable(name="ROS_LOCALHOST_ONLY", value="1"),
        *declare_arguments(defaults),
        OpaqueFunction(function=require_instance_available),
        include_launch(
            "realsense_camera.launch.py",
            configured_arguments(CAMERA_DEFAULTS),
        ),
        include_launch("depth_camera_arm.launch.py", pipeline_arguments),
    ])
