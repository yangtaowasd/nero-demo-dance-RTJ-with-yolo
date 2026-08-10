"""Start the C++ RealSense camera and standalone YOLO RGB-D recognition."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package_file, arguments):
    """Include a launch file with locally scoped arguments."""
    return GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("demo2"), "launch", package_file
                    ])
                ),
                launch_arguments=arguments.items(),
            )
        ],
    )


def camera_topic(suffix):
    """Build a topic from the configurable camera namespace and name."""
    return [
        "/",
        LaunchConfiguration("camera_namespace"),
        "/",
        LaunchConfiguration("camera_name"),
        suffix,
    ]


def generate_launch_description():
    """Build the camera plus YOLO recognition launch description."""
    defaults = {
        "camera_namespace": "usb_realsense",
        "camera_name": "d435i",
        "serial_no": "'108322074190'",
        "color_profile": "640x480x30",
        "depth_profile": "640x480x30",
        "initial_reset": "false",
        "spatial_filter_enabled": "false",
        "temporal_filter_enabled": "true",
        "model_path": PathJoinSubstitution([
            FindPackageShare("demo2"),
            "model",
            "yolo26n-pose.torchscript",
        ]),
        "inference_size": "384",
        "torch_threads": "1",
        "person_confidence": "0.30",
        "min_landmark_confidence": "0.35",
        "target_lock_enabled": "true",
        "target_lock_min_iou": "0.05",
        "target_lock_max_center_distance_ratio": "1.25",
        "target_lock_max_missed_frames": "8",
        "keypoint_smoothing_alpha": "0.55",
        "depth_window_radius": "4",
        "min_valid_depth_pixels": "4",
        "depth_cluster_tolerance_m": "0.08",
        "min_depth_m": "0.15",
        "max_depth_m": "8.0",
        "sync_tolerance_sec": "0.02",
        "sync_wait_sec": "0.02",
        "landmarks_topic": "/realsense/landmarks_3d",
        "debug_image_topic": "/realsense/arm_pose_debug",
        "publish_debug_image": "true",
        "show_gui": "true",
    }
    camera_names = (
        "camera_namespace",
        "camera_name",
        "serial_no",
        "color_profile",
        "depth_profile",
        "initial_reset",
        "spatial_filter_enabled",
        "temporal_filter_enabled",
    )
    detector_names = (
        "model_path",
        "inference_size",
        "torch_threads",
        "person_confidence",
        "min_landmark_confidence",
        "target_lock_enabled",
        "target_lock_min_iou",
        "target_lock_max_center_distance_ratio",
        "target_lock_max_missed_frames",
        "keypoint_smoothing_alpha",
        "depth_window_radius",
        "min_valid_depth_pixels",
        "depth_cluster_tolerance_m",
        "min_depth_m",
        "max_depth_m",
        "sync_tolerance_sec",
        "sync_wait_sec",
        "landmarks_topic",
        "debug_image_topic",
        "publish_debug_image",
        "show_gui",
    )
    detector_arguments = {
        "color_topic": camera_topic("/color/image_raw"),
        "aligned_depth_topic": camera_topic(
            "/aligned_depth_to_color/image_raw"
        ),
        "camera_info_topic": camera_topic("/color/camera_info"),
    }
    detector_arguments.update({
        name: LaunchConfiguration(name) for name in detector_names
    })
    return LaunchDescription([
        SetEnvironmentVariable(
            name="ROS_LOCALHOST_ONLY", value="1"
        ),
        *[
            DeclareLaunchArgument(name, default_value=value)
            for name, value in defaults.items()
        ],
        include(
            "realsense_camera.launch.py",
            {name: LaunchConfiguration(name) for name in camera_names},
        ),
        include("depth_pose_detector.launch.py", detector_arguments),
    ])
