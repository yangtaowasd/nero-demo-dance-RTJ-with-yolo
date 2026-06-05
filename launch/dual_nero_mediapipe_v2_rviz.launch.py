from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_description():
    model = PathJoinSubstitution([
        FindPackageShare("demo2v2"),
        "urdf",
        "nero_description.urdf",
    ])
    return {
        "robot_description": ParameterValue(Command(["xacro ", model]), value_type=str)
    }


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
            parameters=[robot_description(), {"frame_prefix": prefix}],
            remappings=[("joint_states", f"/{namespace}/joint_states")],
            output="screen",
        ),
    ]


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare("demo2v2"),
        "config",
        "dual_nero.rviz",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("pose_backend", default_value="mediapipe"),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value="/home/yang/demo_ws/src/demo2/model/yolo26s-pose.pt",
        ),
        DeclareLaunchArgument("camera_id", default_value="0"),
        DeclareLaunchArgument("camera_width", default_value="1280"),
        DeclareLaunchArgument("camera_height", default_value="720"),
        DeclareLaunchArgument("camera_fps", default_value="30.0"),
        DeclareLaunchArgument("camera_fourcc", default_value="MJPG"),
        DeclareLaunchArgument("show_gui", default_value="true"),
        DeclareLaunchArgument("use_realsense", default_value="false"),
        DeclareLaunchArgument("center_roi_enabled", default_value="true"),
        DeclareLaunchArgument("center_roi_fraction", default_value="0.80"),
        DeclareLaunchArgument("require_all_arm_points_in_roi", default_value="true"),
        DeclareLaunchArgument("image_points_topic", default_value="/arm_pose_v2/image_points"),
        DeclareLaunchArgument("yolo_calibration_duration", default_value="2.0"),
        DeclareLaunchArgument("yolo_still_threshold_px", default_value="180.0"),
        DeclareLaunchArgument("yolo_j1_home_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j2_home_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j3_home_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j4_home_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j1_forward_max_deg", default_value="70.0"),
        DeclareLaunchArgument("yolo_j1_forward_full_ratio", default_value="0.45"),
        DeclareLaunchArgument("yolo_j3_down_min_deg", default_value="-90.0"),
        DeclareLaunchArgument("yolo_j3_down_max_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j3_up_min_deg", default_value="-155.0"),
        DeclareLaunchArgument("yolo_j3_up_max_deg", default_value="-90.0"),
        DeclareLaunchArgument("yolo_j4_work_min_deg", default_value="0.0"),
        DeclareLaunchArgument("yolo_j4_work_max_deg", default_value="110.0"),
        DeclareLaunchArgument("yolo_j2_gain", default_value="1.0"),
        DeclareLaunchArgument("yolo_j3_gain", default_value="1.0"),
        DeclareLaunchArgument("yolo_j4_gain", default_value="1.0"),
        DeclareLaunchArgument("yolo_left_j2_sign", default_value="1.0"),
        DeclareLaunchArgument("yolo_right_j2_sign", default_value="-1.0"),
        DeclareLaunchArgument("yolo_left_j3_sign", default_value="1.0"),
        DeclareLaunchArgument("yolo_right_j3_sign", default_value="-1.0"),
        DeclareLaunchArgument("stable_pose_required", default_value="true"),
        DeclareLaunchArgument("stable_pose_duration", default_value="1.0"),
        DeclareLaunchArgument("pose_read_duration", default_value="2.0"),
        DeclareLaunchArgument("stable_motion_threshold_px", default_value="180.0"),
        DeclareLaunchArgument("stable_range_threshold_px", default_value="180.0"),
        DeclareLaunchArgument("show_initial_pose_guide", default_value="true"),
        DeclareLaunchArgument("lock_j1_enabled", default_value="false"),
        DeclareLaunchArgument("lock_j2_enabled", default_value="false"),
        DeclareLaunchArgument("lock_j5_j6_j7_enabled", default_value="true"),
        DeclareLaunchArgument("j1_j2_only", default_value="false"),
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="demo2v2",
            executable="arm_mediapipe_v2",
            name="arm_mediapipe_v2",
            output="screen",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("pose_backend"), "' == 'mediapipe'"
            ])),
            parameters=[{
                "camera_id": LaunchConfiguration("camera_id"),
                "camera_width": ParameterValue(
                    LaunchConfiguration("camera_width"),
                    value_type=int,
                ),
                "camera_height": ParameterValue(
                    LaunchConfiguration("camera_height"),
                    value_type=int,
                ),
                "camera_fps": ParameterValue(
                    LaunchConfiguration("camera_fps"),
                    value_type=float,
                ),
                "camera_fourcc": LaunchConfiguration("camera_fourcc"),
                "use_realsense": ParameterValue(
                    LaunchConfiguration("use_realsense"),
                    value_type=bool,
                ),
                "use_camera_topic": False,
                "output_topic": "/arm_pose_v2",
                "image_points_topic": LaunchConfiguration("image_points_topic"),
                "show_gui": ParameterValue(LaunchConfiguration("show_gui"), value_type=bool),
                "show_coords": True,
                "model_complexity": 1,
                "depth_sample_radius": 2,
                "center_roi_enabled": ParameterValue(
                    LaunchConfiguration("center_roi_enabled"),
                    value_type=bool,
                ),
                "center_roi_fraction": ParameterValue(
                    LaunchConfiguration("center_roi_fraction"),
                    value_type=float,
                ),
                "require_all_arm_points_in_roi": ParameterValue(
                    LaunchConfiguration("require_all_arm_points_in_roi"),
                    value_type=bool,
                ),
            }],
        ),
        Node(
            package="demo2v2",
            executable="arm_yolo26s_v2",
            name="arm_yolo26s_v2",
            output="screen",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("pose_backend"), "' == 'yolo26s'"
            ])),
            parameters=[{
                "model_path": LaunchConfiguration("yolo_model_path"),
                "camera_id": LaunchConfiguration("camera_id"),
                "camera_width": ParameterValue(
                    LaunchConfiguration("camera_width"),
                    value_type=int,
                ),
                "camera_height": ParameterValue(
                    LaunchConfiguration("camera_height"),
                    value_type=int,
                ),
                "camera_fps": ParameterValue(
                    LaunchConfiguration("camera_fps"),
                    value_type=float,
                ),
                "camera_fourcc": LaunchConfiguration("camera_fourcc"),
                "output_topic": "/arm_pose_v2",
                "image_points_topic": LaunchConfiguration("image_points_topic"),
                "show_gui": ParameterValue(LaunchConfiguration("show_gui"), value_type=bool),
                "center_roi_enabled": ParameterValue(
                    LaunchConfiguration("center_roi_enabled"),
                    value_type=bool,
                ),
                "center_roi_fraction": ParameterValue(
                    LaunchConfiguration("center_roi_fraction"),
                    value_type=float,
                ),
                "require_all_arm_points_in_roi": ParameterValue(
                    LaunchConfiguration("require_all_arm_points_in_roi"),
                    value_type=bool,
                ),
                "stable_pose_required": ParameterValue(
                    LaunchConfiguration("stable_pose_required"),
                    value_type=bool,
                ),
                "stable_pose_duration": ParameterValue(
                    LaunchConfiguration("stable_pose_duration"),
                    value_type=float,
                ),
                "pose_read_duration": ParameterValue(
                    LaunchConfiguration("pose_read_duration"),
                    value_type=float,
                ),
                "stable_motion_threshold_px": ParameterValue(
                    LaunchConfiguration("stable_motion_threshold_px"),
                    value_type=float,
                ),
                "stable_range_threshold_px": ParameterValue(
                    LaunchConfiguration("stable_range_threshold_px"),
                    value_type=float,
                ),
                "show_initial_pose_guide": ParameterValue(
                    LaunchConfiguration("show_initial_pose_guide"),
                    value_type=bool,
                ),
            }],
        ),
        Node(
            package="demo2v2",
            executable="position_to_angle_v2",
            name="position_to_angle_v2",
            output="screen",
            parameters=[{
                "input_topic": "/arm_pose_v2",
                "image_points_topic": LaunchConfiguration("image_points_topic"),
                "left_joint_state_topic": "/left/joint_states",
                "right_joint_state_topic": "/right/joint_states",
                "solve_mode": ParameterValue(
                    PythonExpression([
                        "'yolo_pixel' if '", LaunchConfiguration("pose_backend"),
                        "' == 'yolo26s' else 'forearm'"
                    ]),
                    value_type=str,
                ),
                "lock_j1_enabled": ParameterValue(
                    LaunchConfiguration("lock_j1_enabled"),
                    value_type=bool,
                ),
                "lock_j2_enabled": ParameterValue(
                    LaunchConfiguration("lock_j2_enabled"),
                    value_type=bool,
                ),
                "lock_j5_j6_j7_enabled": ParameterValue(
                    LaunchConfiguration("lock_j5_j6_j7_enabled"),
                    value_type=bool,
                ),
                "lock_j1_deg": 0.0,
                "lock_j2_deg": 0.0,
                "j1_j2_only": ParameterValue(
                    LaunchConfiguration("j1_j2_only"),
                    value_type=bool,
                ),
                "forearm_yaw_gain": 1.0,
                "forearm_pitch_gain": 1.6,
                "yolo_calibration_duration": ParameterValue(
                    LaunchConfiguration("yolo_calibration_duration"),
                    value_type=float,
                ),
                "yolo_still_threshold_px": ParameterValue(
                    LaunchConfiguration("yolo_still_threshold_px"),
                    value_type=float,
                ),
                "yolo_j1_home_deg": ParameterValue(
                    LaunchConfiguration("yolo_j1_home_deg"),
                    value_type=float,
                ),
                "yolo_j2_home_deg": ParameterValue(
                    LaunchConfiguration("yolo_j2_home_deg"),
                    value_type=float,
                ),
                "yolo_j3_home_deg": ParameterValue(
                    LaunchConfiguration("yolo_j3_home_deg"),
                    value_type=float,
                ),
                "yolo_j4_home_deg": ParameterValue(
                    LaunchConfiguration("yolo_j4_home_deg"),
                    value_type=float,
                ),
                "yolo_j1_forward_max_deg": ParameterValue(
                    LaunchConfiguration("yolo_j1_forward_max_deg"),
                    value_type=float,
                ),
                "yolo_j1_forward_full_ratio": ParameterValue(
                    LaunchConfiguration("yolo_j1_forward_full_ratio"),
                    value_type=float,
                ),
                "yolo_j3_down_min_deg": ParameterValue(
                    LaunchConfiguration("yolo_j3_down_min_deg"),
                    value_type=float,
                ),
                "yolo_j3_down_max_deg": ParameterValue(
                    LaunchConfiguration("yolo_j3_down_max_deg"),
                    value_type=float,
                ),
                "yolo_j3_up_min_deg": ParameterValue(
                    LaunchConfiguration("yolo_j3_up_min_deg"),
                    value_type=float,
                ),
                "yolo_j3_up_max_deg": ParameterValue(
                    LaunchConfiguration("yolo_j3_up_max_deg"),
                    value_type=float,
                ),
                "yolo_j4_work_min_deg": ParameterValue(
                    LaunchConfiguration("yolo_j4_work_min_deg"),
                    value_type=float,
                ),
                "yolo_j4_work_max_deg": ParameterValue(
                    LaunchConfiguration("yolo_j4_work_max_deg"),
                    value_type=float,
                ),
                "yolo_j2_gain": ParameterValue(
                    LaunchConfiguration("yolo_j2_gain"),
                    value_type=float,
                ),
                "yolo_j3_gain": ParameterValue(
                    LaunchConfiguration("yolo_j3_gain"),
                    value_type=float,
                ),
                "yolo_j4_gain": ParameterValue(
                    LaunchConfiguration("yolo_j4_gain"),
                    value_type=float,
                ),
                "yolo_left_j2_sign": ParameterValue(
                    LaunchConfiguration("yolo_left_j2_sign"),
                    value_type=float,
                ),
                "yolo_right_j2_sign": ParameterValue(
                    LaunchConfiguration("yolo_right_j2_sign"),
                    value_type=float,
                ),
                "yolo_left_j3_sign": ParameterValue(
                    LaunchConfiguration("yolo_left_j3_sign"),
                    value_type=float,
                ),
                "yolo_right_j3_sign": ParameterValue(
                    LaunchConfiguration("yolo_right_j3_sign"),
                    value_type=float,
                ),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_v2_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
