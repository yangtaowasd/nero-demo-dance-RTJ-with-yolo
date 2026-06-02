from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
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
        "robot_description": ParameterValue(
            Command(["xacro ", model]),
            value_type=str,
        )
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
        FindPackageShare("demo2v2"),
        "config",
        "dual_nero.rviz",
    ])
    model_path = PathJoinSubstitution([
        FindPackageShare("demo2v2"),
        "model",
        "yolo26s-pose.pt",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("camera_id", default_value="0"),
        DeclareLaunchArgument("show_gui", default_value="true"),
        DeclareLaunchArgument("motion_scale", default_value="1.0"),
        DeclareLaunchArgument("max_joint_step", default_value="0.08"),
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="demo2v2",
            executable="body_yolo",
            name="yolo_body_pose_node",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "camera_id": LaunchConfiguration("camera_id"),
                "use_camera_topic": False,
                "body_pose": "/body_pose",
                "show_gui": LaunchConfiguration("show_gui"),
                "world_scale": 1.0,
                "z_sign": -1.0,
                "arm_depth_mode": "near",
            }],
        ),
        Node(
            package="demo2v2",
            executable="position_to_angle",
            name="position_to_angle",
            output="screen",
            parameters=[{
                "backend": "rviz_joint_state",
                "motion_scale": LaunchConfiguration("motion_scale"),
                "max_joint_step": LaunchConfiguration("max_joint_step"),
                "left_joint_state_topic": "/left/joint_states",
                "right_joint_state_topic": "/right/joint_states",
                "left_camera_to_robot_matrix": [
                    0.0, 0.0, -1.0,
                    -1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                ],
                "right_camera_to_robot_matrix": [
                    0.0, 0.0, -1.0,
                    1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                ],
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
