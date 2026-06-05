from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
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

    return LaunchDescription([
        *state_publisher("left", 0.35, -1.5707963, -1.5707963),
        *state_publisher("right", -0.35, -1.5707963, 1.5707963),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="left_joint_state_publisher_gui",
            parameters=[
                robot_description(),
                {"zeros": {"joint2": 1.5707963}},
            ],
            remappings=[
                ("joint_states", "/left/joint_states"),
                ("robot_description", "/left/robot_description"),
            ],
            output="screen",
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="right_joint_state_publisher_gui",
            parameters=[
                robot_description(),
                {"zeros": {"joint2": 1.5707963}},
            ],
            remappings=[
                ("joint_states", "/right/joint_states"),
                ("robot_description", "/right/robot_description"),
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual_nero_rviz",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
