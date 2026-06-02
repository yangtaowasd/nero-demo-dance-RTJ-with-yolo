#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


def read_bool(node, name):
    value = node.get_parameter(name).value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def read_matrix(node, name):
    values = list(node.get_parameter(name).value)
    if len(values) != 9:
        node.get_logger().warning(f"{name} must have 9 values, using identity")
        return np.eye(3)
    return np.asarray(values, dtype=float).reshape(3, 3)


def default_camera_to_robot_matrix(side):
    if side == "right":
        return [
            0.0, 0.0, -1.0,
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
        ]
    return [
        0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
    ]


def read_quat(node, name):
    values = list(node.get_parameter(name).value)
    if len(values) != 4:
        node.get_logger().warning(f"{name} must have 4 values, using identity quaternion")
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) for v in values]


class KalmanPoint3D:
    def __init__(self, dt, sigma_a, sigma_pos, sigma_z, gate_threshold):
        self.dt = dt
        self.gate_threshold = gate_threshold
        self.x = np.zeros((6, 1))
        self.p = np.eye(6) * 0.1
        self.initialized = False

        self.f = np.eye(6)
        self.f[0, 3] = dt
        self.f[1, 4] = dt
        self.f[2, 5] = dt

        q = np.array([
            [dt**4 / 4, 0.0, 0.0, dt**3 / 2, 0.0, 0.0],
            [0.0, dt**4 / 4, 0.0, 0.0, dt**3 / 2, 0.0],
            [0.0, 0.0, dt**4 / 4, 0.0, 0.0, dt**3 / 2],
            [dt**3 / 2, 0.0, 0.0, dt**2, 0.0, 0.0],
            [0.0, dt**3 / 2, 0.0, 0.0, dt**2, 0.0],
            [0.0, 0.0, dt**3 / 2, 0.0, 0.0, dt**2],
        ])
        self.q = q * sigma_a**2

        self.h = np.zeros((3, 6))
        self.h[0, 0] = 1.0
        self.h[1, 1] = 1.0
        self.h[2, 2] = 1.0
        self.r = np.diag([sigma_pos**2, sigma_pos**2, sigma_z**2])

    def initialize(self, z):
        self.x[:3] = np.asarray(z, dtype=float).reshape(3, 1)
        self.x[3:] = 0.0
        self.initialized = True
        return self.position()

    def predict(self):
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.position()

    def update(self, z):
        if not self.initialized:
            return self.initialize(z), True

        self.predict()
        z = np.asarray(z, dtype=float).reshape(3, 1)
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return self.position(), False

        distance = np.sqrt(y.T @ s_inv @ y).item()
        if distance > self.gate_threshold:
            return self.position(), False

        k = self.p @ self.h.T @ s_inv
        self.x = self.x + k @ y
        self.p = (np.eye(6) - k @ self.h) @ self.p
        return self.position(), True

    def position(self):
        return self.x[:3].flatten()


class ArmPointFilterV2(Node):
    def __init__(self):
        super().__init__("arm_point_filter_v2")

        self.declare_parameter("input_topic", "/body_pose")
        self.declare_parameter("left_output_topic", "/left_arm")
        self.declare_parameter("right_output_topic", "/right_arm")
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("sigma_a", 0.8)
        self.declare_parameter("sigma_pos", 0.015)
        self.declare_parameter("sigma_z", 0.03)
        self.declare_parameter("kalman_gate_threshold", 8.0)
        self.declare_parameter("max_missing_frames", 8)
        self.declare_parameter("publish_predicted_points", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.left_output_topic = self.get_parameter("left_output_topic").value
        self.right_output_topic = self.get_parameter("right_output_topic").value
        self.dt = float(self.get_parameter("dt").value)
        self.sigma_a = float(self.get_parameter("sigma_a").value)
        self.sigma_pos = float(self.get_parameter("sigma_pos").value)
        self.sigma_z = float(self.get_parameter("sigma_z").value)
        self.kalman_gate_threshold = float(self.get_parameter("kalman_gate_threshold").value)
        self.max_missing_frames = int(self.get_parameter("max_missing_frames").value)
        self.publish_predicted_points = read_bool(self, "publish_predicted_points")

        self.num_keypoints = 17
        self.expected_len = 1 + self.num_keypoints * 3
        self.target_points = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
        }
        self.left_order = ["left_shoulder", "left_elbow", "left_wrist"]
        self.right_order = ["right_shoulder", "right_elbow", "right_wrist"]

        self.filters = {}
        self.missing_counts = {}
        for name in self.target_points:
            self.filters[name] = KalmanPoint3D(
                self.dt,
                self.sigma_a,
                self.sigma_pos,
                self.sigma_z,
                self.kalman_gate_threshold,
            )
            self.missing_counts[name] = self.max_missing_frames + 1

        self.create_subscription(Float32MultiArray, self.input_topic, self.callback, 10)
        self.left_pub = self.create_publisher(Float32MultiArray, self.left_output_topic, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, self.right_output_topic, 10)

        self.get_logger().info(
            f"arm_point_filter_v2 started: input={self.input_topic}, "
            f"left={self.left_output_topic}, right={self.right_output_topic}, "
            f"max_missing_frames={self.max_missing_frames}"
        )

    def get_keypoint_xyz(self, data, keypoint_id):
        base = 1 + keypoint_id * 3
        return [data[base], data[base + 1], data[base + 2]]

    def is_valid_xyz(self, xyz):
        return all(np.isfinite(v) for v in xyz)

    def use_prediction(self, name, already_predicted=False):
        point_filter = self.filters[name]
        if not point_filter.initialized or not self.publish_predicted_points:
            return None
        if not already_predicted:
            point_filter.predict()
        self.missing_counts[name] += 1
        if self.missing_counts[name] > self.max_missing_frames:
            return None
        return point_filter.position()

    def update_point(self, name, raw_pos):
        if not self.is_valid_xyz(raw_pos):
            self.get_logger().warning(f"{name} invalid xyz: {raw_pos}", throttle_duration_sec=1.0)
            return self.use_prediction(name)

        point, accepted = self.filters[name].update(raw_pos)
        if accepted:
            self.missing_counts[name] = 0
            return point

        self.get_logger().warning(f"{name} rejected by Kalman gate", throttle_duration_sec=1.0)
        return self.use_prediction(name, already_predicted=True)

    def arm_ready(self, points, order):
        for name in order:
            point = points.get(name)
            if point is None or not self.is_valid_xyz(point):
                return False
            if self.missing_counts[name] > self.max_missing_frames:
                return False
        return True

    def make_relative_msg(self, points, order):
        shoulder = np.asarray(points[order[0]], dtype=float)
        data = []
        for name in order:
            xyz = np.asarray(points[name], dtype=float) - shoulder
            data.extend([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        return Float32MultiArray(data=data)

    def callback(self, msg):
        data = list(msg.data)
        if len(data) < 1:
            self.get_logger().warning("empty body pose data", throttle_duration_sec=1.0)
            return
        if int(data[0]) <= 0:
            self.get_logger().warning("no person detected", throttle_duration_sec=1.0)
            return
        if len(data) != self.expected_len:
            self.get_logger().warning(
                f"body pose length error: {len(data)}, expected={self.expected_len}",
                throttle_duration_sec=1.0,
            )
            return

        points = {}
        for name, keypoint_id in self.target_points.items():
            updated = self.update_point(name, self.get_keypoint_xyz(data, keypoint_id))
            if updated is not None:
                points[name] = updated

        if self.arm_ready(points, self.left_order):
            self.left_pub.publish(self.make_relative_msg(points, self.left_order))
        else:
            self.get_logger().warning("left arm points are not ready", throttle_duration_sec=1.0)

        if self.arm_ready(points, self.right_order):
            self.right_pub.publish(self.make_relative_msg(points, self.right_order))
        else:
            self.get_logger().warning("right arm points are not ready", throttle_duration_sec=1.0)


class AgxArmFollowerV2(Node):
    def __init__(self):
        super().__init__("agx_arm_follower_v2")

        self.declare_parameter("backend", "agx_move_p")
        self.declare_parameter("left_input_topic", "/left_arm")
        self.declare_parameter("right_input_topic", "/right_arm")
        self.declare_parameter("left_tcp_feedback_topic", "/left/feedback/tcp_pose")
        self.declare_parameter("right_tcp_feedback_topic", "/right/feedback/tcp_pose")
        self.declare_parameter("left_move_p_topic", "/left/control/move_p")
        self.declare_parameter("right_move_p_topic", "/right/control/move_p")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("motion_scale", 0.7)
        self.declare_parameter("max_target_delta", 0.08)
        self.declare_parameter("min_command_interval", 0.1)
        self.declare_parameter("arm_points_are_relative", True)
        self.declare_parameter("use_current_orientation", True)
        self.declare_parameter("fixed_target_orientation", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("left_camera_to_robot_matrix", default_camera_to_robot_matrix("left"))
        self.declare_parameter("right_camera_to_robot_matrix", default_camera_to_robot_matrix("right"))
        self.declare_parameter("camera_to_robot_matrix", default_camera_to_robot_matrix("left"))

        self.backend = str(self.get_parameter("backend").value).strip().lower()
        if self.backend != "agx_move_p":
            self.get_logger().warning(
                f"position_to_angle_v2 ignores backend='{self.backend}'. "
                "It does not run a local or MoveIt IK solver; using AGX move_p instead."
            )
            self.backend = "agx_move_p"
        self.motion_scale = float(self.get_parameter("motion_scale").value)
        self.max_target_delta = float(self.get_parameter("max_target_delta").value)
        self.min_command_interval = float(self.get_parameter("min_command_interval").value)
        self.arm_points_are_relative = read_bool(self, "arm_points_are_relative")
        self.use_current_orientation = read_bool(self, "use_current_orientation")
        self.fixed_target_orientation = read_quat(self, "fixed_target_orientation")
        self.left_camera_to_robot_matrix = read_matrix(self, "left_camera_to_robot_matrix")
        self.right_camera_to_robot_matrix = read_matrix(self, "right_camera_to_robot_matrix")
        read_matrix(self, "camera_to_robot_matrix")

        self.left_tcp_feedback_topic = self.get_parameter("left_tcp_feedback_topic").value
        self.right_tcp_feedback_topic = self.get_parameter("right_tcp_feedback_topic").value
        self.left_move_p_topic = self.get_parameter("left_move_p_topic").value
        self.right_move_p_topic = self.get_parameter("right_move_p_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.home_tcp = {"left": None, "right": None}
        self.home_quat = {"left": None, "right": None}
        self.last_command_time = {"left": 0.0, "right": 0.0}

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("left_input_topic").value,
            lambda msg: self.arm_callback("left", msg),
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("right_input_topic").value,
            lambda msg: self.arm_callback("right", msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.left_tcp_feedback_topic,
            lambda msg: self.tcp_feedback_callback("left", msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.right_tcp_feedback_topic,
            lambda msg: self.tcp_feedback_callback("right", msg),
            10,
        )

        self.left_move_p_pub = self.create_publisher(PoseStamped, self.left_move_p_topic, 10)
        self.right_move_p_pub = self.create_publisher(PoseStamped, self.right_move_p_topic, 10)

        self.get_logger().warning(
            f"position_to_angle_v2 uses AGX move_p only: {self.left_move_p_topic}, {self.right_move_p_topic}. "
            "No local IK, no MoveIt IK. Keep agx control disabled until feedback is ready and targets look safe."
        )

    def side_values(self, side):
        if side == "left":
            return self.left_move_p_pub, self.left_tcp_feedback_topic
        return self.right_move_p_pub, self.right_tcp_feedback_topic

    def tcp_feedback_callback(self, side, msg):
        pos = msg.pose.position
        quat = msg.pose.orientation
        if self.home_tcp[side] is None:
            self.home_tcp[side] = np.asarray([pos.x, pos.y, pos.z], dtype=float)
            self.get_logger().info(f"{side} home tcp set from AGX feedback: {self.home_tcp[side].tolist()}")
        if self.home_quat[side] is None:
            self.home_quat[side] = [quat.x, quat.y, quat.z, quat.w]

    def rate_limited(self, side):
        now = time.monotonic()
        if now - self.last_command_time[side] < self.min_command_interval:
            return True
        self.last_command_time[side] = now
        return False

    def get_relative_points(self, shoulder, elbow, wrist):
        if self.arm_points_are_relative:
            return elbow, wrist
        return elbow - shoulder, wrist - shoulder

    def limit_delta(self, delta):
        if self.max_target_delta <= 0.0:
            return delta
        norm = np.linalg.norm(delta)
        if norm <= self.max_target_delta:
            return delta
        return delta / norm * self.max_target_delta

    def camera_matrix_for_side(self, side):
        if side == "left":
            return self.left_camera_to_robot_matrix
        return self.right_camera_to_robot_matrix

    def build_target_xyz(self, side, shoulder, elbow, wrist, home_tcp):
        _, wrist_rel = self.get_relative_points(shoulder, elbow, wrist)
        robot_delta = self.motion_scale * (self.camera_matrix_for_side(side) @ wrist_rel)
        return home_tcp + self.limit_delta(robot_delta)

    def make_pose_stamped(self, side, target_xyz):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(target_xyz[0])
        msg.pose.position.y = float(target_xyz[1])
        msg.pose.position.z = float(target_xyz[2])

        quat = self.home_quat[side] if self.use_current_orientation and self.home_quat[side] else self.fixed_target_orientation
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        return msg

    def publish_move_p(self, side, pose_msg):
        pose_pub, _ = self.side_values(side)
        pose_pub.publish(pose_msg)

    def arm_callback(self, side, msg):
        if len(msg.data) != 9:
            self.get_logger().warning(f"{side} arm data length error: {len(msg.data)}", throttle_duration_sec=1.0)
            return
        if self.rate_limited(side):
            return

        home_tcp = self.home_tcp[side]
        if home_tcp is None:
            _, tcp_topic = self.side_values(side)
            self.get_logger().warning(
                f"{side} AGX TCP feedback is not ready. Waiting for '{tcp_topic}'.",
                throttle_duration_sec=1.0,
            )
            return

        shoulder = np.asarray(msg.data[0:3], dtype=float)
        elbow = np.asarray(msg.data[3:6], dtype=float)
        wrist = np.asarray(msg.data[6:9], dtype=float)
        target_xyz = self.build_target_xyz(side, shoulder, elbow, wrist, home_tcp)
        pose_msg = self.make_pose_stamped(side, target_xyz)

        self.publish_move_p(side, pose_msg)


class RvizJointStateFollower(Node):
    def __init__(self):
        super().__init__("rviz_joint_state_follower")

        self.declare_parameter("left_input_topic", "/left_arm")
        self.declare_parameter("right_input_topic", "/right_arm")
        self.declare_parameter("left_joint_state_topic", "/left/joint_states")
        self.declare_parameter("right_joint_state_topic", "/right/joint_states")
        self.declare_parameter("motion_scale", 1.0)
        self.declare_parameter("max_joint_step", 0.08)
        self.declare_parameter("min_command_interval", 0.05)
        self.declare_parameter("arm_points_are_relative", True)
        self.declare_parameter("left_camera_to_robot_matrix", default_camera_to_robot_matrix("left"))
        self.declare_parameter("right_camera_to_robot_matrix", default_camera_to_robot_matrix("right"))

        self.motion_scale = float(self.get_parameter("motion_scale").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        self.min_command_interval = float(self.get_parameter("min_command_interval").value)
        self.arm_points_are_relative = read_bool(self, "arm_points_are_relative")
        self.left_camera_to_robot_matrix = read_matrix(self, "left_camera_to_robot_matrix")
        self.right_camera_to_robot_matrix = read_matrix(self, "right_camera_to_robot_matrix")

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.home = np.asarray([0.0, 1.5707963, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.latest = {"left": self.home.copy(), "right": self.home.copy()}
        self.last_command_time = {"left": 0.0, "right": 0.0}
        self.limits = np.asarray([
            [-2.70526, 2.70526],
            [-1.74, 1.74],
            [-2.75, 2.75],
            [-1.01, 2.14],
            [-2.75, 2.75],
            [-0.73, 0.95],
            [-1.5707963, 1.5707963],
        ], dtype=float)

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("left_input_topic").value,
            lambda msg: self.arm_callback("left", msg),
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("right_input_topic").value,
            lambda msg: self.arm_callback("right", msg),
            10,
        )
        self.left_pub = self.create_publisher(
            JointState, self.get_parameter("left_joint_state_topic").value, 10
        )
        self.right_pub = self.create_publisher(
            JointState, self.get_parameter("right_joint_state_topic").value, 10
        )

        self.timer = self.create_timer(0.1, self.publish_home_until_pose_ready)
        self.get_logger().warning(
            "RViz joint_state backend enabled. This is for visual validation only; "
            "real Nero control should still use AGX SDK move_p/move_j safety paths."
        )

    def camera_matrix_for_side(self, side):
        if side == "left":
            return self.left_camera_to_robot_matrix
        return self.right_camera_to_robot_matrix

    def rate_limited(self, side):
        now = time.monotonic()
        if now - self.last_command_time[side] < self.min_command_interval:
            return True
        self.last_command_time[side] = now
        return False

    def publish_home_until_pose_ready(self):
        self.publish_joint_state("left", self.latest["left"])
        self.publish_joint_state("right", self.latest["right"])

    def get_relative_vectors(self, shoulder, elbow, wrist):
        if self.arm_points_are_relative:
            upper = elbow
            forearm = wrist - elbow
        else:
            upper = elbow - shoulder
            forearm = wrist - elbow
        return upper, forearm

    def vector_angle(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-6 or b_norm < 1e-6:
            return 0.0
        cos_value = float(np.dot(a, b) / (a_norm * b_norm))
        return float(np.arccos(np.clip(cos_value, -1.0, 1.0)))

    def solve_visual_joints(self, side, shoulder, elbow, wrist):
        upper_cam, forearm_cam = self.get_relative_vectors(shoulder, elbow, wrist)
        matrix = self.camera_matrix_for_side(side)
        upper = self.motion_scale * (matrix @ upper_cam)
        forearm = self.motion_scale * (matrix @ forearm_cam)
        wrist_vec = upper + forearm

        horizontal = max(np.hypot(upper[0], upper[1]), 1e-6)
        q = self.home.copy()

        q[0] = np.arctan2(upper[1], upper[0])
        q[1] = 1.5707963 + np.arctan2(upper[2], horizontal)
        q[2] = 0.35 * np.arctan2(forearm[1], max(abs(forearm[0]), 1e-6))

        elbow_bend = self.vector_angle(upper, forearm)
        q[3] = np.clip(1.5707963 - elbow_bend, self.limits[3, 0], self.limits[3, 1])

        wrist_horizontal = max(np.hypot(wrist_vec[0], wrist_vec[1]), 1e-6)
        q[4] = 0.35 * np.arctan2(wrist_vec[1], wrist_vec[0])
        q[5] = 0.35 * np.arctan2(wrist_vec[2], wrist_horizontal)
        q[6] = 0.0

        if side == "right":
            q[0] = -q[0]
            q[2] = -q[2]
            q[4] = -q[4]

        return np.clip(q, self.limits[:, 0], self.limits[:, 1])

    def limit_step(self, side, target):
        current = self.latest[side]
        delta = np.clip(target - current, -self.max_joint_step, self.max_joint_step)
        return np.clip(current + delta, self.limits[:, 0], self.limits[:, 1])

    def publish_joint_state(self, side, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(value) for value in joints]
        if side == "left":
            self.left_pub.publish(msg)
        else:
            self.right_pub.publish(msg)

    def arm_callback(self, side, msg):
        if len(msg.data) != 9:
            self.get_logger().warning(
                f"{side} arm data length error: {len(msg.data)}",
                throttle_duration_sec=1.0,
            )
            return
        if self.rate_limited(side):
            return

        shoulder = np.asarray(msg.data[0:3], dtype=float)
        elbow = np.asarray(msg.data[3:6], dtype=float)
        wrist = np.asarray(msg.data[6:9], dtype=float)
        if not (
            np.all(np.isfinite(shoulder))
            and np.all(np.isfinite(elbow))
            and np.all(np.isfinite(wrist))
        ):
            return

        target = self.solve_visual_joints(side, shoulder, elbow, wrist)
        joints = self.limit_step(side, target)
        self.latest[side] = joints
        self.publish_joint_state(side, joints)


def main(args=None):
    rclpy.init(args=args)
    filter_node = ArmPointFilterV2()
    backend_probe = Node("position_to_angle")
    backend_probe.declare_parameter("backend", "agx_move_p")
    backend = str(backend_probe.get_parameter("backend").value).strip().lower()
    backend_probe.destroy_node()

    if backend == "rviz_joint_state":
        follower_node = RvizJointStateFollower()
    else:
        follower_node = AgxArmFollowerV2()

    executor = MultiThreadedExecutor()
    executor.add_node(filter_node)
    executor.add_node(follower_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if rclpy.ok():
            raise exc
    finally:
        executor.shutdown()
        filter_node.destroy_node()
        follower_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
