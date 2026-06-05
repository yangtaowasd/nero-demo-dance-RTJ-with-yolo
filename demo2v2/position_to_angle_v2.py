#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


def read_matrix(node, name):
    values = list(node.get_parameter(name).value)
    if len(values) != 9:
        node.get_logger().warning(f"{name} must have 9 values; using identity")
        return np.eye(3)
    return np.asarray(values, dtype=float).reshape(3, 3)


def default_matrix(side):
    if side == "right":
        return [
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
        ]
    return [
        0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
    ]


def unit(vector, fallback=None):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def yaw_pitch(vector):
    horizontal = max(float(np.hypot(vector[0], vector[1])), 1e-6)
    return float(np.arctan2(vector[1], vector[0])), float(np.arctan2(vector[2], horizontal))


def make_msg(points):
    data = []
    for point in points:
        data.extend(np.asarray(point, dtype=float).tolist())
    return Float32MultiArray(data=data)


def fmt_degrees(values):
    return np.rad2deg(np.asarray(values, dtype=float)).round(1).tolist()


JOINT_LIMITS_DEG = np.asarray([
    [-150.0, 150.0],
    [-70.0, 100.0],
    [-150.0, 150.0],
    [-50.0, 110.0],
    [-150.0, 150.0],
    [-35.0, 50.0],
    [-80.0, 80.0],
], dtype=float)

JOINT_SPEED_LIMITS_DEG = np.asarray([
    180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0
], dtype=float)


class PositionToAngleV2(Node):
    def __init__(self):
        super().__init__("position_to_angle_v2")

        self.declare_parameter("input_topic", "/arm_pose_v2")
        self.declare_parameter("image_points_topic", "/arm_pose_v2/image_points")
        self.declare_parameter("left_joint_state_topic", "/left/joint_states")
        self.declare_parameter("right_joint_state_topic", "/right/joint_states")
        self.declare_parameter("left_debug_topic", "/left_nero_points_v2")
        self.declare_parameter("right_debug_topic", "/right_nero_points_v2")
        self.declare_parameter("left_vector_debug_topic", "/left_limb_vectors_v2")
        self.declare_parameter("right_vector_debug_topic", "/right_limb_vectors_v2")
        self.declare_parameter("left_camera_to_robot_matrix", default_matrix("left"))
        self.declare_parameter("right_camera_to_robot_matrix", default_matrix("right"))
        self.declare_parameter("solve_mode", "forearm")
        self.declare_parameter("yolo_calibration_duration", 2.0)
        self.declare_parameter("yolo_still_threshold_px", 180.0)
        self.declare_parameter("yolo_j1_gain", 1.0)
        self.declare_parameter("yolo_j2_gain", 1.0)
        self.declare_parameter("yolo_j3_gain", 1.0)
        self.declare_parameter("yolo_j4_gain", 1.0)
        self.declare_parameter("yolo_j1_forward_max_deg", 70.0)
        self.declare_parameter("yolo_j1_forward_full_ratio", 0.45)
        self.declare_parameter("yolo_j3_down_min_deg", -90.0)
        self.declare_parameter("yolo_j3_down_max_deg", 0.0)
        self.declare_parameter("yolo_j3_up_min_deg", -155.0)
        self.declare_parameter("yolo_j3_up_max_deg", -90.0)
        self.declare_parameter("yolo_j4_work_min_deg", 0.0)
        self.declare_parameter("yolo_j4_work_max_deg", 110.0)
        self.declare_parameter("yolo_j1_home_deg", 0.0)
        self.declare_parameter("yolo_j2_home_deg", 0.0)
        self.declare_parameter("yolo_j3_home_deg", 0.0)
        self.declare_parameter("yolo_j4_home_deg", 0.0)
        self.declare_parameter("yolo_left_j1_sign", -1.0)
        self.declare_parameter("yolo_right_j1_sign", 1.0)
        self.declare_parameter("yolo_left_j2_sign", 1.0)
        self.declare_parameter("yolo_right_j2_sign", -1.0)
        self.declare_parameter("yolo_left_j3_sign", 1.0)
        self.declare_parameter("yolo_right_j3_sign", -1.0)
        self.declare_parameter("lock_j1_enabled", False)
        self.declare_parameter("lock_j2_enabled", False)
        self.declare_parameter("lock_j5_j6_j7_enabled", True)
        self.declare_parameter("lock_j1_deg", 0.0)
        self.declare_parameter("lock_j2_deg", 0.0)
        self.declare_parameter("j1_j2_only", False)
        self.declare_parameter("forearm_yaw_gain", 1.0)
        self.declare_parameter("forearm_pitch_gain", 1.6)
        self.declare_parameter("left_j1_offset_deg", -90.0)
        self.declare_parameter("right_j1_offset_deg", 90.0)
        self.declare_parameter("left_j1_sign", -1.0)
        self.declare_parameter("right_j1_sign", -1.0)
        self.declare_parameter("left_j1_soft_limit_deg", [-155.0, 0.0])
        self.declare_parameter("right_j1_soft_limit_deg", [-157.0, 157.0])
        self.declare_parameter("left_j1_forward_sign", -1.0)
        self.declare_parameter("right_j1_forward_sign", 1.0)

        self.left_matrix = read_matrix(self, "left_camera_to_robot_matrix")
        self.right_matrix = read_matrix(self, "right_camera_to_robot_matrix")
        self.solve_mode = str(self.get_parameter("solve_mode").value)
        self.image_points_topic = self.get_parameter("image_points_topic").value
        self.yolo_calibration_duration = float(self.get_parameter("yolo_calibration_duration").value)
        self.yolo_still_threshold_px = float(self.get_parameter("yolo_still_threshold_px").value)
        self.yolo_gains = {
            "j1": float(self.get_parameter("yolo_j1_gain").value),
            "j2": float(self.get_parameter("yolo_j2_gain").value),
            "j3": float(self.get_parameter("yolo_j3_gain").value),
            "j4": float(self.get_parameter("yolo_j4_gain").value),
        }
        self.yolo_j1_forward_max = np.deg2rad(
            float(self.get_parameter("yolo_j1_forward_max_deg").value)
        )
        self.yolo_j1_forward_full_ratio = float(
            np.clip(float(self.get_parameter("yolo_j1_forward_full_ratio").value), 0.05, 0.95)
        )
        self.yolo_j3_down_range = np.deg2rad(np.asarray([
            float(self.get_parameter("yolo_j3_down_min_deg").value),
            float(self.get_parameter("yolo_j3_down_max_deg").value),
        ], dtype=float))
        self.yolo_j3_up_range = np.deg2rad(np.asarray([
            float(self.get_parameter("yolo_j3_up_min_deg").value),
            float(self.get_parameter("yolo_j3_up_max_deg").value),
        ], dtype=float))
        self.yolo_j4_work_limit = np.deg2rad(np.asarray([
            float(self.get_parameter("yolo_j4_work_min_deg").value),
            float(self.get_parameter("yolo_j4_work_max_deg").value),
        ], dtype=float))
        self.yolo_homes = {
            "j1": np.deg2rad(float(self.get_parameter("yolo_j1_home_deg").value)),
            "j2": np.deg2rad(float(self.get_parameter("yolo_j2_home_deg").value)),
            "j3": np.deg2rad(float(self.get_parameter("yolo_j3_home_deg").value)),
            "j4": np.deg2rad(float(self.get_parameter("yolo_j4_home_deg").value)),
        }
        self.yolo_j1_signs = {
            "left": float(self.get_parameter("yolo_left_j1_sign").value),
            "right": float(self.get_parameter("yolo_right_j1_sign").value),
        }
        self.yolo_j2_signs = {
            "left": float(self.get_parameter("yolo_left_j2_sign").value),
            "right": float(self.get_parameter("yolo_right_j2_sign").value),
        }
        self.yolo_j3_signs = {
            "left": float(self.get_parameter("yolo_left_j3_sign").value),
            "right": float(self.get_parameter("yolo_right_j3_sign").value),
        }
        self.lock_j1_enabled = bool(self.get_parameter("lock_j1_enabled").value)
        self.lock_j2_enabled = bool(self.get_parameter("lock_j2_enabled").value)
        self.lock_j5_j6_j7_enabled = bool(self.get_parameter("lock_j5_j6_j7_enabled").value)
        self.lock_j1 = np.deg2rad(float(self.get_parameter("lock_j1_deg").value))
        self.lock_j2 = np.deg2rad(float(self.get_parameter("lock_j2_deg").value))
        self.j1_j2_only = bool(self.get_parameter("j1_j2_only").value)
        self.forearm_yaw_gain = float(self.get_parameter("forearm_yaw_gain").value)
        self.forearm_pitch_gain = float(self.get_parameter("forearm_pitch_gain").value)
        self.j1_offsets = {
            "left": np.deg2rad(float(self.get_parameter("left_j1_offset_deg").value)),
            "right": np.deg2rad(float(self.get_parameter("right_j1_offset_deg").value)),
        }
        self.j1_signs = {
            "left": float(self.get_parameter("left_j1_sign").value),
            "right": float(self.get_parameter("right_j1_sign").value),
        }
        self.j1_soft_limits = {
            "left": np.deg2rad(np.asarray(self.get_parameter("left_j1_soft_limit_deg").value, dtype=float)),
            "right": np.deg2rad(np.asarray(self.get_parameter("right_j1_soft_limit_deg").value, dtype=float)),
        }
        self.j1_forward_signs = {
            "left": float(self.get_parameter("left_j1_forward_sign").value),
            "right": float(self.get_parameter("right_j1_forward_sign").value),
        }
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_limits = np.deg2rad(JOINT_LIMITS_DEG)
        self.joint_speed_limits = np.deg2rad(JOINT_SPEED_LIMITS_DEG)
        self.latest = {
            "left": np.asarray([0.0, np.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            "right": np.asarray([0.0, np.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        }
        self.latest["left"] = self.apply_position_limits(self.latest["left"])
        self.latest["right"] = self.apply_position_limits(self.latest["right"])
        self.last_update_time = {"left": None, "right": None}
        self.last_limit_log = {"left": 0.0, "right": 0.0}
        self.last_log = 0.0
        self.yolo_calibration_start = None
        self.yolo_calibration_samples = []
        self.yolo_prev_points = None
        self.yolo_baseline = None

        if self.solve_mode == "yolo_pixel":
            for side in ("left", "right"):
                self.latest[side][0] = self.yolo_homes["j1"]
                self.latest[side][1] = self.yolo_homes["j2"]
                self.latest[side][2] = self.yolo_homes["j3"]
                self.latest[side][3] = self.yolo_homes["j4"]

        self.create_subscription(
            Float32MultiArray, self.get_parameter("input_topic").value, self.callback, 10
        )
        self.create_subscription(
            Float32MultiArray, self.image_points_topic, self.image_points_callback, 10
        )
        self.left_pub = self.create_publisher(
            JointState, self.get_parameter("left_joint_state_topic").value, 10
        )
        self.right_pub = self.create_publisher(
            JointState, self.get_parameter("right_joint_state_topic").value, 10
        )
        self.left_debug_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("left_debug_topic").value, 10
        )
        self.right_debug_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("right_debug_topic").value, 10
        )
        self.left_vector_debug_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("left_vector_debug_topic").value, 10
        )
        self.right_vector_debug_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("right_vector_debug_topic").value, 10
        )
        self.timer = self.create_timer(0.1, self.publish_latest)
        self.get_logger().info("position_to_angle_v2 started for RViz simulation")
        self.get_logger().info(f"v2 solve_mode={self.solve_mode}")
        self.get_logger().info(
            f"v2 locks j1={self.lock_j1_enabled}:{fmt_degrees([self.lock_j1])[0]} "
            f"j2={self.lock_j2_enabled}:{fmt_degrees([self.lock_j2])[0]} "
            f"j5_j6_j7={self.lock_j5_j6_j7_enabled}"
        )
        self.get_logger().info(f"v2 j1_j2_only={self.j1_j2_only}")
        self.get_logger().info(
            f"v2 forearm_gains yaw={self.forearm_yaw_gain} pitch={self.forearm_pitch_gain}"
        )
        self.get_logger().info(f"v2 j1_offsets deg={fmt_degrees([self.j1_offsets['left'], self.j1_offsets['right']])}")
        self.get_logger().info(f"v2 j1_signs left={self.j1_signs['left']} right={self.j1_signs['right']}")
        self.get_logger().info(
            f"v2 j1_soft_limits deg left={fmt_degrees(self.j1_soft_limits['left'])} "
            f"right={fmt_degrees(self.j1_soft_limits['right'])}"
        )
        self.get_logger().info(
            f"v2 j1_forward_signs left={self.j1_forward_signs['left']} "
            f"right={self.j1_forward_signs['right']}"
        )
        self.get_logger().info(
            f"v2 physical limits deg={fmt_degrees(self.joint_limits)} "
            f"speed deg/s={fmt_degrees(self.joint_speed_limits)}"
        )
        if self.solve_mode == "yolo_pixel":
            self.get_logger().info(
                f"v2 yolo_pixel reads {self.image_points_topic}; "
                f"calibration={self.yolo_calibration_duration:.1f}s "
                f"still_threshold={self.yolo_still_threshold_px:.1f}px"
            )
            self.get_logger().info(
                f"v2 yolo homes deg j1={fmt_degrees([self.yolo_homes['j1']])[0]} "
                f"j2={fmt_degrees([self.yolo_homes['j2']])[0]} "
                f"j3={fmt_degrees([self.yolo_homes['j3']])[0]} "
                f"j4={fmt_degrees([self.yolo_homes['j4']])[0]}"
            )

    def matrix(self, side):
        return self.left_matrix if side == "left" else self.right_matrix

    def transform_side(self, side, block):
        matrix = self.matrix(side)
        vectors = [np.asarray(block[i:i + 3], dtype=float) for i in range(0, len(block), 3)]
        shoulder, elbow, wrist, hand, palm_normal, hand_x = vectors
        shoulder0 = shoulder
        points = [matrix @ (point - shoulder0) for point in (shoulder, elbow, wrist, hand)]
        normal = unit(matrix @ palm_normal, [0.0, 0.0, 1.0])
        hand_axis = unit(matrix @ hand_x, points[3] - points[2])
        return points, normal, hand_axis

    def solve_side(self, side, points, palm_normal, hand_axis):
        shoulder, elbow, wrist, hand = points
        upper = elbow - shoulder
        forearm = wrist - elbow
        hand_vec = hand - wrist

        upper_for_j1 = upper.copy()
        upper_for_j1[0] *= self.j1_forward_signs[side]
        upper_yaw_for_j1, upper_pitch = yaw_pitch(upper_for_j1)
        upper_yaw_for_forearm, _ = yaw_pitch(upper)
        forearm_yaw, forearm_pitch = yaw_pitch(forearm)
        hand_yaw, hand_pitch = yaw_pitch(hand_vec)

        q = np.zeros(7, dtype=float)
        q[0] = self.apply_j1_soft_limit(
            side, self.j1_signs[side] * upper_yaw_for_j1 + self.j1_offsets[side]
        )
        q[1] = np.pi / 2.0 + upper_pitch

        if side == "right":
            q[1] = np.pi - q[1]
        if self.lock_j1_enabled:
            q[0] = self.lock_j1
        if self.lock_j2_enabled:
            q[1] = self.lock_j2
        if self.solve_mode == "upper_arm":
            return q

        if self.solve_mode == "forearm":
            q[2] = self.forearm_yaw_gain * normalize_angle(forearm_yaw)
            q[3] = self.forearm_pitch_gain * forearm_pitch
        else:
            q[2] = self.forearm_yaw_gain * normalize_angle(forearm_yaw - upper_yaw_for_forearm)
            q[3] = self.forearm_pitch_gain * normalize_angle(forearm_pitch - upper_pitch)
        if side == "right":
            q[2] = -q[2]
        if self.solve_mode == "forearm":
            return q

        q[4] = 0.45 * normalize_angle(hand_yaw - forearm_yaw)
        q[5] = 0.45 * normalize_angle(hand_pitch - forearm_pitch)
        q[6] = 0.45 * self.hand_roll(hand_vec, palm_normal)

        if side == "right":
            q[4] = -q[4]
        return q

    def apply_position_limits(self, joints):
        return np.clip(joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def apply_working_limit(self, joint_index, angle, margin_deg):
        margin = np.deg2rad(margin_deg)
        lower = self.joint_limits[joint_index, 0] + margin
        upper = self.joint_limits[joint_index, 1] - margin
        return float(np.clip(angle, lower, upper))

    def apply_custom_working_limit(self, joint_index, angle, limits):
        physical = self.joint_limits[joint_index]
        lower = max(float(limits[0]), float(physical[0]))
        upper = min(float(limits[1]), float(physical[1]))
        return float(np.clip(angle, lower, upper))

    def apply_j1_soft_limit(self, side, angle):
        soft = self.j1_soft_limits[side]
        physical = self.joint_limits[0]
        lower = max(float(soft[0]), float(physical[0]))
        upper = min(float(soft[1]), float(physical[1]))
        return float(np.clip(angle, lower, upper))

    def hand_roll(self, hand_vec, palm_normal):
        forward = unit(hand_vec)
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        side_axis = unit(np.cross(world_up, forward), [0.0, 1.0, 0.0])
        up_axis = unit(np.cross(forward, side_axis), world_up)
        return float(np.arctan2(np.dot(palm_normal, side_axis), np.dot(palm_normal, up_axis)))

    def limit_step(self, side, target):
        now = time.monotonic()
        last = self.last_update_time[side]
        self.last_update_time[side] = now
        dt = 0.1 if last is None else max(now - last, 1e-3)
        current = self.latest[side]
        raw_target = np.asarray(target, dtype=float)
        if self.lock_j1_enabled:
            raw_target[0] = self.lock_j1
        if self.lock_j2_enabled:
            raw_target[1] = self.lock_j2
        if self.j1_j2_only:
            raw_target[2:] = 0.0
        if self.lock_j5_j6_j7_enabled:
            raw_target[4:] = 0.0
        target = self.apply_position_limits(raw_target)
        max_delta = self.joint_speed_limits * dt
        raw_delta = target - current
        delta = np.clip(raw_delta, -max_delta, max_delta)
        self.latest[side] = self.apply_position_limits(current + delta)
        if (
            np.any(np.abs(target - raw_target) > 1e-6)
            or np.any(np.abs(delta - raw_delta) > 1e-6)
        ) and now - self.last_limit_log[side] > 1.0:
            self.last_limit_log[side] = now
            self.get_logger().warning(
                f"{side} v2 limited raw={fmt_degrees(raw_target)} "
                f"target={fmt_degrees(target)} out={fmt_degrees(self.latest[side])}"
            )

    def parse_image_points(self, data):
        if len(data) != 43 or int(data[0]) <= 0:
            return None
        points = np.asarray(data[3:], dtype=float).reshape(8, 5)
        if not np.all(np.isfinite(points)):
            return None
        return points

    def image_points_callback(self, msg):
        if self.solve_mode != "yolo_pixel":
            return
        points = self.parse_image_points(list(msg.data))
        if points is None:
            return
        if not self.update_yolo_calibration(points):
            return

        for side in ("left", "right"):
            target = self.solve_yolo_pixel_side(side, points)
            self.limit_step(side, target)
        self.log_debug()

    def update_yolo_calibration(self, points):
        xy = points[:, :2]
        now = time.monotonic()
        if self.yolo_prev_points is not None:
            max_motion = float(np.max(np.linalg.norm(xy - self.yolo_prev_points, axis=1)))
            if max_motion > self.yolo_still_threshold_px:
                self.yolo_calibration_start = now
                self.yolo_calibration_samples = []
                self.yolo_baseline = None
                self.get_logger().info(
                    f"yolo_pixel calibration reset: motion={max_motion:.1f}px",
                    throttle_duration_sec=1.0,
                )
        if self.yolo_calibration_start is None:
            self.yolo_calibration_start = now

        self.yolo_prev_points = xy.copy()
        if self.yolo_baseline is not None:
            return True

        self.yolo_calibration_samples.append(xy.copy())
        elapsed = now - self.yolo_calibration_start
        if elapsed < self.yolo_calibration_duration:
            self.get_logger().info(
                f"hold still for yolo_pixel calibration: {elapsed:.1f}/{self.yolo_calibration_duration:.1f}s",
                throttle_duration_sec=0.5,
            )
            return False

        sample_points = np.asarray(self.yolo_calibration_samples, dtype=float)
        baseline_points = np.mean(sample_points, axis=0)
        max_range = float(np.max(np.linalg.norm(sample_points - baseline_points, axis=2)))
        if max_range > self.yolo_still_threshold_px:
            self.yolo_calibration_start = now
            self.yolo_calibration_samples = []
            self.yolo_baseline = None
            self.get_logger().info(
                f"yolo_pixel calibration range too wide: {max_range:.1f}px",
                throttle_duration_sec=1.0,
            )
            return False

        self.yolo_baseline = {
            "left": self.make_yolo_baseline_side(baseline_points, 0),
            "right": self.make_yolo_baseline_side(baseline_points, 4),
        }
        self.get_logger().info(
            f"yolo_pixel calibrated: left={self.yolo_baseline['left']} "
            f"right={self.yolo_baseline['right']} range={max_range:.1f}px"
        )
        return True

    def make_yolo_baseline_side(self, points, offset):
        shoulder = points[offset + 0]
        elbow = points[offset + 1]
        wrist = points[offset + 2]
        return {
            "upper_len": self.segment_len(shoulder, elbow),
            "forearm_len": self.segment_len(elbow, wrist),
            "upper_angle": self.segment_angle(shoulder, elbow),
            "forearm_angle": self.segment_angle(elbow, wrist),
        }

    def segment_len(self, start, end):
        return max(float(np.linalg.norm(np.asarray(end) - np.asarray(start))), 1e-6)

    def segment_angle(self, start, end):
        vec = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        return float(np.arctan2(vec[0], -vec[1]))

    def ratio_to_bend(self, current_len, baseline_len):
        ratio = np.clip(current_len / max(baseline_len, 1e-6), 0.0, 1.0)
        return float(np.arccos(ratio))

    def ratio_to_linear_forward(self, current_len, baseline_len, max_angle, full_ratio):
        ratio = np.clip(current_len / max(baseline_len, 1e-6), 0.0, 1.0)
        shrink = 1.0 - ratio
        full_shrink = max(1.0 - full_ratio, 1e-6)
        progress = np.clip(shrink / full_shrink, 0.0, 1.0)
        return float(progress * max_angle)

    def forearm_angle_to_j3(self, forearm_angle):
        angle_from_up = abs(normalize_angle(forearm_angle))
        half_turn = np.pi / 2.0
        if angle_from_up <= half_turn:
            progress = 1.0 - np.clip(angle_from_up / half_turn, 0.0, 1.0)
            return float(
                self.yolo_j3_up_range[1]
                + progress * (self.yolo_j3_up_range[0] - self.yolo_j3_up_range[1])
            )

        progress = np.clip((angle_from_up - half_turn) / half_turn, 0.0, 1.0)
        return float(
            self.yolo_j3_down_range[0]
            + progress * (self.yolo_j3_down_range[1] - self.yolo_j3_down_range[0])
        )

    def solve_yolo_pixel_side(self, side, points):
        offset = 0 if side == "left" else 4
        shoulder = points[offset + 0, :2]
        elbow = points[offset + 1, :2]
        wrist = points[offset + 2, :2]
        baseline = self.yolo_baseline[side]

        upper_len = self.segment_len(shoulder, elbow)
        forearm_len = self.segment_len(elbow, wrist)
        upper_angle = self.segment_angle(shoulder, elbow)
        forearm_angle = self.segment_angle(elbow, wrist)

        q = np.zeros(7, dtype=float)
        upper_delta = normalize_angle(upper_angle - baseline["upper_angle"])
        q[0] = self.yolo_homes["j1"] + (
            self.yolo_j1_signs[side]
            * self.yolo_gains["j1"]
            * self.ratio_to_linear_forward(
                upper_len,
                baseline["upper_len"],
                self.yolo_j1_forward_max,
                self.yolo_j1_forward_full_ratio,
            )
        )
        q[1] = self.yolo_homes["j2"] + (
            self.yolo_j2_signs[side] * self.yolo_gains["j2"] * upper_delta
        )
        q[0] = self.apply_j1_soft_limit(side, q[0])
        q[1] = self.apply_working_limit(1, q[1], 0.0)
        forearm_delta = normalize_angle(forearm_angle - baseline["forearm_angle"])
        q[2] = self.yolo_homes["j3"] + (
            self.yolo_j3_signs[side]
            * self.yolo_gains["j3"]
            * self.forearm_angle_to_j3(forearm_angle)
        )
        q[3] = self.yolo_homes["j4"] + (
            self.yolo_gains["j4"] * abs(forearm_delta)
        )
        q[2] = self.apply_working_limit(2, q[2], 0.0)
        q[3] = self.apply_custom_working_limit(3, q[3], self.yolo_j4_work_limit)
        if self.lock_j1_enabled:
            q[0] = self.lock_j1
        if self.lock_j2_enabled:
            q[1] = self.lock_j2
        return q

    def callback(self, msg):
        if self.solve_mode == "yolo_pixel":
            return
        data = list(msg.data)
        if len(data) != 37 or int(data[0]) <= 0:
            return
        left_block = data[1:19]
        right_block = data[19:37]
        for side, block in (("left", left_block), ("right", right_block)):
            points, palm_normal, hand_axis = self.transform_side(side, block)
            target = self.solve_side(side, points, palm_normal, hand_axis)
            self.limit_step(side, target)
            debug_msg = make_msg([*points, palm_normal, hand_axis])
            if side == "left":
                self.left_debug_pub.publish(debug_msg)
                self.left_vector_debug_pub.publish(self.make_vector_debug(points))
            else:
                self.right_debug_pub.publish(debug_msg)
                self.right_vector_debug_pub.publish(self.make_vector_debug(points))
        self.log_debug()

    def make_vector_debug(self, points):
        shoulder, elbow, wrist, hand = points
        upper = elbow - shoulder
        forearm = wrist - elbow
        hand_vec = hand - wrist
        upper_yaw, upper_pitch = yaw_pitch(upper)
        forearm_yaw, forearm_pitch = yaw_pitch(forearm)
        hand_yaw, hand_pitch = yaw_pitch(hand_vec)
        return make_msg([
            upper,
            forearm,
            hand_vec,
            [upper_yaw, upper_pitch, 0.0],
            [forearm_yaw, forearm_pitch, 0.0],
            [hand_yaw, hand_pitch, 0.0],
        ])

    def log_debug(self):
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return
        self.last_log = now
        left_deg = np.rad2deg(self.latest["left"]).round(1).tolist()
        right_deg = np.rad2deg(self.latest["right"]).round(1).tolist()
        self.get_logger().info(f"v2 joints deg left={left_deg} right={right_deg}")

    def publish_latest(self):
        for side, pub in (("left", self.left_pub), ("right", self.right_pub)):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            self.latest[side] = self.apply_position_limits(self.latest[side])
            msg.position = self.latest[side].astype(float).tolist()
            pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PositionToAngleV2()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
