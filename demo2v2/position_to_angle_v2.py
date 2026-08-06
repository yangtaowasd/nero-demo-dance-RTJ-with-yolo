#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def fmt_degrees(values):
    return np.rad2deg(np.asarray(values, dtype=float)).round(1).tolist()


def lerp(bounds, progress):
    return float(bounds[0] + progress * (bounds[1] - bounds[0]))


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

YOLO_J3_DOWN_RANGE = np.deg2rad(np.asarray([0.0, 90.0], dtype=float))
YOLO_J3_UP_RANGE = np.deg2rad(np.asarray([-155.0, 0.0], dtype=float))
YOLO_J3_SIGNS = {"left": 1.0, "right": -1.0}
YOLO_J4_WORK_LIMIT = np.deg2rad(np.asarray([0.0, 110.0], dtype=float))

PARAM_DEFAULTS = {
    "image_points_topic": "/arm_pose_v2/image_points",
    "left_joint_state_topic": "/left/joint_states",
    "right_joint_state_topic": "/right/joint_states",
    "publish_joint_states_enabled": True,
    "command_output_enabled": False,
    "left_command_topic": "/left/neroarm/command_joints",
    "right_command_topic": "/right/neroarm/command_joints",
    "yolo_calibration_duration": 2.0,
    "yolo_still_threshold_px": 180.0,
    "yolo_j1_gain": 1.0,
    "yolo_j2_gain": 1.0,
    "yolo_j1_forward_max_deg": 70.0,
    "yolo_j1_forward_full_ratio": 0.45,
    "yolo_j1_home_deg": 0.0,
    "yolo_j2_home_deg": 0.0,
    "yolo_left_j1_sign": -1.0,
    "yolo_right_j1_sign": 1.0,
    "yolo_left_j2_sign": 1.0,
    "yolo_right_j2_sign": -1.0,
    "lock_j1_enabled": False,
    "lock_j2_enabled": False,
    "lock_j5_j6_j7_enabled": True,
    "lock_j1_deg": 0.0,
    "lock_j2_deg": 0.0,
    "j1_j2_only": False,
    "left_j1_soft_limit_deg": [-155.0, 0.0],
    "right_j1_soft_limit_deg": [-157.0, 157.0],
}


class PositionToAngleV2(Node):
    def __init__(self):
        super().__init__("position_to_angle_v2")

        for name, default in PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.image_points_topic = self.get_parameter("image_points_topic").value
        self.publish_joint_states_enabled = bool(
            self.get_parameter("publish_joint_states_enabled").value
        )
        self.command_output_enabled = bool(self.get_parameter("command_output_enabled").value)
        self.yolo_calibration_duration = float(
            self.get_parameter("yolo_calibration_duration").value
        )
        self.yolo_still_threshold_px = float(
            self.get_parameter("yolo_still_threshold_px").value
        )
        self.yolo_gains = {
            "j1": float(self.get_parameter("yolo_j1_gain").value),
            "j2": float(self.get_parameter("yolo_j2_gain").value),
        }
        self.yolo_j1_forward_max = np.deg2rad(
            float(self.get_parameter("yolo_j1_forward_max_deg").value)
        )
        self.yolo_j1_forward_full_ratio = float(
            np.clip(float(self.get_parameter("yolo_j1_forward_full_ratio").value), 0.05, 0.95)
        )
        self.yolo_homes = {
            "j1": np.deg2rad(float(self.get_parameter("yolo_j1_home_deg").value)),
            "j2": np.deg2rad(float(self.get_parameter("yolo_j2_home_deg").value)),
        }
        self.yolo_j1_signs = {
            "left": float(self.get_parameter("yolo_left_j1_sign").value),
            "right": float(self.get_parameter("yolo_right_j1_sign").value),
        }
        self.yolo_j2_signs = {
            "left": float(self.get_parameter("yolo_left_j2_sign").value),
            "right": float(self.get_parameter("yolo_right_j2_sign").value),
        }
        self.lock_j1_enabled = bool(self.get_parameter("lock_j1_enabled").value)
        self.lock_j2_enabled = bool(self.get_parameter("lock_j2_enabled").value)
        self.lock_j5_j6_j7_enabled = bool(
            self.get_parameter("lock_j5_j6_j7_enabled").value
        )
        self.lock_j1 = np.deg2rad(float(self.get_parameter("lock_j1_deg").value))
        self.lock_j2 = np.deg2rad(float(self.get_parameter("lock_j2_deg").value))
        self.j1_j2_only = bool(self.get_parameter("j1_j2_only").value)
        self.j1_soft_limits = {
            side: np.deg2rad(
                np.asarray(self.get_parameter(f"{side}_j1_soft_limit_deg").value, dtype=float)
            )
            for side in ("left", "right")
        }
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_limits = np.deg2rad(JOINT_LIMITS_DEG)
        self.joint_speed_limits = np.deg2rad(JOINT_SPEED_LIMITS_DEG)
        self.latest = {
            side: self.apply_position_limits(np.asarray([
                self.yolo_homes["j1"],
                self.yolo_homes["j2"],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ], dtype=float))
            for side in ("left", "right")
        }
        self.last_update_time = {"left": None, "right": None}
        self.last_limit_log = {"left": 0.0, "right": 0.0}
        self.last_log = 0.0
        self.yolo_calibration_start = None
        self.yolo_calibration_samples = []
        self.yolo_prev_points = None
        self.yolo_baseline = None

        self.create_subscription(
            Float32MultiArray, self.image_points_topic, self.image_points_callback, 10
        )
        self.joint_pubs = {
            side: self.create_publisher(
                JointState, self.get_parameter(f"{side}_joint_state_topic").value, 10
            )
            for side in ("left", "right")
        }
        self.command_pubs = {
            side: self.create_publisher(
                Float32MultiArray, self.get_parameter(f"{side}_command_topic").value, 10
            )
            for side in ("left", "right")
        }
        self.timer = self.create_timer(0.1, self.publish_latest)

        self.get_logger().info(f"position_to_angle_v2 YOLO-only: {self.image_points_topic}")
        self.get_logger().info(
            f"publish_joint_states={self.publish_joint_states_enabled} "
            f"command_output={self.command_output_enabled}"
        )
        self.get_logger().info(
            f"yolo calibration={self.yolo_calibration_duration:.1f}s "
            f"still_threshold={self.yolo_still_threshold_px:.1f}px"
        )

    def apply_position_limits(self, joints):
        return np.clip(joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def apply_working_limit(self, joint_index, angle):
        return float(
            np.clip(
                angle,
                self.joint_limits[joint_index, 0],
                self.joint_limits[joint_index, 1],
            )
        )

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
        limited = (
            np.any(np.abs(target - raw_target) > 1e-6)
            or np.any(np.abs(delta - raw_delta) > 1e-6)
        )
        if limited and now - self.last_limit_log[side] > 1.0:
            self.last_limit_log[side] = now
            self.get_logger().warning(
                f"{side} limited raw={fmt_degrees(raw_target)} "
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
        points = self.parse_image_points(list(msg.data))
        if points is None:
            return
        if not self.update_yolo_calibration(points):
            return

        for side in ("left", "right"):
            self.limit_step(side, self.solve_yolo_side(side, points))
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
                    f"yolo calibration reset: motion={max_motion:.1f}px",
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
                f"hold still for yolo calibration: {elapsed:.1f}/"
                f"{self.yolo_calibration_duration:.1f}s",
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
                f"yolo calibration range too wide: {max_range:.1f}px",
                throttle_duration_sec=1.0,
            )
            return False

        self.yolo_baseline = {
            "left": self.make_baseline_side(baseline_points, 0),
            "right": self.make_baseline_side(baseline_points, 4),
        }
        self.get_logger().info(
            f"yolo calibrated: left={self.yolo_baseline['left']} "
            f"right={self.yolo_baseline['right']} range={max_range:.1f}px"
        )
        return True

    def make_baseline_side(self, points, offset):
        shoulder = points[offset + 0]
        elbow = points[offset + 1]
        wrist = points[offset + 2]
        forearm_angle = self.segment_angle(elbow, wrist)
        return {
            "upper_len": self.segment_len(shoulder, elbow),
            "upper_angle": self.segment_angle(shoulder, elbow),
            "forearm_angle": forearm_angle,
            "j3_angle": self.forearm_angle_to_j3(forearm_angle),
        }

    def segment_len(self, start, end):
        return max(float(np.linalg.norm(np.asarray(end) - np.asarray(start))), 1e-6)

    def segment_angle(self, start, end):
        vec = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        return float(np.arctan2(vec[0], -vec[1]))

    def ratio_to_linear_forward(self, current_len, baseline_len):
        ratio = np.clip(current_len / max(baseline_len, 1e-6), 0.0, 1.0)
        shrink = 1.0 - ratio
        full_shrink = max(1.0 - self.yolo_j1_forward_full_ratio, 1e-6)
        return float(np.clip(shrink / full_shrink, 0.0, 1.0) * self.yolo_j1_forward_max)

    def forearm_angle_to_j3(self, forearm_angle):
        angle_from_up = abs(normalize_angle(forearm_angle))
        half_turn = np.pi / 2.0
        if angle_from_up <= half_turn:
            progress = 1.0 - np.clip(angle_from_up / half_turn, 0.0, 1.0)
            return lerp(YOLO_J3_UP_RANGE[::-1], progress)
        progress = np.clip((angle_from_up - half_turn) / half_turn, 0.0, 1.0)
        return lerp(YOLO_J3_DOWN_RANGE, progress)

    def solve_yolo_side(self, side, points):
        offset = 0 if side == "left" else 4
        shoulder = points[offset + 0, :2]
        elbow = points[offset + 1, :2]
        wrist = points[offset + 2, :2]
        baseline = self.yolo_baseline[side]

        upper_len = self.segment_len(shoulder, elbow)
        upper_angle = self.segment_angle(shoulder, elbow)
        forearm_angle = self.segment_angle(elbow, wrist)
        upper_delta = normalize_angle(upper_angle - baseline["upper_angle"])
        forearm_delta = normalize_angle(forearm_angle - baseline["forearm_angle"])

        q = np.zeros(7, dtype=float)
        q[0] = self.yolo_homes["j1"] + (
            self.yolo_j1_signs[side]
            * self.yolo_gains["j1"]
            * self.ratio_to_linear_forward(upper_len, baseline["upper_len"])
        )
        q[1] = self.yolo_homes["j2"] + (
            self.yolo_j2_signs[side] * self.yolo_gains["j2"] * upper_delta
        )
        q[2] = YOLO_J3_SIGNS[side] * (
            self.forearm_angle_to_j3(forearm_angle) - baseline["j3_angle"]
        )
        q[3] = abs(forearm_delta)
        q[0] = self.apply_j1_soft_limit(side, q[0])
        q[1] = self.apply_working_limit(1, q[1])
        q[2] = self.apply_working_limit(2, q[2])
        q[3] = self.apply_custom_working_limit(3, q[3], YOLO_J4_WORK_LIMIT)
        return q

    def log_debug(self):
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return
        self.last_log = now
        left_deg = np.rad2deg(self.latest["left"]).round(1).tolist()
        right_deg = np.rad2deg(self.latest["right"]).round(1).tolist()
        self.get_logger().info(f"yolo joints deg left={left_deg} right={right_deg}")

    def publish_latest(self):
        command_ready = self.yolo_baseline is not None
        for side in ("left", "right"):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            self.latest[side] = self.apply_position_limits(self.latest[side])
            msg.position = self.latest[side].astype(float).tolist()
            if self.publish_joint_states_enabled:
                self.joint_pubs[side].publish(msg)
            if self.command_output_enabled and command_ready:
                self.command_pubs[side].publish(Float32MultiArray(data=msg.position))


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
