#!/usr/bin/env python3
"""Generic aligned-depth MediaPipe tracker and Nero direction-IK node."""

from collections import deque
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32MultiArray

from demo2.depth_arm_geometry import (
    DepthLandmarkReconstructor,
    PinholeIntrinsics,
    color_message_to_bgr,
    depth_message_to_meters,
)
from demo2.nero_direction_ik import DirectionIK, NeroKinematics
from demo2.stereo_arm_geometry import (
    align_vector_rotation,
    arm_direction_components,
    unit,
)


LANDMARK_IDS = (11, 13, 15, 23, 12, 14, 16, 24)
SIDES = ("left", "right")


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class DepthArmFollower(Node):
    def __init__(self):
        super().__init__("depth_arm_follower")
        repository = Path(__file__).resolve().parents[2]
        defaults = {
            "color_topic": "/camera/camera/color/image_raw",
            "aligned_depth_topic": (
                "/camera/camera/aligned_depth_to_color/image_raw"
            ),
            "camera_info_topic": "/camera/camera/color/camera_info",
            "urdf_file": str(repository / "urdf/nero_description.urdf"),
            "model_complexity": 0,
            "min_visibility": 0.55,
            "depth_uint16_scale": 0.001,
            "depth_window_radius": 4,
            "min_valid_depth_pixels": 4,
            "depth_cluster_tolerance_m": 0.08,
            "min_depth_m": 0.25,
            "max_depth_m": 4.0,
            "sync_tolerance_sec": 0.05,
            "point_smoothing_alpha": 0.45,
            "max_point_jump_m": 0.25,
            "bone_length_tolerance_ratio": 0.30,
            "neutral_calibration_sec": 2.0,
            "max_direction_error_deg": 25.0,
            "max_joint_speed_deg_sec": 120.0,
            "pose_timeout_sec": 0.35,
            "show_gui": True,
            "publish_joint_states_enabled": True,
            "command_output_enabled": False,
            "left_joint_state_topic": "/left/joint_states",
            "right_joint_state_topic": "/right/joint_states",
            "left_command_topic": "/left/neroarm/command_joints",
            "right_command_topic": "/right/neroarm/command_joints",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def value(name):
            return self.get_parameter(name).value

        self.min_visibility = float(value("min_visibility"))
        self.uint16_scale = float(value("depth_uint16_scale"))
        self.sync_tolerance = float(value("sync_tolerance_sec"))
        self.smoothing_alpha = float(
            np.clip(float(value("point_smoothing_alpha")), 0.0, 1.0)
        )
        self.max_point_jump = float(value("max_point_jump_m"))
        self.bone_tolerance = float(value("bone_length_tolerance_ratio"))
        self.neutral_seconds = float(value("neutral_calibration_sec"))
        self.max_direction_error = float(value("max_direction_error_deg"))
        self.max_joint_speed = np.deg2rad(
            float(value("max_joint_speed_deg_sec"))
        )
        self.pose_timeout = float(value("pose_timeout_sec"))
        self.show_gui = bool(value("show_gui"))
        self.publish_joint_states_enabled = bool(
            value("publish_joint_states_enabled")
        )
        self.command_output_enabled = bool(value("command_output_enabled"))
        self.reconstructor = DepthLandmarkReconstructor(
            radius=int(value("depth_window_radius")),
            min_depth_m=float(value("min_depth_m")),
            max_depth_m=float(value("max_depth_m")),
            min_valid_pixels=int(value("min_valid_depth_pixels")),
            cluster_tolerance_m=float(value("depth_cluster_tolerance_m")),
        )

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(
                np.clip(int(value("model_complexity")), 0, 2)
            ),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        kinematics = NeroKinematics(str(value("urdf_file")))
        self.solvers = {side: DirectionIK(kinematics) for side in SIDES}
        self.joint_limits = kinematics.limits
        self.joint_names = [f"joint{index}" for index in range(1, 8)]

        self.data_lock = threading.Lock()
        self.latest_color = None
        self.depth_frames = deque(maxlen=8)
        self.intrinsics = None
        self.processed_color_stamp = None
        self.previous_points = None
        self.calibration_start = None
        self.calibration_samples = {side: [] for side in SIDES}
        self.calibration_bone_lengths = []
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.baseline_bone_lengths = None
        self.latest_joints = {side: np.zeros(7) for side in SIDES}
        self.latest_valid = False
        self.last_valid_time = None
        self.last_solution_time = None
        self.latest_status = "waiting for aligned color and depth"
        self.running = True

        self.create_subscription(
            Image,
            str(value("color_topic")),
            self.color_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(value("aligned_depth_topic")),
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(value("camera_info_topic")),
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.joint_publishers = {
            side: self.create_publisher(
                JointState, str(value(f"{side}_joint_state_topic")), 10
            )
            for side in SIDES
        }
        self.command_publishers = {
            side: self.create_publisher(
                Float32MultiArray, str(value(f"{side}_command_topic")), 10
            )
            for side in SIDES
        }
        self.create_timer(0.05, self.publish_latest)
        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            "depth follower started; aligned depth is required; commands are "
            f"{'ENABLED' if self.command_output_enabled else 'disabled'}"
        )

    def color_callback(self, message):
        try:
            frame = color_message_to_bgr(message)
        except Exception as exc:
            self.get_logger().error(
                f"color conversion failed: {exc}", throttle_duration_sec=1.0
            )
            return
        with self.data_lock:
            self.latest_color = (
                stamp_seconds(message.header.stamp),
                np.asarray(frame).copy(),
            )

    def depth_callback(self, message):
        try:
            depth = depth_message_to_meters(message, self.uint16_scale)
        except Exception as exc:
            self.get_logger().error(
                f"depth conversion failed: {exc}", throttle_duration_sec=1.0
            )
            return
        with self.data_lock:
            self.depth_frames.append(
                (stamp_seconds(message.header.stamp), depth.copy())
            )

    def camera_info_callback(self, message):
        try:
            intrinsics = PinholeIntrinsics.from_camera_info(message)
        except ValueError as exc:
            self.get_logger().error(
                f"invalid CameraInfo: {exc}", throttle_duration_sec=1.0
            )
            return
        with self.data_lock:
            self.intrinsics = intrinsics

    def take_synchronized_pair(self):
        with self.data_lock:
            color = self.latest_color
            depths = list(self.depth_frames)
            intrinsics = self.intrinsics
        if color is None or not depths or intrinsics is None:
            return None
        color_stamp, frame = color
        if color_stamp == self.processed_color_stamp:
            return None
        depth_stamp, depth = min(
            depths, key=lambda item: abs(item[0] - color_stamp)
        )
        delta = abs(depth_stamp - color_stamp)
        if delta > self.sync_tolerance:
            self.set_invalid(
                f"color/depth timestamp mismatch {delta * 1000.0:.0f} ms"
            )
            return None
        self.processed_color_stamp = color_stamp
        return frame, depth, intrinsics

    def detect_pixels(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        landmarks = result.pose_landmarks
        if landmarks is None or len(landmarks.landmark) < 25:
            return None
        selected = [landmarks.landmark[index] for index in LANDMARK_IDS]
        if min(float(point.visibility) for point in selected) < self.min_visibility:
            return None
        height, width = frame.shape[:2]
        pixels = np.asarray(
            [[point.x * width, point.y * height] for point in selected],
            dtype=float,
        )
        return pixels if np.all(np.isfinite(pixels)) else None

    @staticmethod
    def bone_lengths(points):
        return np.asarray([
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[5] - points[4]),
            np.linalg.norm(points[6] - points[5]),
        ])

    def filter_points(self, points):
        if self.previous_points is None:
            self.previous_points = points.copy()
            return points
        jumps = np.linalg.norm(points - self.previous_points, axis=1)
        if float(np.max(jumps)) > self.max_point_jump:
            return None
        filtered = (
            self.smoothing_alpha * points
            + (1.0 - self.smoothing_alpha) * self.previous_points
        )
        self.previous_points = filtered
        return filtered

    def bone_lengths_valid(self, points):
        lengths = self.bone_lengths(points)
        if np.any(lengths < 0.12) or np.any(lengths > 0.60):
            return False
        if self.baseline_bone_lengths is None:
            return True
        error = np.abs(lengths / self.baseline_bone_lengths - 1.0)
        return float(np.max(error)) <= self.bone_tolerance

    def update_neutral_calibration(self, components, points):
        now = time.monotonic()
        if self.calibration_start is None:
            self.calibration_start = now
        for side in SIDES:
            self.calibration_samples[side].append(components[side][0].copy())
        self.calibration_bone_lengths.append(self.bone_lengths(points))
        elapsed = now - self.calibration_start
        if elapsed < self.neutral_seconds:
            self.latest_status = (
                f"hold straight-arm neutral pose {elapsed:.1f}/"
                f"{self.neutral_seconds:.1f}s"
            )
            return False
        for side in SIDES:
            average = unit(np.mean(self.calibration_samples[side], axis=0))
            self.corrections[side] = align_vector_rotation(
                average, [1.0, 0.0, 0.0]
            )
        self.baseline_bone_lengths = np.mean(
            self.calibration_bone_lengths, axis=0
        )
        self.latest_status = "depth tracking ready"
        return True

    def solve_points(self, points):
        components = {
            side: arm_direction_components(points, side) for side in SIDES
        }
        if self.baseline_bone_lengths is None:
            return self.update_neutral_calibration(components, points)

        proposals = {}
        errors = []
        seeds = {side: self.solvers[side].previous.copy() for side in SIDES}
        for side in SIDES:
            upper, forearm = components[side]
            correction = self.corrections[side]
            proposal, error = self.solvers[side].solve(
                correction @ upper, correction @ forearm
            )
            proposals[side] = proposal
            errors.extend(error.tolist())
        maximum_error = max(errors)
        if maximum_error > self.max_direction_error:
            for side in SIDES:
                self.solvers[side].previous = seeds[side]
            self.latest_status = (
                f"IK rejected: direction error={maximum_error:.1f} deg"
            )
            return False

        now = time.monotonic()
        dt = 1.0 / 30.0
        if self.last_solution_time is not None:
            dt = max(now - self.last_solution_time, 1e-3)
        self.last_solution_time = now
        max_delta = self.max_joint_speed * dt
        with self.data_lock:
            for side in SIDES:
                previous = self.latest_joints[side]
                target = previous + np.clip(
                    proposals[side] - previous, -max_delta, max_delta
                )
                self.latest_joints[side] = np.clip(
                    target, self.joint_limits[:, 0], self.joint_limits[:, 1]
                )
                self.solvers[side].previous = self.latest_joints[side].copy()
            self.last_valid_time = now
        self.latest_status = (
            f"tracking; max direction error={maximum_error:.1f} deg"
        )
        return True

    def set_invalid(self, status):
        with self.data_lock:
            self.latest_valid = False
            self.latest_status = status

    def processing_loop(self):
        while self.running and rclpy.ok():
            synchronized = self.take_synchronized_pair()
            if synchronized is None:
                time.sleep(0.002)
                continue
            frame, depth, intrinsics = synchronized
            if frame.shape[:2] != (intrinsics.height, intrinsics.width):
                self.set_invalid("color dimensions do not match CameraInfo")
                continue
            pixels = self.detect_pixels(frame)
            valid = False
            if pixels is None:
                status = "show shoulders, elbows, wrists and hips"
            else:
                try:
                    points, depths = self.reconstructor.reconstruct(
                        pixels, depth, intrinsics
                    )
                    if points is None:
                        status = "missing valid depth around one or more landmarks"
                    else:
                        points = self.filter_points(points)
                        if points is None:
                            status = "3-D landmark jump rejected"
                        elif not self.bone_lengths_valid(points):
                            status = "3-D bone lengths rejected"
                        else:
                            valid = self.solve_points(points)
                            status = self.latest_status
                except (ValueError, np.linalg.LinAlgError) as exc:
                    status = f"depth/IK rejected: {exc}"

            if pixels is not None:
                for point in pixels:
                    cv2.circle(
                        frame,
                        tuple(np.round(point).astype(int)),
                        4,
                        (70, 230, 100),
                        -1,
                    )
            color = (70, 230, 100) if valid else (0, 180, 255)
            cv2.putText(
                frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA,
            )
            with self.data_lock:
                self.latest_valid = valid
                self.latest_status = status
            if self.show_gui:
                cv2.imshow("Depth camera arm tracking", frame)
                cv2.waitKey(1)

    def publish_latest(self):
        now = time.monotonic()
        with self.data_lock:
            valid = self.latest_valid
            fresh = (
                self.last_valid_time is not None
                and now - self.last_valid_time <= self.pose_timeout
            )
            joints = {
                side: self.latest_joints[side].copy() for side in SIDES
            }
        stamp = self.get_clock().now().to_msg()
        for side in SIDES:
            message = JointState()
            message.header.stamp = stamp
            message.name = self.joint_names
            message.position = joints[side].tolist()
            if self.publish_joint_states_enabled:
                self.joint_publishers[side].publish(message)
            if self.command_output_enabled and valid and fresh:
                self.command_publishers[side].publish(
                    Float32MultiArray(data=message.position)
                )

    def destroy_node(self):
        self.running = False
        self.worker.join(timeout=2.0)
        self.pose.close()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthArmFollower()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
