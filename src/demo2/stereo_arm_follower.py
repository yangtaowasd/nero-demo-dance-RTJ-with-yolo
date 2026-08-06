#!/usr/bin/env python3
"""Standalone two-camera MediaPipe tracker and Nero direction-IK node.

This experimental node intentionally lives beside, and does not replace, the
existing monocular pipeline. Run it directly from the repository with::

    PYTHONPATH=src python3 src/demo2/stereo_arm_follower.py \
      --ros-args -p calibration_file:=config/stereo_vertical_example.yaml
"""

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from demo2.nero_direction_ik import DirectionIK, NeroKinematics
from demo2.stereo_arm_geometry import (
    CameraModel,
    StereoRig,
    align_vector_rotation,
    arm_direction_components,
    unit,
)


LANDMARK_IDS = (11, 13, 15, 23, 12, 14, 16, 24)
SIDES = ("left", "right")


def open_camera(camera_id, width, height, fps, fourcc):
    camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)
    if fourcc:
        code = str(fourcc).strip().upper()[:4].ljust(4)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open camera {camera_id}")
    return camera


class StereoArmFollower(Node):
    def __init__(self):
        super().__init__("stereo_arm_follower")
        repository = Path(__file__).resolve().parents[2]
        defaults = {
            "lower_camera_id": 0,
            "upper_camera_id": 2,
            "camera_width": 640,
            "camera_height": 480,
            "camera_fps": 30.0,
            "camera_fourcc": "MJPG",
            "calibration_file": str(repository / "config/stereo_vertical_example.yaml"),
            "urdf_file": str(repository / "urdf/nero_description.urdf"),
            "model_complexity": 0,
            "min_visibility": 0.55,
            "upper_fisheye_balance": 0.55,
            "neutral_calibration_sec": 2.0,
            "max_ray_gap_m": 0.08,
            "min_ray_angle_deg": 1.5,
            "max_reprojection_error_px": 8.0,
            "bone_length_tolerance_ratio": 0.30,
            "max_direction_error_deg": 25.0,
            "max_joint_speed_deg_sec": 120.0,
            "show_gui": True,
            "publish_joint_states_enabled": True,
            "command_output_enabled": False,
            "left_joint_state_topic": "/left/joint_states",
            "right_joint_state_topic": "/right/joint_states",
            "left_command_topic": "/left/neroarm/command_joints",
            "right_command_topic": "/right/neroarm/command_joints",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def value(name):
            return self.get_parameter(name).value
        self.width = int(value("camera_width"))
        self.height = int(value("camera_height"))
        self.fps = float(value("camera_fps"))
        fourcc = str(value("camera_fourcc"))
        self.lower_camera = open_camera(
            int(value("lower_camera_id")), self.width, self.height, self.fps, fourcc
        )
        self.upper_camera = open_camera(
            int(value("upper_camera_id")), self.width, self.height, self.fps, fourcc
        )

        raw_rig = StereoRig.from_yaml(str(value("calibration_file")))
        self.upper_map = None
        if raw_rig.upper.model == "fisheye":
            size = (self.width, self.height)
            new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                raw_rig.upper.camera_matrix,
                raw_rig.upper.distortion.reshape(4, 1),
                size,
                np.eye(3),
                balance=float(value("upper_fisheye_balance")),
            )
            self.upper_map = cv2.fisheye.initUndistortRectifyMap(
                raw_rig.upper.camera_matrix,
                raw_rig.upper.distortion.reshape(4, 1),
                np.eye(3),
                new_matrix,
                size,
                cv2.CV_32FC1,
            )
            upper_model = CameraModel("pinhole", new_matrix, np.zeros(5))
            self.rig = StereoRig(
                raw_rig.lower,
                upper_model,
                raw_rig.rotation,
                raw_rig.translation,
            )
        else:
            self.rig = raw_rig

        pose_options = dict(
            static_image_mode=False,
            model_complexity=int(np.clip(int(value("model_complexity")), 0, 2)),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.lower_pose = mp.solutions.pose.Pose(**pose_options)
        self.upper_pose = mp.solutions.pose.Pose(**pose_options)
        self.min_visibility = float(value("min_visibility"))
        self.max_ray_gap_m = float(value("max_ray_gap_m"))
        self.min_ray_angle_deg = float(value("min_ray_angle_deg"))
        self.max_reprojection_error_px = float(value("max_reprojection_error_px"))
        self.bone_tolerance = float(value("bone_length_tolerance_ratio"))
        self.max_direction_error = float(value("max_direction_error_deg"))
        self.neutral_seconds = max(float(value("neutral_calibration_sec")), 0.1)
        self.max_joint_speed = np.deg2rad(float(value("max_joint_speed_deg_sec")))
        self.show_gui = bool(value("show_gui"))
        self.publish_joint_states_enabled = bool(value("publish_joint_states_enabled"))
        self.command_output_enabled = bool(value("command_output_enabled"))

        kinematics = NeroKinematics(str(value("urdf_file")))
        self.solvers = {side: DirectionIK(kinematics) for side in SIDES}
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.calibration_start = None
        self.calibration_samples = {side: [] for side in SIDES}
        self.baseline_bone_lengths = None
        self.latest_joints = {side: np.zeros(7) for side in SIDES}
        self.latest_valid = False
        self.latest_status = "starting cameras"
        self.last_solution_time = None
        self.joint_names = [f"joint{index}" for index in range(1, 8)]

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
        self.lock = threading.Lock()
        self.running = True
        self.latest_previews = None
        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()
        self.create_timer(0.05, self.publish_latest)
        self.get_logger().info(
            "stereo follower started; hold a straight-arm neutral pose for "
            f"{self.neutral_seconds:.1f}s; command output is "
            f"{'ENABLED' if self.command_output_enabled else 'disabled'}"
        )

    def grab_pair(self):
        # grab() on both devices minimizes the software synchronization skew.
        lower_ok = self.lower_camera.grab()
        upper_ok = self.upper_camera.grab()
        if not lower_ok or not upper_ok:
            return None, None
        lower_ok, lower = self.lower_camera.retrieve()
        upper_ok, upper = self.upper_camera.retrieve()
        if not lower_ok or not upper_ok:
            return None, None
        if self.upper_map is not None:
            upper = cv2.remap(
                upper,
                self.upper_map[0],
                self.upper_map[1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return lower, upper

    def detect_pixels(self, pose, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)
        landmarks = result.pose_landmarks
        if landmarks is None or len(landmarks.landmark) < 25:
            return None
        selected = [landmarks.landmark[index] for index in LANDMARK_IDS]
        if min(float(point.visibility) for point in selected) < self.min_visibility:
            return None
        pixels = np.asarray(
            [[point.x * self.width, point.y * self.height] for point in selected],
            dtype=float,
        )
        if not np.all(np.isfinite(pixels)):
            return None
        return pixels

    @staticmethod
    def bone_lengths(points):
        return np.asarray([
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[5] - points[4]),
            np.linalg.norm(points[6] - points[5]),
        ])

    def bone_lengths_valid(self, points):
        lengths = self.bone_lengths(points)
        if np.any(lengths < 0.12) or np.any(lengths > 0.60):
            return False
        if self.baseline_bone_lengths is None:
            return True
        relative_error = np.abs(lengths / self.baseline_bone_lengths - 1.0)
        return float(np.max(relative_error)) <= self.bone_tolerance

    def update_neutral_calibration(self, components, points):
        now = time.monotonic()
        if self.calibration_start is None:
            self.calibration_start = now
        for side in SIDES:
            self.calibration_samples[side].append(components[side][0].copy())
        elapsed = now - self.calibration_start
        if elapsed < self.neutral_seconds:
            self.latest_status = (
                f"hold straight-arm neutral pose {elapsed:.1f}/{self.neutral_seconds:.1f}s"
            )
            return False

        for side in SIDES:
            average = unit(np.mean(self.calibration_samples[side], axis=0))
            self.corrections[side] = align_vector_rotation(average, [1.0, 0.0, 0.0])
        self.baseline_bone_lengths = self.bone_lengths(points)
        self.latest_status = "stereo tracking ready"
        return True

    def solve_points(self, points):
        components = {
            side: arm_direction_components(points, side) for side in SIDES
        }
        if self.baseline_bone_lengths is None:
            return self.update_neutral_calibration(components, points)

        now = time.monotonic()
        dt = 1.0 / max(self.fps, 1.0)
        if self.last_solution_time is not None:
            dt = max(now - self.last_solution_time, 1e-3)
        self.last_solution_time = now
        max_delta = self.max_joint_speed * dt

        errors = []
        proposals = {}
        solver_seeds = {
            side: self.solvers[side].previous.copy() for side in SIDES
        }
        for side in SIDES:
            upper, forearm = components[side]
            correction = self.corrections[side]
            target, error_deg = self.solvers[side].solve(
                correction @ upper, correction @ forearm
            )
            proposals[side] = target
            errors.extend(error_deg.tolist())

        maximum_error = max(errors)
        if maximum_error > self.max_direction_error:
            for side in SIDES:
                self.solvers[side].previous = solver_seeds[side]
            self.latest_status = (
                f"IK rejected: direction error={maximum_error:.1f} deg"
            )
            return False

        for side in SIDES:
            target = proposals[side]
            with self.lock:
                previous = self.latest_joints[side].copy()
            target = previous + np.clip(target - previous, -max_delta, max_delta)
            with self.lock:
                self.latest_joints[side] = target
            self.solvers[side].previous = target.copy()
        self.latest_status = (
            f"tracking; max direction error={maximum_error:.1f} deg"
        )
        return True

    @staticmethod
    def draw_points(frame, pixels, color):
        if pixels is None:
            return
        for point in pixels:
            cv2.circle(frame, tuple(np.round(point).astype(int)), 4, color, -1)

    def processing_loop(self):
        while self.running and rclpy.ok():
            lower, upper = self.grab_pair()
            if lower is None:
                with self.lock:
                    self.latest_valid = False
                    self.latest_status = "camera frame unavailable"
                time.sleep(0.01)
                continue
            lower_pixels = self.detect_pixels(self.lower_pose, lower)
            upper_pixels = self.detect_pixels(self.upper_pose, upper)
            valid = False
            status = "show shoulders, elbows, wrists and hips to both cameras"
            if lower_pixels is not None and upper_pixels is not None:
                points, quality = self.rig.triangulate_landmarks(
                    lower_pixels,
                    upper_pixels,
                    self.max_ray_gap_m,
                    self.min_ray_angle_deg,
                    self.max_reprojection_error_px,
                )
                if points is None:
                    status = "stereo geometry rejected: check calibration/synchronization"
                elif not self.bone_lengths_valid(points):
                    status = "3-D bone lengths rejected"
                else:
                    try:
                        valid = self.solve_points(points)
                        status = self.latest_status
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        status = f"pose/IK rejected: {exc}"

            self.draw_points(lower, lower_pixels, (70, 230, 100))
            self.draw_points(upper, upper_pixels, (70, 230, 100))
            color = (70, 230, 100) if valid else (0, 180, 255)
            for frame in (lower, upper):
                cv2.putText(
                    frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA,
                )
            with self.lock:
                self.latest_valid = valid
                self.latest_status = status
                self.latest_previews = (lower, upper)
            if self.show_gui:
                cv2.imshow("Stereo lower (laptop)", lower)
                cv2.imshow("Stereo upper (fisheye rectified)", upper)
                cv2.waitKey(1)

    def publish_latest(self):
        with self.lock:
            valid = self.latest_valid
            joints = {side: self.latest_joints[side].copy() for side in SIDES}
        stamp = self.get_clock().now().to_msg()
        for side in SIDES:
            message = JointState()
            message.header.stamp = stamp
            message.name = self.joint_names
            message.position = joints[side].tolist()
            if self.publish_joint_states_enabled:
                self.joint_publishers[side].publish(message)
            if self.command_output_enabled and valid:
                self.command_publishers[side].publish(
                    Float32MultiArray(data=message.position)
                )

    def destroy_node(self):
        self.running = False
        self.worker.join(timeout=2.0)
        self.lower_camera.release()
        self.upper_camera.release()
        self.lower_pose.close()
        self.upper_pose.close()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoArmFollower()
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
