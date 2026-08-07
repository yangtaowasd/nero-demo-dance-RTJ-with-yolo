#!/usr/bin/env python3
"""Convert portable RGB-D arm landmarks into Nero joint commands."""

from collections import deque
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, PointCloud
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger

from demo2.arm_sides import (
    ARM_LANDMARK_INDICES,
    DEFAULT_COMMAND_TOPICS,
    DEFAULT_JOINT_STATE_TOPICS,
    REQUIRED_LANDMARK_INDICES,
    SIDES,
    TORSO_LANDMARK_INDICES,
    side_landmarks_valid,
)
from demo2.dual_joint_state_publisher import acquire_instance_lock
from demo2.nero_direction_ik import (
    DirectionIK,
    NeroKinematics,
    side_mount_components,
)
from demo2.person_camera_calibration import (
    PersonCameraReference,
    StablePoseSamples,
    relative_person_pose,
    rotation_angle_deg,
    rotation_to_quaternion,
    torso_pose,
)
from demo2.arm_geometry import (
    arm_direction_components,
)


DEFAULT_DISPLAY_JOINTS = (0.0, 1.5707963, 0.0, 0.0, 0.0, 0.0, 0.0)
BONE_CALIBRATION_WINDOW = 12
BONE_CALIBRATION_MIN_SAMPLES = 8


def smooth_joint_target(
    previous,
    proposal,
    dt,
    smoothing_tau,
    deadband_rad,
    max_speed_rad_sec,
):
    """Apply joint deadband, time-based smoothing, and a speed limit."""
    previous = np.asarray(previous, dtype=float)
    proposal = np.asarray(proposal, dtype=float)
    dt = max(float(dt), 1e-3)
    delta = proposal - previous
    delta[np.abs(delta) < max(float(deadband_rad), 0.0)] = 0.0
    smoothing_tau = max(float(smoothing_tau), 0.0)
    alpha = 1.0
    if smoothing_tau > 1e-6:
        alpha = 1.0 - np.exp(-dt / smoothing_tau)
    delta *= alpha
    maximum_delta = max(float(max_speed_rad_sec), 0.0) * dt
    return previous + np.clip(delta, -maximum_delta, maximum_delta)


def robust_bone_length_baseline(
    samples, min_samples=BONE_CALIBRATION_MIN_SAMPLES
):
    """Return median limb lengths only after enough plausible frames."""
    values = np.asarray(tuple(samples), dtype=float)
    if values.ndim != 2 or values.shape[1:] != (4,):
        return None
    valid = values[
        np.all(np.isfinite(values), axis=1)
        & np.all((values >= 0.12) & (values <= 0.60), axis=1)
    ]
    if len(valid) < max(int(min_samples), 1):
        return None
    return np.median(valid, axis=0)


def default_urdf_path():
    """Return the installed URDF, with a source-tree fallback."""
    try:
        share = Path(get_package_share_directory("demo2"))
    except Exception:
        share = Path(__file__).resolve().parents[2]
    return str(share / "urdf/nero_description.urdf")


class DepthArmController(Node):
    """Robot-specific consumer for the standalone depth-pose topic."""

    def __init__(self):
        """Configure calibration, IK, diagnostics, and command gating."""
        super().__init__("depth_arm_controller")
        defaults = {
            "landmarks_topic": "/realsense/landmarks_3d",
            "urdf_file": default_urdf_path(),
            "min_landmark_confidence": 0.35,
            "min_torso_confidence": 0.45,
            "torso_hold_sec": 0.25,
            "point_smoothing_alpha": 0.30,
            "point_median_window": 3,
            "max_point_jump_m": 0.25,
            "bone_length_tolerance_ratio": 0.30,
            "neutral_calibration_sec": 3.0,
            "calibration_min_samples": 8,
            "calibration_file": (
                "~/.ros/demo2/realsense_person_calibration.json"
            ),
            "calibration_camera_id": "unspecified",
            "load_calibration_on_start": True,
            "calibration_max_translation_step_m": 0.08,
            "calibration_max_rotation_step_deg": 12.0,
            "calibration_max_consecutive_outliers": 3,
            "max_person_translation_m": 1.0,
            "max_person_rotation_deg": 100.0,
            "max_direction_error_deg": 25.0,
            "max_joint_speed_deg_sec": 120.0,
            "joint_smoothing_tau_sec": 0.20,
            "joint_deadband_deg": 0.35,
            "pose_timeout_sec": 0.35,
            "initial_display_positions": list(DEFAULT_DISPLAY_JOINTS),
            "publish_joint_states_enabled": True,
            "command_output_enabled": False,
            "left_joint_state_topic": DEFAULT_JOINT_STATE_TOPICS["left"],
            "right_joint_state_topic": DEFAULT_JOINT_STATE_TOPICS["right"],
            "left_command_topic": DEFAULT_COMMAND_TOPICS["left"],
            "right_command_topic": DEFAULT_COMMAND_TOPICS["right"],
            "left_tracking_status_topic": "/left/tracking_status",
            "right_tracking_status_topic": "/right/tracking_status",
            "person_camera_pose_topic": "/realsense/person_camera_pose",
            "person_relative_pose_topic": "/realsense/person_relative_pose",
            "calibration_status_topic": "/realsense/calibration_status",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def value(name):
            return self.get_parameter(name).value

        self.min_landmark_confidence = float(
            value("min_landmark_confidence")
        )
        self.min_torso_confidence = float(value("min_torso_confidence"))
        self.torso_hold_sec = max(float(value("torso_hold_sec")), 0.0)
        self.smoothing_alpha = float(
            np.clip(float(value("point_smoothing_alpha")), 0.0, 1.0)
        )
        self.point_median_window = max(
            int(value("point_median_window")), 1
        )
        self.max_point_jump = float(value("max_point_jump_m"))
        self.bone_tolerance = float(value("bone_length_tolerance_ratio"))
        self.neutral_seconds = float(value("neutral_calibration_sec"))
        self.calibration_min_samples = int(value("calibration_min_samples"))
        self.calibration_file = Path(
            str(value("calibration_file"))
        ).expanduser()
        self.calibration_camera_id = str(
            value("calibration_camera_id")
        ).strip()
        self.load_calibration_on_start = bool(
            value("load_calibration_on_start")
        )
        self.calibration_max_translation_step = float(
            value("calibration_max_translation_step_m")
        )
        self.calibration_max_rotation_step = float(
            value("calibration_max_rotation_step_deg")
        )
        self.calibration_max_consecutive_outliers = int(
            value("calibration_max_consecutive_outliers")
        )
        self.max_person_translation = float(
            value("max_person_translation_m")
        )
        self.max_person_rotation = float(
            value("max_person_rotation_deg")
        )
        self.max_direction_error = float(value("max_direction_error_deg"))
        self.max_joint_speed = np.deg2rad(
            float(value("max_joint_speed_deg_sec"))
        )
        self.joint_smoothing_tau = float(
            value("joint_smoothing_tau_sec")
        )
        self.joint_deadband = np.deg2rad(
            float(value("joint_deadband_deg"))
        )
        self.pose_timeout = float(value("pose_timeout_sec"))
        self.initial_display_positions = np.asarray(
            value("initial_display_positions"), dtype=float
        )
        if self.initial_display_positions.shape != (7,):
            raise ValueError(
                "initial_display_positions must contain seven values"
            )
        self.publish_joint_states_enabled = bool(
            value("publish_joint_states_enabled")
        )
        self.command_output_enabled = bool(value("command_output_enabled"))

        kinematics = NeroKinematics(str(value("urdf_file")))
        self.solvers = {side: DirectionIK(kinematics) for side in SIDES}
        self.joint_limits = kinematics.limits
        self.joint_names = [f"joint{index}" for index in range(1, 8)]

        self.data_lock = threading.Lock()
        self.side_previous_points = {side: None for side in SIDES}
        self.side_point_history = {
            side: deque(maxlen=self.point_median_window) for side in SIDES
        }
        self.calibration_samples = StablePoseSamples(
            self.calibration_max_translation_step,
            self.calibration_max_rotation_step,
            self.calibration_max_consecutive_outliers,
        )
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.baseline_bone_lengths = None
        self.bone_length_samples = deque(
            maxlen=BONE_CALIBRATION_WINDOW
        )
        self.reference_origin = None
        self.reference_basis = None
        self.reference_frame_id = None
        self.cached_torso_points = None
        self.cached_torso_confidence = None
        self.cached_torso_time = None
        self.latest_joints = {
            side: self.initial_display_positions.copy() for side in SIDES
        }
        self.target_joints = {
            side: self.initial_display_positions.copy() for side in SIDES
        }
        self.has_joint_solution = {side: False for side in SIDES}
        self.latest_valid = {side: False for side in SIDES}
        self.last_valid_time = {side: None for side in SIDES}
        self.last_publish_filter_time = time.monotonic()

        self.create_subscription(
            PointCloud,
            str(value("landmarks_topic")),
            self.landmarks_callback,
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
        self.side_status_publishers = {
            side: self.create_publisher(
                String, str(value(f"{side}_tracking_status_topic")), 10
            )
            for side in SIDES
        }
        self.person_camera_pose_publisher = self.create_publisher(
            PoseStamped, str(value("person_camera_pose_topic")), 10
        )
        self.person_relative_pose_publisher = self.create_publisher(
            PoseStamped, str(value("person_relative_pose_topic")), 10
        )
        self.calibration_status_publisher = self.create_publisher(
            String, str(value("calibration_status_topic")), 10
        )
        self.create_service(
            Trigger, "~/recalibrate", self.recalibrate_callback
        )
        self.create_timer(0.05, self.publish_latest)
        if self.load_calibration_on_start:
            self.load_reference()
        command_state = (
            "ENABLED" if self.command_output_enabled else "disabled"
        )
        self.get_logger().info(
            "depth arm controller ready: anatomical left -> "
            f"{value('left_joint_state_topic')}, anatomical right -> "
            f"{value('right_joint_state_topic')}; "
            f"commands={command_state}"
        )

    @staticmethod
    def bone_lengths(points):
        """Return left/right upper-arm and forearm lengths."""
        return np.asarray([
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[5] - points[4]),
            np.linalg.norm(points[6] - points[5]),
        ])

    @staticmethod
    def channel_values(message, name):
        """Read one optional numeric channel from a PointCloud."""
        for channel in message.channels:
            if channel.name == name:
                return np.asarray(channel.values, dtype=float)
        return None

    @classmethod
    def confidence_values(cls, message):
        """Read the optional confidence channel from a PointCloud."""
        return cls.channel_values(message, "confidence")

    @staticmethod
    def side_bone_lengths(points, side):
        """Return upper-arm and forearm lengths for one anatomical side."""
        shoulder, elbow, wrist = ARM_LANDMARK_INDICES[side]
        return np.asarray([
            np.linalg.norm(points[elbow] - points[shoulder]),
            np.linalg.norm(points[wrist] - points[elbow]),
        ])

    def publish_status(self, status):
        """Publish a human-readable calibration/control state."""
        self.calibration_status_publisher.publish(String(data=str(status)))

    def publish_side_status(self, side, status):
        """Publish an independently observable status for one arm."""
        self.side_status_publishers[side].publish(String(data=str(status)))

    def apply_reference(self, reference):
        """Load a validated person reference into the live controller."""
        self.reference_origin = reference.origin_camera_m.copy()
        self.reference_basis = reference.basis_camera.copy()
        self.reference_frame_id = reference.frame_id
        self.corrections = {
            "left": reference.left_correction.copy(),
            "right": reference.right_correction.copy(),
        }
        if np.all(np.isfinite(reference.bone_lengths_m)):
            self.baseline_bone_lengths = reference.bone_lengths_m.copy()
        else:
            self.baseline_bone_lengths = None
        self.bone_length_samples.clear()

    def load_reference(self):
        """Load a saved camera-to-person reference when available."""
        if not self.calibration_file.is_file():
            self.get_logger().info(
                "no saved person-camera calibration; stand still once"
            )
            return False
        try:
            reference = PersonCameraReference.load(self.calibration_file)
            if reference.camera_id != self.calibration_camera_id:
                raise ValueError(
                    "saved calibration belongs to camera "
                    f"{reference.camera_id!r}, current camera is "
                    f"{self.calibration_camera_id!r}"
                )
            self.apply_reference(reference)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().warning(
                f"person-camera calibration load failed: {exc}"
            )
            return False
        status = f"loaded person-camera calibration: {self.calibration_file}"
        self.publish_status(status)
        self.get_logger().info(status)
        return True

    def save_reference(self):
        """Persist the completed person-camera reference."""
        reference = PersonCameraReference(
            self.reference_frame_id,
            self.calibration_camera_id,
            self.reference_origin,
            self.reference_basis,
            self.corrections["left"],
            self.corrections["right"],
            self.baseline_bone_lengths,
        )
        reference.save(self.calibration_file)

    def reset_calibration(self):
        """Clear the live calibration and close the command gate."""
        self.reference_origin = None
        self.reference_basis = None
        self.reference_frame_id = None
        self.baseline_bone_lengths = None
        self.bone_length_samples.clear()
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.calibration_samples.reset()
        self.cached_torso_points = None
        self.cached_torso_confidence = None
        self.cached_torso_time = None
        for side in SIDES:
            self.side_previous_points[side] = None
            self.side_point_history[side].clear()
        with self.data_lock:
            for side in SIDES:
                self.latest_valid[side] = False
                self.has_joint_solution[side] = False
                self.last_valid_time[side] = None
                self.latest_joints[side] = (
                    self.initial_display_positions.copy()
                )
                self.target_joints[side] = (
                    self.initial_display_positions.copy()
                )
                self.solvers[side].previous = np.zeros(7)
            self.last_publish_filter_time = time.monotonic()

    def recalibrate_callback(self, _request, response):
        """Start a fresh natural-standing calibration via Trigger service."""
        self.reset_calibration()
        try:
            self.calibration_file.unlink(missing_ok=True)
        except OSError as exc:
            response.success = False
            response.message = f"could not clear calibration file: {exc}"
            return response
        status = (
            "recalibration started: stand naturally and keep still; "
            "keep both shoulders and hips visible; arm pose is unrestricted"
        )
        self.publish_status(status)
        self.get_logger().info(status)
        response.success = True
        response.message = status
        return response

    @staticmethod
    def make_pose_message(header, frame_id, position, rotation):
        """Build a PoseStamped from metric position and a rotation matrix."""
        message = PoseStamped()
        message.header.stamp = header.stamp
        message.header.frame_id = str(frame_id)
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        quaternion = rotation_to_quaternion(rotation)
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    def publish_person_poses(self, header, points):
        """Publish current camera-relative and calibrated person poses."""
        origin, basis = torso_pose(points)
        self.person_camera_pose_publisher.publish(
            self.make_pose_message(header, header.frame_id, origin, basis)
        )
        if self.reference_origin is None or self.reference_basis is None:
            return origin, basis, None, None
        translation, rotation = relative_person_pose(
            self.reference_origin,
            self.reference_basis,
            origin,
            basis,
        )
        self.person_relative_pose_publisher.publish(
            self.make_pose_message(
                header, "person_reference", translation, rotation
            )
        )
        return origin, basis, translation, rotation

    def decode_landmark_state(self, message):
        """Decode partial points and per-landmark confidence/depth flags."""
        if len(message.points) != 8:
            return None, None, None, "waiting for eight RGB-D landmarks"
        points = np.asarray(
            [[point.x, point.y, point.z] for point in message.points],
            dtype=float,
        )
        confidence = self.confidence_values(message)
        if confidence is None:
            confidence = np.ones(8, dtype=float)
        if confidence.shape != (8,):
            return None, None, None, "invalid landmark confidence channel"
        depth_valid = self.channel_values(message, "depth_valid")
        if depth_valid is None:
            depth_valid = np.all(np.isfinite(points), axis=1)
        if depth_valid.shape != (8,):
            return None, None, None, "invalid landmark depth channel"
        return points, confidence, depth_valid >= 0.5, None

    def prepare_torso_state(
        self, points, confidence, depth_valid, now=None
    ):
        """Use current torso data or a short calibrated torso hold."""
        points = np.asarray(points, dtype=float).copy()
        confidence = np.asarray(confidence, dtype=float).copy()
        depth_valid = np.asarray(depth_valid, dtype=bool).copy()
        indices = list(TORSO_LANDMARK_INDICES)
        torso_points = points[indices]
        if not np.all(np.isfinite(torso_points)):
            error = "missing shoulder/hip depth"
        elif not np.all(np.isfinite(confidence[indices])):
            error = "invalid torso confidence"
        elif float(np.min(confidence[indices])) < self.min_torso_confidence:
            error = "low-confidence shoulder/hip landmark"
        elif not np.all(depth_valid[indices]):
            error = "invalid shoulder/hip depth"
        else:
            timestamp = time.monotonic() if now is None else float(now)
            self.cached_torso_points = torso_points.copy()
            self.cached_torso_confidence = confidence[indices].copy()
            self.cached_torso_time = timestamp
            return points, confidence, depth_valid, False, None

        timestamp = time.monotonic() if now is None else float(now)
        cache_fresh = (
            self.reference_origin is not None
            and self.cached_torso_points is not None
            and self.cached_torso_confidence is not None
            and self.cached_torso_time is not None
            and timestamp - self.cached_torso_time <= self.torso_hold_sec
        )
        if not cache_fresh:
            return None, None, None, False, error
        points[indices] = self.cached_torso_points
        confidence[indices] = self.cached_torso_confidence
        depth_valid[indices] = True
        return points, confidence, depth_valid, True, None

    def filter_side_points(self, points, side):
        """Filter one arm and its torso inputs without touching the other."""
        indices = list(REQUIRED_LANDMARK_INDICES[side])
        selected = np.asarray(points, dtype=float)[indices]
        previous = self.side_previous_points[side]
        history = self.side_point_history[side]
        if previous is None:
            history.clear()
            history.append(selected.copy())
            self.side_previous_points[side] = selected.copy()
            return np.asarray(points, dtype=float).copy()
        jumps = np.linalg.norm(selected - previous, axis=1)
        if float(np.max(jumps)) > self.max_point_jump:
            return None
        history.append(selected.copy())
        median_points = np.median(
            np.stack(tuple(history), axis=0), axis=0
        )
        filtered = (
            self.smoothing_alpha * median_points
            + (1.0 - self.smoothing_alpha) * previous
        )
        self.side_previous_points[side] = filtered
        output = np.asarray(points, dtype=float).copy()
        output[indices] = filtered
        return output

    def side_bone_lengths_valid(self, points, side):
        """Validate one arm without requiring the opposite arm."""
        lengths = self.side_bone_lengths(points, side)
        if np.any(lengths < 0.12) or np.any(lengths > 0.60):
            return False
        if self.baseline_bone_lengths is None:
            return True
        start = 0 if side == "left" else 2
        baseline = self.baseline_bone_lengths[start:start + 2]
        error = np.abs(lengths / baseline - 1.0)
        return float(np.max(error)) <= self.bone_tolerance

    def update_neutral_calibration(self, points, frame_id):
        """Learn a camera-to-person reference while the person stands still."""
        now = time.monotonic()
        origin, basis = torso_pose(points)
        update = self.calibration_samples.add(origin, basis, now)
        if not update.accepted:
            status = (
                "calibration ignored one unstable torso frame "
                f"(offset={update.translation_error_m:.2f}m, "
                f"rotation={update.rotation_error_deg:.1f}deg)"
            )
            self.publish_status(status)
            self.get_logger().warning(status, throttle_duration_sec=1.0)
            return False
        if update.restarted:
            status = (
                "calibration restarted after sustained motion; stand "
                "naturally and keep still"
            )
            self.publish_status(status)
            self.get_logger().warning(status, throttle_duration_sec=1.0)
        elapsed = self.calibration_samples.elapsed()
        sample_count = self.calibration_samples.sample_count
        if (
            elapsed < self.neutral_seconds
            or sample_count < self.calibration_min_samples
        ):
            status = (
                f"calibrating person-camera reference {elapsed:.1f}/"
                f"{self.neutral_seconds:.1f}s, samples="
                f"{sample_count}/{self.calibration_min_samples}"
            )
            self.publish_status(status)
            self.get_logger().info(
                status, throttle_duration_sec=1.0
            )
            return False
        self.reference_origin, self.reference_basis = (
            self.calibration_samples.center()
        )
        self.reference_frame_id = str(frame_id)
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.baseline_bone_lengths = None
        self.bone_length_samples.clear()
        try:
            self.save_reference()
            saved = f"; saved to {self.calibration_file}"
        except (OSError, ValueError) as exc:
            saved = f"; save failed: {exc}"
        status = "person-camera calibration complete" + saved
        self.publish_status(status)
        self.get_logger().info(status)
        return False

    def learn_complete_bone_lengths(self, points):
        """Persist a robust baseline from several complete RGB-D frames."""
        if self.baseline_bone_lengths is not None:
            return
        lengths = self.bone_lengths(points)
        self.bone_length_samples.append(lengths)
        baseline = robust_bone_length_baseline(
            self.bone_length_samples
        )
        if baseline is None:
            return
        self.baseline_bone_lengths = baseline
        self.bone_length_samples.clear()
        try:
            self.save_reference()
        except (OSError, ValueError) as exc:
            self.get_logger().warning(
                f"could not update calibrated bone lengths: {exc}"
            )

    def solve_side(self, points, side):
        """Solve and update exactly one anatomical arm."""
        if self.reference_origin is None or self.reference_basis is None:
            self.publish_side_status(side, "waiting for torso calibration")
            return False
        upper, forearm = arm_direction_components(points, side)
        correction = self.corrections[side]
        upper = side_mount_components(correction @ upper, side)
        forearm = side_mount_components(correction @ forearm, side)
        proposal, errors = self.solvers[side].solve(
            upper, forearm
        )
        maximum_error = float(np.max(errors))
        if maximum_error > self.max_direction_error:
            # Keep the safe, unpublished solver progress as the next warm
            # start. Restoring the old seed here made a large valid motion
            # repeat the same rejected iteration forever.
            self.set_invalid(
                f"{side} IK rejected: error={maximum_error:.1f} deg",
                side,
            )
            return False

        now = time.monotonic()
        with self.data_lock:
            self.target_joints[side] = np.clip(
                proposal,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
            self.has_joint_solution[side] = True
            self.latest_valid[side] = True
            self.last_valid_time[side] = now
        self.publish_side_status(
            side, f"tracking; IK error={maximum_error:.1f} deg"
        )
        return True

    def set_invalid(self, status, side=None):
        """Hold a side briefly, then close only its command gate."""
        now = time.monotonic()
        targets = SIDES if side is None else (side,)
        stale_sides = []
        with self.data_lock:
            for target in targets:
                last_valid = self.last_valid_time[target]
                stale = (
                    last_valid is None
                    or now - last_valid > self.pose_timeout
                )
                if stale:
                    self.latest_valid[target] = False
                    stale_sides.append(target)
        for target in targets:
            self.publish_side_status(target, status)
        self.get_logger().warning(status, throttle_duration_sec=1.0)
        for target in stale_sides:
            self.side_previous_points[target] = None
            self.side_point_history[target].clear()

    def person_motion_valid(self, translation, rotation):
        """Apply common calibrated torso motion limits."""
        if translation is None:
            return True
        distance = float(np.linalg.norm(translation))
        angle = rotation_angle_deg(rotation)
        if distance > self.max_person_translation:
            self.set_invalid(
                "person moved too far from calibration: "
                f"{distance:.2f} m"
            )
            return False
        if angle > self.max_person_rotation:
            self.set_invalid(
                "person rotation exceeds safe tracking range: "
                f"{angle:.1f} deg"
            )
            return False
        return True

    def landmarks_callback(self, message):
        """Calibrate the torso and update left/right arms independently."""
        points, confidence, depth_valid, decode_error = (
            self.decode_landmark_state(message)
        )
        if decode_error is not None:
            self.set_invalid(decode_error)
            if self.reference_origin is None:
                self.publish_status(decode_error)
            return
        (
            points,
            confidence,
            depth_valid,
            torso_cached,
            torso_error,
        ) = self.prepare_torso_state(points, confidence, depth_valid)
        if torso_error is not None:
            self.set_invalid(torso_error)
            if self.reference_origin is None:
                self.publish_status(torso_error)
            return
        if torso_cached:
            self.get_logger().warning(
                "brief torso dropout; using the latest calibrated torso",
                throttle_duration_sec=1.0,
            )
        torso_points = points
        try:
            if (
                self.reference_frame_id is not None
                and self.reference_frame_id != message.header.frame_id
            ):
                old_frame = self.reference_frame_id
                self.reset_calibration()
                self.get_logger().warning(
                    "saved person calibration frame changed from "
                    f"{old_frame!r} to {message.header.frame_id!r}; "
                    "standing calibration restarted"
                )
            _, _, translation, rotation = self.publish_person_poses(
                message.header, torso_points
            )
            if self.reference_origin is None:
                self.update_neutral_calibration(
                    torso_points, message.header.frame_id
                )
                return
            if not self.person_motion_valid(translation, rotation):
                return

            side_ready = {
                side: side_landmarks_valid(
                    points,
                    confidence,
                    depth_valid,
                    side,
                    self.min_landmark_confidence,
                )
                for side in SIDES
            }
            if all(side_ready.values()):
                self.learn_complete_bone_lengths(points)
            for side in SIDES:
                if not side_ready[side]:
                    self.set_invalid(
                        f"{side} landmarks unavailable; holding last pose",
                        side,
                    )
                    continue
                if not self.side_bone_lengths_valid(points, side):
                    self.set_invalid(
                        f"{side} 3-D bone lengths rejected", side
                    )
                    continue
                filtered = self.filter_side_points(points, side)
                if filtered is None:
                    self.set_invalid(
                        f"{side} 3-D landmark jump rejected", side
                    )
                    continue
                self.solve_side(filtered, side)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.set_invalid(f"depth/IK rejected: {exc}")

    def publish_latest(self):
        """Publish and gate each arm's visualization and commands."""
        now = time.monotonic()
        with self.data_lock:
            dt = max(now - self.last_publish_filter_time, 1e-3)
            self.last_publish_filter_time = now
            for side in SIDES:
                if not self.has_joint_solution[side]:
                    continue
                self.latest_joints[side] = np.clip(
                    smooth_joint_target(
                        self.latest_joints[side],
                        self.target_joints[side],
                        dt,
                        self.joint_smoothing_tau,
                        self.joint_deadband,
                        self.max_joint_speed,
                    ),
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1],
                )
            states = {
                side: {
                    "valid": self.latest_valid[side],
                    "fresh": (
                        self.last_valid_time[side] is not None
                        and now - self.last_valid_time[side]
                        <= self.pose_timeout
                    ),
                    "joints": (
                        self.latest_joints[side].copy()
                        if self.has_joint_solution[side]
                        else self.initial_display_positions.copy()
                    ),
                }
                for side in SIDES
            }
        stamp = self.get_clock().now().to_msg()
        for side in SIDES:
            state = states[side]
            message = JointState()
            message.header.stamp = stamp
            message.name = self.joint_names
            message.position = state["joints"].tolist()
            if self.publish_joint_states_enabled:
                self.joint_publishers[side].publish(message)
            if (
                self.command_output_enabled
                and state["valid"]
                and state["fresh"]
            ):
                self.command_publishers[side].publish(
                    Float32MultiArray(data=message.position)
                )


def main(args=None):
    """Run the robot-specific RGB-D pose consumer."""
    try:
        lock_descriptor = acquire_instance_lock()
    except RuntimeError as exc:
        print(f"depth arm controller not started: {exc}", file=sys.stderr)
        return
    node = None
    try:
        rclpy.init(args=args)
        node = DepthArmController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except (KeyboardInterrupt, Exception):
                pass
        if rclpy.ok():
            rclpy.shutdown()
        os.close(lock_descriptor)


if __name__ == "__main__":
    main()
