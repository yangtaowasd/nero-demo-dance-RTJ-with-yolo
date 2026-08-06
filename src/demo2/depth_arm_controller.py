#!/usr/bin/env python3
"""Convert portable RGB-D arm landmarks into Nero joint commands."""

from pathlib import Path
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

from demo2.nero_direction_ik import DirectionIK, NeroKinematics
from demo2.person_camera_calibration import (
    PersonCameraReference,
    StablePoseSamples,
    relative_person_pose,
    rotation_angle_deg,
    rotation_to_quaternion,
    rotation_to_rpy,
    torso_pose,
)
from demo2.stereo_arm_geometry import (
    arm_direction_components,
)


SIDES = ("left", "right")
TORSO_INDICES = (0, 3, 4, 7)


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
            "pose_topic": "/realsense/arm_pose_3d",
            "landmarks_topic": "/realsense/landmarks_3d",
            "urdf_file": default_urdf_path(),
            "min_landmark_confidence": 0.45,
            "min_torso_confidence": 0.55,
            "point_smoothing_alpha": 0.45,
            "max_point_jump_m": 0.25,
            "bone_length_tolerance_ratio": 0.30,
            "neutral_calibration_sec": 3.0,
            "calibration_min_samples": 8,
            "calibration_file": (
                "~/.ros/demo2/realsense_person_calibration.json"
            ),
            "load_calibration_on_start": True,
            "calibration_max_translation_step_m": 0.08,
            "calibration_max_rotation_step_deg": 12.0,
            "calibration_max_consecutive_outliers": 3,
            "max_person_translation_m": 1.0,
            "max_person_rotation_deg": 100.0,
            "max_direction_error_deg": 25.0,
            "max_joint_speed_deg_sec": 120.0,
            "pose_timeout_sec": 0.35,
            "publish_joint_states_enabled": True,
            "command_output_enabled": False,
            "left_joint_state_topic": "/left/joint_states",
            "right_joint_state_topic": "/right/joint_states",
            "left_command_topic": "/left/neroarm/command_joints",
            "right_command_topic": "/right/neroarm/command_joints",
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
        self.smoothing_alpha = float(
            np.clip(float(value("point_smoothing_alpha")), 0.0, 1.0)
        )
        self.max_point_jump = float(value("max_point_jump_m"))
        self.bone_tolerance = float(value("bone_length_tolerance_ratio"))
        self.neutral_seconds = float(value("neutral_calibration_sec"))
        self.calibration_min_samples = int(value("calibration_min_samples"))
        self.calibration_file = Path(
            str(value("calibration_file"))
        ).expanduser()
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
        self.pose_timeout = float(value("pose_timeout_sec"))
        self.publish_joint_states_enabled = bool(
            value("publish_joint_states_enabled")
        )
        self.command_output_enabled = bool(value("command_output_enabled"))

        kinematics = NeroKinematics(str(value("urdf_file")))
        self.solvers = {side: DirectionIK(kinematics) for side in SIDES}
        self.joint_limits = kinematics.limits
        self.joint_names = [f"joint{index}" for index in range(1, 8)]

        self.data_lock = threading.Lock()
        self.previous_points = None
        self.calibration_samples = StablePoseSamples(
            self.calibration_max_translation_step,
            self.calibration_max_rotation_step,
            self.calibration_max_consecutive_outliers,
        )
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.baseline_bone_lengths = None
        self.reference_origin = None
        self.reference_basis = None
        self.reference_frame_id = None
        self.latest_relative_translation = None
        self.latest_relative_rpy = None
        self.latest_joints = {side: np.zeros(7) for side in SIDES}
        self.latest_valid = False
        self.last_valid_time = None
        self.last_solution_time = None
        self.latest_status = "waiting for RGB-D pose"

        self.create_subscription(
            PointCloud,
            str(value("pose_topic")),
            self.pose_callback,
            qos_profile_sensor_data,
        )
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
            f"depth arm controller ready: pose_topic={value('pose_topic')}; "
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
    def confidence_values(message):
        """Read the optional confidence channel from a PointCloud."""
        for channel in message.channels:
            if channel.name == "confidence":
                return np.asarray(channel.values, dtype=float)
        return None

    def publish_status(self, status):
        """Publish a human-readable calibration/control state."""
        self.latest_status = str(status)
        self.calibration_status_publisher.publish(String(data=str(status)))

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

    def load_reference(self):
        """Load a saved camera-to-person reference when available."""
        if not self.calibration_file.is_file():
            self.get_logger().info(
                "no saved person-camera calibration; stand still once"
            )
            return False
        try:
            reference = PersonCameraReference.load(self.calibration_file)
            self.apply_reference(reference)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().warning(
                f"person-camera calibration load failed: {exc}"
            )
            return False
        self.publish_status(
            f"loaded person-camera calibration: {self.calibration_file}"
        )
        self.get_logger().info(self.latest_status)
        return True

    def save_reference(self):
        """Persist the completed person-camera reference."""
        reference = PersonCameraReference(
            self.reference_frame_id,
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
        self.corrections = {side: np.eye(3) for side in SIDES}
        self.calibration_samples.reset()
        self.previous_points = None
        self.last_solution_time = None
        self.latest_relative_translation = None
        self.latest_relative_rpy = None
        with self.data_lock:
            self.latest_valid = False
            self.last_valid_time = None
            for side in SIDES:
                self.latest_joints[side] = np.zeros(7)
                self.solvers[side].previous = np.zeros(7)

    def recalibrate_callback(self, _request, response):
        """Start a fresh natural-standing calibration via Trigger service."""
        self.reset_calibration()
        try:
            self.calibration_file.unlink(missing_ok=True)
        except OSError as exc:
            response.success = False
            response.message = f"could not clear calibration file: {exc}"
            return response
        status = "recalibration started: stand naturally and keep still"
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
        self.latest_relative_translation = translation
        self.latest_relative_rpy = rotation_to_rpy(rotation)
        return origin, basis, translation, rotation

    def decode_pose(self, message):
        """Validate and decode the eight-point transport message."""
        if len(message.points) != 8:
            return None, "waiting for eight RGB-D landmarks"
        points = np.asarray(
            [[point.x, point.y, point.z] for point in message.points],
            dtype=float,
        )
        if not np.all(np.isfinite(points)):
            return None, "non-finite RGB-D landmark"
        confidence = self.confidence_values(message)
        if confidence is not None:
            if confidence.shape != (8,) or not np.all(np.isfinite(confidence)):
                return None, "invalid landmark confidence channel"
            if float(np.min(confidence)) < self.min_landmark_confidence:
                return None, "low-confidence RGB-D landmark"
        return points, None

    def decode_torso_landmarks(self, message):
        """Decode shoulder/hip coordinates without requiring arm landmarks."""
        if len(message.points) != 8:
            return None, "waiting for shoulder and hip landmarks"
        points = np.asarray(
            [[point.x, point.y, point.z] for point in message.points],
            dtype=float,
        )
        torso_points = points[list(TORSO_INDICES)]
        if not np.all(np.isfinite(torso_points)):
            return None, "missing shoulder/hip depth for calibration"
        confidence = self.confidence_values(message)
        if confidence is not None:
            if confidence.shape != (8,) or not np.all(
                np.isfinite(confidence[list(TORSO_INDICES)])
            ):
                return None, "invalid torso confidence"
            if (
                float(np.min(confidence[list(TORSO_INDICES)]))
                < self.min_torso_confidence
            ):
                return None, "low-confidence shoulder/hip landmark"
        for channel in message.channels:
            if channel.name != "depth_valid":
                continue
            valid = np.asarray(channel.values, dtype=float)
            if valid.shape != (8,) or np.any(
                valid[list(TORSO_INDICES)] < 0.5
            ):
                return None, "invalid shoulder/hip depth"
        return points, None

    def filter_points(self, points):
        """Apply jump rejection followed by an exponential point filter."""
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
        """Reject implausible limbs and depth association changes."""
        lengths = self.bone_lengths(points)
        if np.any(lengths < 0.12) or np.any(lengths > 0.60):
            return False
        if self.baseline_bone_lengths is None:
            return True
        error = np.abs(lengths / self.baseline_bone_lengths - 1.0)
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
        try:
            self.save_reference()
            saved = f"; saved to {self.calibration_file}"
        except (OSError, ValueError) as exc:
            saved = f"; save failed: {exc}"
        status = "person-camera calibration complete" + saved
        self.publish_status(status)
        self.get_logger().info(status)
        return False

    def solve_points(self, points):
        """Solve both arms and apply direction-error and velocity limits."""
        components = {
            side: arm_direction_components(points, side) for side in SIDES
        }
        if self.reference_origin is None or self.reference_basis is None:
            self.publish_status(
                "waiting for natural-standing person-camera calibration"
            )
            return False
        if self.baseline_bone_lengths is None:
            self.baseline_bone_lengths = self.bone_lengths(points)
            try:
                self.save_reference()
            except (OSError, ValueError) as exc:
                self.get_logger().warning(
                    f"could not update calibrated bone lengths: {exc}"
                )

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
            self.publish_status(
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
        relative = ""
        if (
            self.latest_relative_translation is not None
            and self.latest_relative_rpy is not None
        ):
            x, y, z = self.latest_relative_translation
            yaw = np.rad2deg(self.latest_relative_rpy[2])
            relative = (
                f"; person delta=({x:+.2f},{y:+.2f},{z:+.2f})m"
                f", yaw={yaw:+.1f} deg"
            )
        self.publish_status(
            f"tracking; max direction error={maximum_error:.1f} deg"
            + relative
        )
        return True

    def set_invalid(self, status):
        """Close the hardware-command gate and record a diagnostic status."""
        now = time.monotonic()
        with self.data_lock:
            self.latest_valid = False
            stale = (
                self.last_valid_time is None
                or now - self.last_valid_time > self.pose_timeout
            )
        self.publish_status(status)
        self.get_logger().warning(status, throttle_duration_sec=1.0)
        if stale:
            self.previous_points = None
            self.last_solution_time = None

    def landmarks_callback(self, message):
        """Calibrate and track torso pose from shoulder/hip landmarks."""
        points, error = self.decode_torso_landmarks(message)
        if error is not None:
            if self.reference_origin is None:
                self.publish_status(error)
            return
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
            self.publish_person_poses(message.header, points)
            if self.reference_origin is None:
                self.update_neutral_calibration(
                    points, message.header.frame_id
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            if self.reference_origin is None:
                self.publish_status(f"person calibration rejected: {exc}")

    def pose_callback(self, message):
        """Consume one portable RGB-D skeleton."""
        points, error = self.decode_pose(message)
        if error is not None:
            if self.reference_origin is None:
                with self.data_lock:
                    self.latest_valid = False
                return
            self.set_invalid(error)
            return
        try:
            points = self.filter_points(points)
            if points is None:
                self.set_invalid("3-D landmark jump rejected")
                return
            if (
                self.reference_frame_id is not None
                and self.reference_frame_id != message.header.frame_id
            ):
                old_frame = self.reference_frame_id
                self.reset_calibration()
                self.get_logger().warning(
                    "saved person calibration frame changed from "
                    f"{old_frame!r} to {message.header.frame_id!r}; "
                    "a new still-standing calibration is required"
                )
            _, _, translation, rotation = self.publish_person_poses(
                message.header, points
            )
            if not self.bone_lengths_valid(points):
                self.set_invalid("3-D bone lengths rejected")
                return
            if translation is not None:
                distance = float(np.linalg.norm(translation))
                angle = rotation_angle_deg(rotation)
                if distance > self.max_person_translation:
                    self.set_invalid(
                        "person moved too far from calibration: "
                        f"{distance:.2f} m"
                    )
                    return
                if angle > self.max_person_rotation:
                    self.set_invalid(
                        "person rotation exceeds safe tracking range: "
                        f"{angle:.1f} deg"
                    )
                    return
            valid = self.solve_points(points)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.set_invalid(f"depth/IK rejected: {exc}")
            return
        with self.data_lock:
            self.latest_valid = valid

    def publish_latest(self):
        """Publish visualization state and gated real-arm commands."""
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


def main(args=None):
    """Run the robot-specific RGB-D pose consumer."""
    rclpy.init(args=args)
    node = DepthArmController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
