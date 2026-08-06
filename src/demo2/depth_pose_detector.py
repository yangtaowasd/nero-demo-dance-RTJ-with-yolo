#!/usr/bin/env python3
"""Detect arm landmarks in aligned RGB-D images and publish 3-D points."""

from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, ChannelFloat32, Image, PointCloud

from demo2.depth_arm_geometry import (
    DepthLandmarkReconstructor,
    PinholeIntrinsics,
    color_message_to_bgr,
    depth_message_to_meters,
)


MEDIAPIPE_LANDMARK_IDS = (11, 13, 15, 23, 12, 14, 16, 24)
YOLO_LANDMARK_IDS = (5, 7, 9, 11, 6, 8, 10, 12)
LANDMARK_LABELS = ("LS", "LE", "LW", "LH", "RS", "RE", "RW", "RH")
ARM_CONNECTIONS = (
    (0, 1), (1, 2),
    (4, 5), (5, 6),
    (0, 4), (0, 3), (4, 7), (3, 7),
)


def stamp_seconds(stamp):
    """Convert a ROS timestamp to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def default_model_path():
    """Return the installed model, with a source-tree fallback."""
    try:
        share = Path(get_package_share_directory("demo2"))
    except Exception:
        share = Path(__file__).resolve().parents[2]
    return str(share / "model/yolo26n-pose.pt")


class MediaPipeDetector:
    """Small adapter exposing MediaPipe landmarks as pixels/confidences."""

    def __init__(self, model_complexity, detection_confidence, tracking_confidence):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "detector_backend=mediapipe requires the mediapipe package"
            ) from exc
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(np.clip(model_complexity, 0, 2)),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(detection_confidence),
            min_tracking_confidence=float(tracking_confidence),
        )

    def detect(self, frame):
        """Return the eight arm/hip pixel locations and confidences."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        landmarks = result.pose_landmarks
        if landmarks is None or len(landmarks.landmark) < 25:
            return None, None
        selected = [
            landmarks.landmark[index] for index in MEDIAPIPE_LANDMARK_IDS
        ]
        height, width = frame.shape[:2]
        pixels = np.asarray(
            [[point.x * width, point.y * height] for point in selected],
            dtype=float,
        )
        confidence = np.asarray(
            [point.visibility for point in selected], dtype=float
        )
        return pixels, confidence

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


class YoloDetector:
    """Small adapter exposing Ultralytics pose landmarks as pixels."""

    def __init__(self, model_path, person_confidence):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "detector_backend=yolo requires the ultralytics package"
            ) from exc
        self.model = YOLO(str(model_path), task="pose")
        self.person_confidence = float(person_confidence)
        self.last_box = None
        self.last_person_confidence = None

    def detect(self, frame):
        """Return landmarks for the person with the strongest arm scores."""
        self.last_box = None
        self.last_person_confidence = None
        result = self.model(
            frame,
            conf=self.person_confidence,
            max_det=4,
            verbose=False,
        )[0]
        if result.keypoints is None or result.keypoints.data is None:
            return None, None
        people = result.keypoints.data.cpu().numpy()
        if len(people) == 0 or people.shape[1] <= max(YOLO_LANDMARK_IDS):
            return None, None
        arm_people = people[:, YOLO_LANDMARK_IDS, :]
        person_index = int(np.argmax(np.nanmean(arm_people[:, :, 2], axis=1)))
        selected = arm_people[person_index]
        if result.boxes is not None and len(result.boxes) > person_index:
            self.last_box = (
                result.boxes.xyxy[person_index].cpu().numpy().astype(float)
            )
            self.last_person_confidence = float(
                result.boxes.conf[person_index].cpu().item()
            )
        return selected[:, :2].astype(float), selected[:, 2].astype(float)

    def close(self):
        """Release detector resources (Ultralytics needs no explicit close)."""


class DepthPoseDetector(Node):
    """Portable RGB-D recognition node with no robot or IK dependency."""

    def __init__(self):
        super().__init__("depth_pose_detector")
        defaults = {
            "color_topic": "/camera/camera/color/image_raw",
            "aligned_depth_topic": (
                "/camera/camera/aligned_depth_to_color/image_raw"
            ),
            "camera_info_topic": "/camera/camera/color/camera_info",
            "pose_topic": "/realsense/arm_pose_3d",
            "landmarks_topic": "/realsense/landmarks_3d",
            "debug_image_topic": "/realsense/arm_pose_debug",
            "detector_backend": "yolo",
            "model_path": default_model_path(),
            "person_confidence": 0.45,
            "min_landmark_confidence": 0.45,
            "model_complexity": 0,
            "min_detection_confidence": 0.55,
            "min_tracking_confidence": 0.55,
            "depth_uint16_scale": 0.001,
            "depth_window_radius": 4,
            "min_valid_depth_pixels": 4,
            "depth_cluster_tolerance_m": 0.08,
            "min_depth_m": 0.15,
            "max_depth_m": 5.0,
            "sync_tolerance_sec": 0.05,
            "publish_debug_image": True,
            "show_gui": True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def value(name):
            return self.get_parameter(name).value

        backend = str(value("detector_backend")).strip().lower()
        if backend == "yolo":
            self.detector = YoloDetector(
                value("model_path"), value("person_confidence")
            )
        elif backend == "mediapipe":
            self.detector = MediaPipeDetector(
                value("model_complexity"),
                value("min_detection_confidence"),
                value("min_tracking_confidence"),
            )
        else:
            raise ValueError(
                "detector_backend must be either 'yolo' or 'mediapipe'"
            )
        self.backend = backend
        self.min_landmark_confidence = float(
            value("min_landmark_confidence")
        )
        self.uint16_scale = float(value("depth_uint16_scale"))
        self.sync_tolerance = float(value("sync_tolerance_sec"))
        self.publish_debug_image_enabled = bool(value("publish_debug_image"))
        self.show_gui = bool(value("show_gui"))
        self.reconstructor = DepthLandmarkReconstructor(
            radius=int(value("depth_window_radius")),
            min_depth_m=float(value("min_depth_m")),
            max_depth_m=float(value("max_depth_m")),
            min_valid_pixels=int(value("min_valid_depth_pixels")),
            cluster_tolerance_m=float(value("depth_cluster_tolerance_m")),
        )

        self.data_lock = threading.Lock()
        self.latest_color = None
        self.color_sequence = 0
        self.depth_frames = deque(maxlen=8)
        self.intrinsics = None
        self.processed_color_sequence = -1
        self.inference_fps = None
        self.running = True

        self.pose_publisher = self.create_publisher(
            PointCloud, str(value("pose_topic")), qos_profile_sensor_data
        )
        self.landmarks_publisher = self.create_publisher(
            PointCloud,
            str(value("landmarks_topic")),
            qos_profile_sensor_data,
        )
        self.debug_publisher = self.create_publisher(
            Image, str(value("debug_image_topic")), qos_profile_sensor_data
        )
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
        self.worker = threading.Thread(target=self.processing_loop)
        self.worker.start()
        self.get_logger().info(
            f"RGB-D detector ready: backend={backend}, "
            f"pose_topic={value('pose_topic')}"
        )

    def color_callback(self, message):
        """Store the newest decoded color frame."""
        try:
            frame = color_message_to_bgr(message)
        except Exception as exc:
            self.get_logger().error(
                f"color conversion failed: {exc}", throttle_duration_sec=1.0
            )
            return
        with self.data_lock:
            self.color_sequence += 1
            self.latest_color = (
                self.color_sequence,
                stamp_seconds(message.header.stamp),
                message.header,
                np.asarray(frame).copy(),
            )

    def depth_callback(self, message):
        """Store a short queue of decoded depth frames for timestamp matching."""
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
        """Update color-camera intrinsics."""
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
        """Return the nearest aligned depth image for the latest color image."""
        with self.data_lock:
            color = self.latest_color
            depths = list(self.depth_frames)
            intrinsics = self.intrinsics
        if color is None or not depths or intrinsics is None:
            return None
        sequence, color_stamp, header, frame = color
        if sequence == self.processed_color_sequence:
            return None
        depth_stamp, depth = min(
            depths, key=lambda item: abs(item[0] - color_stamp)
        )
        delta = abs(depth_stamp - color_stamp)
        if delta > self.sync_tolerance:
            self.processed_color_sequence = sequence
            return header, frame, None, intrinsics, (
                f"color/depth mismatch {delta * 1000.0:.0f} ms"
            )
        self.processed_color_sequence = sequence
        return header, frame, depth, intrinsics, None

    @staticmethod
    def pose_message(header, points=None, confidence=None):
        """Build the standard PointCloud transport used by the controller."""
        message = DepthPoseDetector.landmarks_message(
            header, points, confidence
        )
        if points is None or not np.all(np.isfinite(points)):
            message.points = []
            message.channels = []
        return message

    @staticmethod
    def landmarks_message(header, points=None, confidence=None):
        """Build a per-landmark XYZ message that permits invalid depths."""
        message = PointCloud()
        message.header = header
        if points is None:
            return message
        points = np.asarray(points, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        message.points = [
            Point32(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            for point in points
        ]
        message.channels = [
            ChannelFloat32(
                name="confidence",
                values=[float(value) for value in confidence],
            ),
            ChannelFloat32(
                name="landmark_id",
                values=[float(index) for index in range(len(points))],
            ),
            ChannelFloat32(
                name="depth_valid",
                values=[
                    float(np.all(np.isfinite(point))) for point in points
                ],
            ),
        ]
        return message

    @staticmethod
    def debug_image_message(header, frame):
        """Encode a contiguous BGR frame without requiring cv_bridge."""
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        message = Image()
        message.header = header
        message.height, message.width = frame.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = int(message.width) * 3
        message.data = frame.tobytes()
        return message

    def annotate_detection(self, frame, pixels, display_points, confidence):
        """Draw the selected person, arm skeleton, confidence, and depth."""
        box = getattr(self.detector, "last_box", None)
        person_confidence = getattr(
            self.detector, "last_person_confidence", None
        )
        if box is not None and np.all(np.isfinite(box)):
            x1, y1, x2, y2 = np.round(box).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 40), 2)
            label = "YOLO person"
            if person_confidence is not None:
                label += f" {person_confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 170, 40),
                2,
                cv2.LINE_AA,
            )
        if pixels is None or confidence is None:
            return
        pixels = np.asarray(pixels, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if pixels.shape != (8, 2) or confidence.shape != (8,):
            return
        for start, end in ARM_CONNECTIONS:
            if (
                np.all(np.isfinite(pixels[[start, end]]))
                and min(confidence[start], confidence[end])
                >= self.min_landmark_confidence
            ):
                p1 = tuple(np.round(pixels[start]).astype(int))
                p2 = tuple(np.round(pixels[end]).astype(int))
                cv2.line(frame, p1, p2, (70, 230, 100), 2, cv2.LINE_AA)
        for index, pixel in enumerate(pixels):
            if not np.all(np.isfinite(pixel)):
                continue
            location = tuple(np.round(pixel).astype(int))
            accepted = confidence[index] >= self.min_landmark_confidence
            color = (70, 230, 100) if accepted else (0, 180, 255)
            cv2.circle(frame, location, 5, color, -1)
            detail = f"{LANDMARK_LABELS[index]} c={confidence[index]:.2f}"
            cv2.putText(
                frame,
                detail,
                (location[0] + 6, location[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
            if (
                display_points is not None
                and np.all(np.isfinite(display_points[index]))
            ):
                x, y, z = display_points[index]
                coordinates = f"({x:+.2f},{y:+.2f},{z:.2f})m"
            else:
                coordinates = "depth invalid"
            cv2.putText(
                frame,
                coordinates,
                (location[0] + 6, location[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

    def publish_result(
        self, header, frame, pixels, points, display_points,
        confidence, status
    ):
        """Publish points and an annotated image for standalone debugging."""
        valid = points is not None
        self.pose_publisher.publish(
            self.pose_message(header, points, confidence)
        )
        self.landmarks_publisher.publish(
            self.landmarks_message(header, display_points, confidence)
        )
        self.annotate_detection(
            frame, pixels, display_points, confidence
        )
        color = (70, 230, 100) if valid else (0, 180, 255)
        cv2.putText(
            frame,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if self.inference_fps is not None:
            cv2.putText(
                frame,
                f"{self.backend.upper()} inference {self.inference_fps:.1f} FPS",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.publish_debug_image_enabled:
            self.debug_publisher.publish(
                self.debug_image_message(header, frame)
            )
        if self.show_gui:
            cv2.imshow("RealSense RGB-D pose", frame)
            cv2.waitKey(1)

    def processing_loop(self):
        """Run inference and depth reconstruction outside ROS callbacks."""
        while self.running and rclpy.ok():
            synchronized = self.take_synchronized_pair()
            if synchronized is None:
                time.sleep(0.002)
                continue
            header, frame, depth, intrinsics, sync_error = synchronized
            pixels = None
            points = None
            display_points = None
            confidence = None
            if sync_error is not None:
                status = sync_error
            elif frame.shape[:2] != (intrinsics.height, intrinsics.width):
                status = "color dimensions do not match CameraInfo"
            elif depth.shape != (intrinsics.height, intrinsics.width):
                status = "aligned depth dimensions do not match color"
            else:
                try:
                    inference_start = time.perf_counter()
                    pixels, confidence = self.detector.detect(frame)
                    inference_seconds = time.perf_counter() - inference_start
                    instant_fps = 1.0 / max(inference_seconds, 1e-6)
                    if self.inference_fps is None:
                        self.inference_fps = instant_fps
                    else:
                        self.inference_fps = (
                            0.85 * self.inference_fps + 0.15 * instant_fps
                        )
                    if pixels is None:
                        status = "person not detected"
                    elif confidence is None:
                        status = "detector returned no landmark confidence"
                    elif not np.all(np.isfinite(pixels)):
                        status = "non-finite image landmark"
                    else:
                        display_points, _ = (
                            self.reconstructor.reconstruct_partial(
                                pixels, depth, intrinsics
                            )
                        )
                        if (
                            float(np.min(confidence))
                            < self.min_landmark_confidence
                        ):
                            status = (
                                "show shoulders, elbows, wrists and hips"
                            )
                        elif not np.all(np.isfinite(display_points)):
                            status = (
                                "missing depth around one or more landmarks"
                            )
                        else:
                            points = display_points
                            status = f"{self.backend} RGB-D pose ready"
                except Exception as exc:
                    self.get_logger().error(
                        f"RGB-D inference failed: {exc}",
                        throttle_duration_sec=1.0,
                    )
                    points = None
                    confidence = None
                    status = "inference/depth reconstruction failed"
            if not self.running or not rclpy.ok():
                break
            try:
                self.publish_result(
                    header, frame, pixels, points, display_points,
                    confidence, status
                )
            except Exception as exc:
                if self.running and rclpy.ok():
                    self.get_logger().error(
                        f"RGB-D result publish failed: {exc}",
                        throttle_duration_sec=1.0,
                    )

    def destroy_node(self):
        """Stop inference and release detector/GUI resources."""
        self.running = False
        if self.worker.is_alive():
            self.worker.join()
        self.detector.close()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Run the standalone RGB-D detector."""
    rclpy.init(args=args)
    node = DepthPoseDetector()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
