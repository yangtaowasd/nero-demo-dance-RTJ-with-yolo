#!/usr/bin/env python3

import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class AdaptiveEMAFilter:
    def __init__(self, alpha_min=0.15, alpha_max=0.7):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.value = None

    def update(self, new_val, alpha):
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        if self.value is None:
            self.value = new_val.copy()
        else:
            self.value = alpha * new_val + (1.0 - alpha) * self.value
        return self.value.copy()


class ArmMediaPipeNode(Node):
    def __init__(self):
        super().__init__("arm_mediapipe_node")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("use_camera_topic", False)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("arm_pose", "/arm_pose")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("history_size", 5)
        self.declare_parameter("arm_conf_thres", 0.6)
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("model_complexity", 1)
        self.declare_parameter("world_scale", 1.0)
        self.declare_parameter("z_sign", -1.0)

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.use_camera_topic = bool(self.get_parameter("use_camera_topic").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.arm_pose_topic = self.get_parameter("arm_pose").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.history_size = int(self.get_parameter("history_size").value)
        self.arm_conf_thres = float(self.get_parameter("arm_conf_thres").value)
        self.world_scale = float(self.get_parameter("world_scale").value)
        self.z_sign = float(self.get_parameter("z_sign").value)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(self.get_parameter("model_complexity").value),
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=float(
                self.get_parameter("min_detection_confidence").value
            ),
            min_tracking_confidence=float(
                self.get_parameter("min_tracking_confidence").value
            ),
        )

        self.arm_landmarks = [
            ("left_shoulder", 11),
            ("left_elbow", 13),
            ("left_wrist", 15),
            ("right_shoulder", 12),
            ("right_elbow", 14),
            ("right_wrist", 16),
        ]

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_kpts = None
        self.histories = deque(maxlen=self.history_size)
        self.running = True
        self.ema_smoothers = {}
        self.prev_raw_kpts = {}
        self.alpha_min = 0.2
        self.alpha_max = 0.6
        self.alpha_slope = 2.0

        self.pose_pub = self.create_publisher(Float32MultiArray, self.arm_pose_topic, 10)
        self.timer = self.create_timer(0.05, self.publish_callback)

        if self.use_camera_topic:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image

            self.bridge = CvBridge()
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback, 10
            )
            self.get_logger().info(f"Subscribed to {self.image_topic}")
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                self.get_logger().error(f"Cannot open camera {self.camera_id}")
                raise RuntimeError("Camera open failed")
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            self.get_logger().info(f"Direct camera capture started (id={self.camera_id})")

        self.infer_thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_thread.start()
        self.get_logger().info("MediaPipe arm pose node started")

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            with self.lock:
                self.latest_frame = cv_image
        except Exception as exc:
            self.get_logger().error(f"Image callback error: {exc}")

    def capture_loop(self):
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                self.get_logger().warn("Failed to capture frame", throttle_duration_sec=1)
            time.sleep(0.001)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()

    def infer_loop(self):
        while self.running and rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = self.pose.process(rgb)
                kpts = self.result_to_arm_world(result)
                if kpts is not None:
                    self.histories.append(kpts.copy())
                with self.lock:
                    self.latest_kpts = kpts
                if self.show_gui:
                    self.visualize(frame, result, kpts)
            except Exception as exc:
                self.get_logger().error(f"MediaPipe inference error: {exc}")
                time.sleep(0.02)

    def result_to_arm_world(self, result):
        if not result.pose_world_landmarks:
            return None

        world = result.pose_world_landmarks.landmark
        image = result.pose_landmarks.landmark if result.pose_landmarks else None
        kpts = np.full((len(self.arm_landmarks), 4), np.nan, dtype=float)

        for arm_id, (_, mp_id) in enumerate(self.arm_landmarks):
            world_lm = world[mp_id]
            visibility = float(getattr(world_lm, "visibility", 1.0))
            presence = float(getattr(world_lm, "presence", visibility))
            confidence = min(visibility, presence)
            if image is not None:
                confidence = min(confidence, float(getattr(image[mp_id], "visibility", confidence)))

            if confidence < 0.05:
                continue

            kpts[arm_id, 0] = self.world_scale * float(world_lm.x)
            kpts[arm_id, 1] = self.world_scale * float(world_lm.y)
            kpts[arm_id, 2] = self.world_scale * self.z_sign * float(world_lm.z)
            kpts[arm_id, 3] = confidence

        return self.smooth_with_history(kpts)

    def smooth_with_history(self, kpts):
        if not self.histories:
            return kpts

        out = kpts.copy()
        history = list(self.histories)
        for i in range(len(out)):
            if np.all(np.isfinite(out[i, :3])) and out[i, 3] >= self.arm_conf_thres:
                continue
            for old in reversed(history):
                if np.all(np.isfinite(old[i, :3])) and old[i, 3] >= self.arm_conf_thres:
                    out[i] = old[i]
                    break
        return out

    def smooth_current(self, kpts):
        smoothed = kpts.copy()
        for i in range(len(kpts)):
            if not np.all(np.isfinite(kpts[i, :3])):
                continue

            key = i
            if key not in self.ema_smoothers:
                self.ema_smoothers[key] = AdaptiveEMAFilter(
                    alpha_min=self.alpha_min,
                    alpha_max=self.alpha_max,
                )
                self.prev_raw_kpts[key] = kpts[i, :3].copy()

            current = kpts[i, :3]
            prev = self.prev_raw_kpts[key]
            dist = float(np.linalg.norm(current - prev))
            self.prev_raw_kpts[key] = current.copy()
            alpha = max(self.alpha_min, min(self.alpha_max, self.alpha_min + dist * self.alpha_slope))

            if kpts[i, 3] > 0.5:
                smoothed[i, :3] = self.ema_smoothers[key].update(current, alpha)
            elif self.ema_smoothers[key].value is not None:
                smoothed[i, :3] = self.ema_smoothers[key].value
        return smoothed

    def publish_callback(self):
        with self.lock:
            kpts = self.latest_kpts.copy() if self.latest_kpts is not None else None

        if kpts is None:
            self.pose_pub.publish(Float32MultiArray(data=[0.0]))
            return

        kpts = self.smooth_current(kpts)
        xyz = kpts[:, :3]
        if not self.arm_points_ready(kpts):
            self.get_logger().warning("arm keypoints are not ready", throttle_duration_sec=1.0)

        data = [1.0]
        data.extend(xyz.flatten().astype(float).tolist())
        self.pose_pub.publish(Float32MultiArray(data=data))

    def arm_points_ready(self, kpts):
        for idx in range(len(self.arm_landmarks)):
            if not np.all(np.isfinite(kpts[idx, :3])) or kpts[idx, 3] < self.arm_conf_thres:
                return False
        return True

    def visualize(self, frame, result, kpts):
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        if kpts is not None:
            text = "MediaPipe arm world xyz -> /arm_pose"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("MediaPipe Pose", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.running = False
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        self.pose.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmMediaPipeNode()
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
