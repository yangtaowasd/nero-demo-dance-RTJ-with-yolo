#!/usr/bin/env python3

import threading
import time
from collections import deque
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
        self.declare_parameter("arm_conf_thres", 0.25)
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("model_complexity", 2)
        self.declare_parameter("world_scale", 1.0)
        self.declare_parameter("z_sign", -1.0)
        self.declare_parameter("normalize_limb_lengths", True)
        self.declare_parameter("upper_arm_length", 0.30)
        self.declare_parameter("forearm_length", 0.26)
        self.declare_parameter("max_publish_missing_frames", 8)
        self.declare_parameter("center_roi_enabled", False)
        self.declare_parameter("center_roi_fraction", 0.67)

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.use_camera_topic = bool(self.get_parameter("use_camera_topic").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.arm_pose_topic = self.get_parameter("arm_pose").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.history_size = int(self.get_parameter("history_size").value)
        self.arm_conf_thres = float(self.get_parameter("arm_conf_thres").value)
        self.world_scale = float(self.get_parameter("world_scale").value)
        self.z_sign = float(self.get_parameter("z_sign").value)
        self.normalize_limb_lengths = bool(
            self.get_parameter("normalize_limb_lengths").value
        )
        self.upper_arm_length = float(self.get_parameter("upper_arm_length").value)
        self.forearm_length = float(self.get_parameter("forearm_length").value)
        self.model_complexity = self.read_model_complexity()
        self.max_publish_missing_frames = int(
            self.get_parameter("max_publish_missing_frames").value
        )
        self.center_roi_enabled = bool(
            self.get_parameter("center_roi_enabled").value
        )
        self.center_roi_fraction = float(
            self.get_parameter("center_roi_fraction").value
        )

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
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
        self.arm_connections = [(0, 1), (1, 2), (3, 4), (4, 5)]
        self.arm_short_names = ["L-shoulder", "L-elbow", "L-wrist", "R-shoulder", "R-elbow", "R-wrist"]

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_kpts = None
        self.histories = deque(maxlen=self.history_size)
        self.running = True
        self.ema_smoothers = {}
        self.prev_raw_kpts = {}
        self.last_publish_kpts = None
        self.publish_missing_frames = 0
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
        gui_msg = "enabled, window title: MediaPipe Arm" if self.show_gui else "disabled"
        self.get_logger().info(
            f"MediaPipe arm pose node started; model_complexity={self.model_complexity}; "
            f"normalize_limb_lengths={self.normalize_limb_lengths}; "
            f"center_roi_enabled={self.center_roi_enabled}; "
            f"center_roi_fraction={self.center_roi_fraction:.2f}; GUI {gui_msg}"
        )

    def read_model_complexity(self):
        raw_value = int(self.get_parameter("model_complexity").value)
        if raw_value not in (0, 1, 2):
            self.get_logger().warning(
                f"model_complexity must be 0, 1, or 2; got {raw_value}, using 1"
            )
            return 1
        return raw_value

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
        if not self.pose_in_center_roi(result):
            return None

        world = result.pose_world_landmarks.landmark
        image = result.pose_landmarks.landmark if result.pose_landmarks else None
        kpts = np.full((len(self.arm_landmarks), 4), np.nan, dtype=float)

        for arm_id, (_, mp_id) in enumerate(self.arm_landmarks):
            world_lm = world[mp_id]
            confidence_values = []
            for attr in ("visibility", "presence"):
                value = float(getattr(world_lm, attr, 0.0))
                if value > 0.0:
                    confidence_values.append(value)
            if image is not None:
                image_visibility = float(getattr(image[mp_id], "visibility", 0.0))
                if image_visibility > 0.0:
                    confidence_values.append(image_visibility)
            confidence = max(confidence_values) if confidence_values else 1.0

            if confidence < 0.05:
                continue

            kpts[arm_id, 0] = self.world_scale * float(world_lm.y)
            kpts[arm_id, 1] = self.world_scale * float(world_lm.x)
            kpts[arm_id, 2] = self.world_scale * self.z_sign * float(world_lm.z)
            kpts[arm_id, 3] = confidence

        if self.normalize_limb_lengths:
            kpts = self.normalize_arm_segments(kpts)

        return self.smooth_with_history(kpts)

    def roi_bounds_normalized(self):
        fraction = float(np.clip(self.center_roi_fraction, 0.05, 1.0))
        margin = (1.0 - fraction) * 0.5
        return margin, 1.0 - margin, margin, 1.0 - margin

    def point_in_center_roi(self, x, y):
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        return x_min <= x <= x_max and y_min <= y <= y_max

    def pose_in_center_roi(self, result):
        if not self.center_roi_enabled:
            return True
        if not result.pose_landmarks:
            return False

        landmarks = result.pose_landmarks.landmark
        candidates = []
        for left_id, right_id in ((11, 12), (23, 24)):
            left = landmarks[left_id]
            right = landmarks[right_id]
            if left.visibility >= self.arm_conf_thres and right.visibility >= self.arm_conf_thres:
                candidates.append((
                    0.5 * (float(left.x) + float(right.x)),
                    0.5 * (float(left.y) + float(right.y)),
                ))
        if not candidates:
            for _, mp_id in self.arm_landmarks:
                landmark = landmarks[mp_id]
                if landmark.visibility >= self.arm_conf_thres:
                    candidates.append((float(landmark.x), float(landmark.y)))
        if not candidates:
            return False

        center = np.mean(np.asarray(candidates, dtype=float), axis=0)
        return self.point_in_center_roi(float(center[0]), float(center[1]))

    def normalize_arm_segments(self, kpts):
        out = kpts.copy()
        for shoulder_id, elbow_id, wrist_id in ((0, 1, 2), (3, 4, 5)):
            shoulder = out[shoulder_id, :3]
            elbow = out[elbow_id, :3]
            wrist = out[wrist_id, :3]
            if not (
                np.all(np.isfinite(shoulder))
                and np.all(np.isfinite(elbow))
                and np.all(np.isfinite(wrist))
            ):
                continue

            upper = elbow - shoulder
            forearm = wrist - elbow
            upper_norm = float(np.linalg.norm(upper))
            forearm_norm = float(np.linalg.norm(forearm))
            if upper_norm < 1e-6 or forearm_norm < 1e-6:
                continue

            out[elbow_id, :3] = shoulder + upper / upper_norm * self.upper_arm_length
            out[wrist_id, :3] = out[elbow_id, :3] + forearm / forearm_norm * self.forearm_length
        return out

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
        ready, detail = self.arm_points_ready(kpts)
        if not ready:
            self.get_logger().warning(
                f"arm keypoints are not ready: {detail}",
                throttle_duration_sec=1.0,
            )
            kpts = self.get_publish_fallback(kpts)
            if kpts is None:
                self.pose_pub.publish(Float32MultiArray(data=[0.0]))
                return

        xyz = kpts[:, :3]
        if not np.all(np.isfinite(xyz)):
            self.get_logger().warning(
                "blocked invalid /arm_pose publish because xyz still contains nan/inf",
                throttle_duration_sec=1.0,
            )
            self.pose_pub.publish(Float32MultiArray(data=[0.0]))
            return

        data = [1.0]
        data.extend(xyz.flatten().astype(float).tolist())
        self.last_publish_kpts = kpts.copy()
        self.publish_missing_frames = 0
        self.pose_pub.publish(Float32MultiArray(data=data))

    def get_publish_fallback(self, kpts):
        if np.all(np.isfinite(kpts[:, :3])):
            return kpts
        if (
            self.last_publish_kpts is not None
            and self.publish_missing_frames < self.max_publish_missing_frames
        ):
            self.publish_missing_frames += 1
            return self.last_publish_kpts.copy()
        self.publish_missing_frames += 1
        return None

    def arm_points_ready(self, kpts):
        missing = []
        for idx in range(len(self.arm_landmarks)):
            name = self.arm_short_names[idx]
            if not np.all(np.isfinite(kpts[idx, :3])):
                missing.append(f"{name}=nan")
            elif kpts[idx, 3] < self.arm_conf_thres:
                missing.append(f"{name}=conf {kpts[idx, 3]:.2f}")
        if missing:
            return False, ", ".join(missing)
        return True, "ok"

    def visualize(self, frame, result, kpts):
        self.draw_center_roi(frame)
        if result.pose_landmarks:
            self.draw_arm_landmarks(frame, result.pose_landmarks.landmark)
        cv2.imshow("MediaPipe Arm", frame)
        cv2.waitKey(1)

    def draw_center_roi(self, frame):
        if not self.center_roi_enabled:
            return
        height, width = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        p1 = (int(x_min * width), int(y_min * height))
        p2 = (int(x_max * width), int(y_max * height))
        cv2.rectangle(frame, p1, p2, (80, 220, 255), 2)

    def draw_coordinate_panel(self, frame, kpts):
        x0, y0 = 10, 58
        row_h = 30
        panel_w = 560
        panel_h = 18 + row_h * (len(self.arm_short_names) + 3)
        cv2.rectangle(
            frame,
            (x0 - 6, y0 - 22),
            (x0 + panel_w, y0 + panel_h),
            (245, 245, 245),
            -1,
        )
        cv2.rectangle(
            frame,
            (x0 - 6, y0 - 22),
            (x0 + panel_w, y0 + panel_h),
            (0, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            "arm world coordinates (m)",
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        for idx, name in enumerate(self.arm_short_names):
            row_y = y0 + 24 + row_h * idx
            point = kpts[idx] if kpts is not None else None
            if point is not None and np.all(np.isfinite(point[:3])):
                line = (
                    f"{name}: x={point[0]: .3f} "
                    f"y={point[1]: .3f} z={point[2]: .3f} c={point[3]:.2f}"
                )
                color = (0, 110, 0) if point[3] >= self.arm_conf_thres else (0, 120, 180)
            else:
                line = f"{name}: not detected"
                color = (0, 0, 220)
            cv2.putText(
                frame,
                line,
                (x0, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        for row_offset, (label, ids) in enumerate(
            (("L len", (0, 1, 2)), ("R len", (3, 4, 5))),
            start=len(self.arm_short_names),
        ):
            row_y = y0 + 24 + row_h * row_offset
            line, color = self.make_limb_length_line(label, ids, kpts)
            cv2.putText(
                frame,
                line,
                (x0, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

    def make_limb_length_line(self, label, ids, kpts):
        if kpts is None:
            return f"{label}: not detected", (0, 0, 220)
        shoulder, elbow, wrist = (kpts[idx, :3] for idx in ids)
        if not (
            np.all(np.isfinite(shoulder))
            and np.all(np.isfinite(elbow))
            and np.all(np.isfinite(wrist))
        ):
            return f"{label}: not detected", (0, 0, 220)
        upper = float(np.linalg.norm(elbow - shoulder))
        forearm = float(np.linalg.norm(wrist - elbow))
        return f"{label}: upper={upper:.3f} forearm={forearm:.3f}", (0, 110, 0)

    def draw_arm_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        image_points = []
        for _, mp_id in self.arm_landmarks:
            landmark = landmarks[mp_id]
            if landmark.visibility < self.arm_conf_thres:
                image_points.append(None)
                continue
            x = int(np.clip(landmark.x, 0.0, 1.0) * width)
            y = int(np.clip(landmark.y, 0.0, 1.0) * height)
            image_points.append((x, y))

        for start, end in self.arm_connections:
            if image_points[start] is not None and image_points[end] is not None:
                cv2.line(frame, image_points[start], image_points[end], (255, 180, 0), 3)

        for point in image_points:
            if point is not None:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

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
