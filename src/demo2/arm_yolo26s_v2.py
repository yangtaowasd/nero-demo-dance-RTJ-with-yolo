#!/usr/bin/env python3

import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


def default_yolo26s_pose_path():
    return get_package_share_directory("demo2") + "/model/yolo26s-pose.pt"


def unit(vector, fallback=None):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


class ArmYolo26sV2(Node):
    def __init__(self):
        super().__init__("arm_yolo26s_v2")

        self.declare_parameter("model_path", default_yolo26s_pose_path())
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_width", 1280)
        self.declare_parameter("camera_height", 720)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("camera_fourcc", "MJPG")
        self.declare_parameter("output_topic", "/arm_pose_v2")
        self.declare_parameter("image_points_topic", "/arm_pose_v2/image_points")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("conf_thres", 0.5)
        self.declare_parameter("arm_conf_thres", 0.45)
        self.declare_parameter("center_roi_enabled", True)
        self.declare_parameter("center_roi_fraction", 0.80)
        self.declare_parameter("require_all_arm_points_in_roi", True)
        self.declare_parameter("stable_pose_required", True)
        self.declare_parameter("stable_pose_duration", 1.0)
        self.declare_parameter("pose_read_duration", 2.0)
        self.declare_parameter("stable_motion_threshold_px", 15.0)
        self.declare_parameter("stable_range_threshold_px", 20.0)
        self.declare_parameter("hand_extend_ratio", 0.35)
        self.declare_parameter("show_initial_pose_guide", True)

        self.model_path = self.get_parameter("model_path").value
        self.camera_id = int(self.get_parameter("camera_id").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.camera_fourcc = str(self.get_parameter("camera_fourcc").value).strip().upper()
        self.output_topic = self.get_parameter("output_topic").value
        self.image_points_topic = self.get_parameter("image_points_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.arm_conf_thres = float(self.get_parameter("arm_conf_thres").value)
        self.center_roi_enabled = bool(self.get_parameter("center_roi_enabled").value)
        self.center_roi_fraction = float(self.get_parameter("center_roi_fraction").value)
        self.require_all_arm_points_in_roi = bool(
            self.get_parameter("require_all_arm_points_in_roi").value
        )
        self.stable_pose_required = bool(self.get_parameter("stable_pose_required").value)
        self.stable_pose_duration = float(self.get_parameter("stable_pose_duration").value)
        self.pose_read_duration = float(self.get_parameter("pose_read_duration").value)
        self.stable_motion_threshold_px = float(
            self.get_parameter("stable_motion_threshold_px").value
        )
        self.stable_range_threshold_px = float(
            self.get_parameter("stable_range_threshold_px").value
        )
        self.hand_extend_ratio = float(self.get_parameter("hand_extend_ratio").value)
        self.show_initial_pose_guide = bool(
            self.get_parameter("show_initial_pose_guide").value
        )

        self.model = YOLO(self.model_path, task="pose")
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_msg = None
        self.latest_image_points_msg = None
        self.stable_start_time = None
        self.previous_pose_points = None
        self.stable_pose_samples = []
        self.stable_pose_average_records = None
        self.stable_pose_accepted = False
        self.pose_read_start_time = None
        self.pose_status_text = "waiting for person"
        self.pose_status_ok = False
        self.running = True

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.image_points_pub = self.create_publisher(
            Float32MultiArray, self.image_points_topic, 10
        )
        self.timer = self.create_timer(0.05, self.publish_latest)

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().warning(
                f"Cannot open camera {self.camera_id} with V4L2, trying OpenCV default backend"
            )
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)
        if self.camera_fourcc:
            fourcc = self.camera_fourcc[:4].ljust(4)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_id}. Check /dev/video*, camera_id, "
                "camera permissions, and whether another program is using the camera."
            )

        self.log_camera_settings()
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.infer_loop, daemon=True).start()
        self.get_logger().info(
            f"arm_yolo26s_v2 started: model={self.model_path}, output={self.output_topic}, "
            f"image_points={self.image_points_topic}"
        )

    def log_camera_settings(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        self.get_logger().info(
            f"camera actual: id={self.camera_id}, fourcc={fourcc}, "
            f"size={width}x{height}, fps={fps:.1f}"
        )

    def capture_loop(self):
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest_frame = frame
            else:
                self.get_logger().warning("failed to capture frame", throttle_duration_sec=1.0)
            time.sleep(0.002)

    def infer_loop(self):
        while self.running and rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                result = self.model(frame, conf=self.conf_thres, max_det=1, verbose=False)[0]
                kpts = self.best_keypoints(result)
                records = self.arm_records(kpts)
                height, width = frame.shape[:2]
                ready = self.records_ready(records, width, height)
                stable = self.update_pose_stability(records, width, height) if ready else False
                valid = ready and stable
                output_records = (
                    self.stable_pose_average_records
                    if valid and self.stable_pose_average_records is not None
                    else records
                )
                msg = self.make_pose_msg(output_records, frame.shape[:2], valid)
                image_points_msg = self.make_image_points_msg(
                    output_records, frame.shape[:2], valid
                )
                with self.lock:
                    self.latest_msg = msg
                    self.latest_image_points_msg = image_points_msg
                if self.show_gui:
                    self.draw(frame, records)
            except Exception as exc:
                self.get_logger().error(f"YOLO inference error: {exc}", throttle_duration_sec=1.0)
                time.sleep(0.02)

    def best_keypoints(self, result):
        if result.keypoints is None or result.keypoints.data is None:
            return None
        people = result.keypoints.data.cpu().numpy()
        if len(people) == 0:
            return None
        scores = [float(np.nanmean(person[:, 2])) for person in people]
        return people[int(np.argmax(scores))]

    def arm_records(self, kpts):
        if kpts is None or len(kpts) < 17:
            return []
        left_hand = self.estimated_hand(kpts[7], kpts[9])
        right_hand = self.estimated_hand(kpts[8], kpts[10])
        return [
            ("left_shoulder", kpts[5]),
            ("left_elbow", kpts[7]),
            ("left_wrist", kpts[9]),
            ("left_hand", left_hand),
            ("right_shoulder", kpts[6]),
            ("right_elbow", kpts[8]),
            ("right_wrist", kpts[10]),
            ("right_hand", right_hand),
        ]

    def estimated_hand(self, elbow, wrist):
        hand = np.asarray(wrist, dtype=float).copy()
        hand[:2] = wrist[:2] + (wrist[:2] - elbow[:2]) * self.hand_extend_ratio
        hand[2] = min(float(elbow[2]), float(wrist[2]))
        return hand

    def roi_bounds_normalized(self):
        fraction = float(np.clip(self.center_roi_fraction, 0.05, 1.0))
        margin = (1.0 - fraction) * 0.5
        return margin, 1.0 - margin, margin, 1.0 - margin

    def in_roi(self, x_norm, y_norm):
        if not self.center_roi_enabled:
            return True
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        return x_min <= x_norm <= x_max and y_min <= y_norm <= y_max

    def records_ready(self, records, width, height):
        if not records:
            self.reset_pose_stability("waiting for person")
            return False
        if not self.require_all_arm_points_in_roi:
            return True
        for _, point in records:
            if float(point[2]) < self.arm_conf_thres:
                self.reset_pose_stability("all 8 arm points must be visible")
                return False
            if not self.in_roi(float(point[0]) / width, float(point[1]) / height):
                self.reset_pose_stability("move all 8 arm points into the box")
                return False
        return True

    def reset_pose_stability(self, status_text):
        self.stable_start_time = None
        self.previous_pose_points = None
        self.stable_pose_samples = []
        self.stable_pose_average_records = None
        self.stable_pose_accepted = False
        self.pose_read_start_time = None
        self.pose_status_text = status_text
        self.pose_status_ok = False

    def update_pose_stability(self, records, width, height):
        if not self.stable_pose_required:
            self.pose_status_text = "stable gate disabled"
            self.pose_status_ok = True
            self.stable_pose_average_records = None
            self.stable_pose_accepted = True
            self.pose_read_start_time = time.monotonic()
            return True

        if self.stable_pose_accepted:
            now = time.monotonic()
            if self.pose_read_start_time is None:
                self.pose_read_start_time = now
            elapsed = now - self.pose_read_start_time
            if elapsed < self.pose_read_duration:
                self.pose_status_text = (
                    f"reading pose {elapsed:.1f}/{self.pose_read_duration:.1f}s"
                )
            else:
                self.pose_status_text = "tracking pose"
            self.pose_status_ok = True
            self.stable_pose_average_records = None
            return True

        xy = np.asarray([point[:2] for _, point in records], dtype=float)
        now = time.monotonic()
        if self.previous_pose_points is None:
            self.previous_pose_points = xy.copy()
            self.stable_start_time = now
            self.stable_pose_samples = [(now, self.copy_records(records), xy.copy())]
            self.pose_status_text = f"capture stable 0.0/{self.stable_pose_duration:.1f}s"
            self.pose_status_ok = False
            return False

        max_motion = float(np.max(np.linalg.norm(xy - self.previous_pose_points, axis=1)))
        self.previous_pose_points = xy.copy()
        if max_motion > self.stable_motion_threshold_px:
            self.stable_start_time = now
            self.stable_pose_samples = [(now, self.copy_records(records), xy.copy())]
            self.stable_pose_average_records = None
            self.pose_status_text = f"capture stable: motion {max_motion:.0f}px"
            self.pose_status_ok = False
            return False

        if self.stable_start_time is None:
            self.stable_start_time = now
        self.stable_pose_samples.append((now, self.copy_records(records), xy.copy()))

        elapsed = now - self.stable_start_time
        if elapsed < self.stable_pose_duration:
            self.pose_status_text = (
                f"capture stable {elapsed:.1f}/{self.stable_pose_duration:.1f}s"
            )
            self.pose_status_ok = False
            return False

        window_start = now - self.stable_pose_duration
        window_samples = [
            sample for sample in self.stable_pose_samples if sample[0] >= window_start
        ]
        if len(window_samples) < 2:
            window_samples = self.stable_pose_samples

        sample_xy = np.asarray([sample[2] for sample in window_samples], dtype=float)
        mean_xy = np.mean(sample_xy, axis=0)
        max_range = float(np.max(np.linalg.norm(sample_xy - mean_xy, axis=2)))
        if max_range > self.stable_range_threshold_px:
            self.pose_status_text = f"range too wide: {max_range:.0f}px"
            self.pose_status_ok = False
            self.stable_pose_average_records = None
            self.stable_start_time = now
            self.stable_pose_samples = [(now, self.copy_records(records), xy.copy())]
            return False

        self.stable_pose_samples = window_samples
        self.stable_pose_average_records = self.average_records(window_samples)
        self.stable_pose_accepted = True
        self.pose_read_start_time = now
        self.pose_status_text = f"reading pose 0.0/{self.pose_read_duration:.1f}s"
        self.pose_status_ok = True
        return True

    def copy_records(self, records):
        return [(name, np.asarray(point, dtype=float).copy()) for name, point in records]

    def average_records(self, samples):
        records_by_sample = [sample[1] for sample in samples]
        averaged = []
        for index, (name, _) in enumerate(records_by_sample[0]):
            points = np.asarray([records[index][1] for records in records_by_sample], dtype=float)
            averaged.append((name, np.mean(points, axis=0)))
        return averaged

    def make_pose_msg(self, records, image_shape, valid):
        height, width = image_shape
        if not valid:
            return Float32MultiArray(data=[0.0])

        points = {
            name: self.image_to_camera_point(point, width, height)
            for name, point in records
        }
        data = [1.0]
        data.extend(self.side_block(points, "left"))
        data.extend(self.side_block(points, "right"))
        return Float32MultiArray(data=data)

    def image_to_camera_point(self, point, width, height):
        return np.asarray([float(point[1]) / height, float(point[0]) / width, 0.0], dtype=float)

    def side_block(self, points, side):
        shoulder = points[f"{side}_shoulder"]
        elbow = points[f"{side}_elbow"]
        wrist = points[f"{side}_wrist"]
        hand = points[f"{side}_hand"]
        hand_axis = unit(hand - wrist, wrist - elbow)
        palm_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
        values = []
        for vec in (shoulder, elbow, wrist, hand, palm_normal, hand_axis):
            values.extend(vec.astype(float).tolist())
        return values

    def make_image_points_msg(self, records, image_shape, valid):
        height, width = image_shape
        if not valid:
            return Float32MultiArray(data=[0.0])

        data = [1.0, float(width), float(height)]
        for _, point in records:
            x_norm = float(point[0]) / width
            y_norm = float(point[1]) / height
            data.extend([float(point[0]), float(point[1]), x_norm, y_norm, float(point[2])])
        return Float32MultiArray(data=data)

    def publish_latest(self):
        with self.lock:
            msg = self.latest_msg
            image_points_msg = self.latest_image_points_msg
        if msg is not None:
            self.pub.publish(msg)
        if image_points_msg is not None:
            self.image_points_pub.publish(image_points_msg)

    def draw(self, frame, records):
        self.draw_center_roi(frame)
        self.draw_initial_pose_guide(frame, records)
        for side_records in (records[:4], records[4:8]):
            pts = [(int(point[0]), int(point[1])) for _, point in side_records]
            confs = [float(point[2]) for _, point in side_records]
            for start, end, c0, c1 in zip(pts[:-1], pts[1:], confs[:-1], confs[1:]):
                if c0 >= self.arm_conf_thres and c1 >= self.arm_conf_thres:
                    cv2.line(frame, start, end, (0, 255, 0), 3)
            for point, conf in zip(pts, confs):
                color = (0, 180, 255) if conf >= self.arm_conf_thres else (0, 0, 255)
                cv2.circle(frame, point, 5, color, -1)
        self.draw_status(frame)
        cv2.imshow("YOLO26s Arm V2", frame)
        cv2.waitKey(1)

    def initial_pose_guide_active(self):
        if not self.show_initial_pose_guide:
            return False
        if not self.stable_pose_accepted:
            return True
        if self.pose_read_start_time is None:
            return True
        return time.monotonic() - self.pose_read_start_time < self.pose_read_duration

    def draw_initial_pose_guide(self, frame, records):
        if not self.initial_pose_guide_active():
            return

        height, width = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        roi_left = int(x_min * width)
        roi_right = int(x_max * width)
        roi_top = int(y_min * height)
        roi_bottom = int(y_max * height)
        roi_width = roi_right - roi_left
        roi_height = roi_bottom - roi_top

        left_shoulder = self.record_point(records, "left_shoulder")
        right_shoulder = self.record_point(records, "right_shoulder")
        shoulders = [
            point for point in (left_shoulder, right_shoulder)
            if point is not None and float(point[2]) >= self.arm_conf_thres
        ]
        if shoulders:
            guide_y = int(np.mean([point[1] for point in shoulders]))
        else:
            guide_y = roi_top + int(0.34 * roi_height)
        guide_y = int(np.clip(guide_y, roi_top + 20, roi_bottom - 20))

        center_x = int(0.5 * (roi_left + roi_right))
        if len(shoulders) == 2:
            center_x = int(0.5 * (left_shoulder[0] + right_shoulder[0]))
        center_x = int(np.clip(center_x, roi_left + 40, roi_right - 40))

        line_left = roi_left + int(0.08 * roi_width)
        line_right = roi_right - int(0.08 * roi_width)
        color = (255, 220, 40)
        muted = (120, 130, 60)

        overlay = frame.copy()
        cv2.line(overlay, (line_left, guide_y), (line_right, guide_y), color, 3, cv2.LINE_AA)
        cv2.line(
            overlay,
            (center_x, guide_y - int(0.18 * roi_height)),
            (center_x, guide_y + int(0.22 * roi_height)),
            muted,
            2,
            cv2.LINE_AA,
        )

        left_targets = np.linspace(center_x, line_left, 4)
        right_targets = np.linspace(center_x, line_right, 4)
        for targets in (left_targets, right_targets):
            for index, x_value in enumerate(targets):
                radius = 7 if index == 0 else 5
                cv2.circle(overlay, (int(x_value), guide_y), radius, color, 2, cv2.LINE_AA)

        cv2.putText(
            overlay,
            "initial pose: arms straight",
            (max(12, line_left), max(62, guide_y - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)

    def record_point(self, records, name):
        for record_name, point in records:
            if record_name == name:
                return point
        return None

    def draw_status(self, frame):
        text = self.pose_status_text
        color = (80, 255, 120) if self.pose_status_ok else (0, 220, 255)
        x0, y0 = 12, 28
        box_w = min(frame.shape[1] - 24, 560)
        cv2.rectangle(frame, (6, 6), (6 + box_w, 46), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_center_roi(self, frame):
        if not self.center_roi_enabled:
            return
        height, width = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        cv2.rectangle(
            frame,
            (int(x_min * width), int(y_min * height)),
            (int(x_max * width), int(y_max * height)),
            (80, 220, 255),
            2,
        )

    def destroy_node(self):
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmYolo26sV2()
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
