#!/usr/bin/env python3

import os
import threading
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def unit(vector, fallback=None):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


class ArmMediaPipeV2(Node):
    def __init__(self):
        super().__init__("arm_mediapipe_v2")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_width", 1280)
        self.declare_parameter("camera_height", 720)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("camera_fourcc", "MJPG")
        self.declare_parameter("use_camera_topic", False)
        self.declare_parameter("use_realsense", True)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/arm_pose_v2")
        self.declare_parameter("image_points_topic", "/arm_pose_v2/image_points")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("show_coords", True)
        self.declare_parameter("model_complexity", 1)
        self.declare_parameter("world_scale", 1.0)
        self.declare_parameter("z_sign", -1.0)
        self.declare_parameter("depth_sample_radius", 2)
        self.declare_parameter("upper_arm_length", 0.30)
        self.declare_parameter("forearm_length", 0.26)
        self.declare_parameter("hand_length", 0.10)
        self.declare_parameter("center_roi_enabled", False)
        self.declare_parameter("center_roi_fraction", 0.80)
        self.declare_parameter("require_all_arm_points_in_roi", True)

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.camera_fourcc = str(self.get_parameter("camera_fourcc").value).strip().upper()
        self.use_camera_topic = bool(self.get_parameter("use_camera_topic").value)
        self.use_realsense = bool(self.get_parameter("use_realsense").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.image_points_topic = self.get_parameter("image_points_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.show_coords = bool(self.get_parameter("show_coords").value)
        self.world_scale = float(self.get_parameter("world_scale").value)
        self.z_sign = float(self.get_parameter("z_sign").value)
        self.depth_sample_radius = int(self.get_parameter("depth_sample_radius").value)
        self.upper_arm_length = float(self.get_parameter("upper_arm_length").value)
        self.forearm_length = float(self.get_parameter("forearm_length").value)
        self.hand_length = float(self.get_parameter("hand_length").value)
        self.center_roi_enabled = bool(
            self.get_parameter("center_roi_enabled").value
        )
        self.center_roi_fraction = float(
            self.get_parameter("center_roi_fraction").value
        )
        self.require_all_arm_points_in_roi = bool(
            self.get_parameter("require_all_arm_points_in_roi").value
        )

        self.arm_image_points = (
            ("left_shoulder", (11,)),
            ("left_elbow", (13,)),
            ("left_wrist", (15,)),
            ("left_hand", (17, 19, 21)),
            ("right_shoulder", (12,)),
            ("right_elbow", (14,)),
            ("right_wrist", (16,)),
            ("right_hand", (18, 20, 22)),
        )

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(self.get_parameter("model_complexity").value),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_depth = None
        self.depth_intrinsics = None
        self.latest_msg = None
        self.latest_image_points_msg = None
        self.running = True

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.image_points_pub = self.create_publisher(
            Float32MultiArray, self.image_points_topic, 10
        )
        self.timer = self.create_timer(0.05, self.publish_latest)

        if self.use_realsense and not self.use_camera_topic:
            self.start_realsense()
        elif self.use_camera_topic:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image

            self.bridge = CvBridge()
            self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.camera_fourcc:
                fourcc = self.camera_fourcc[:4].ljust(4)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            self.log_camera_settings()
            threading.Thread(target=self.capture_loop, daemon=True).start()

        threading.Thread(target=self.infer_loop, daemon=True).start()
        self.get_logger().info(
            f"arm_mediapipe_v2 started: output={self.output_topic} "
            f"image_points={self.image_points_topic} "
            f"use_realsense={self.use_realsense} "
            f"center_roi_enabled={self.center_roi_enabled} "
            f"center_roi_fraction={self.center_roi_fraction:.2f} "
            f"require_all_arm_points_in_roi={self.require_all_arm_points_in_roi}"
        )

    def log_camera_settings(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        self.get_logger().info(
            f"camera requested: id={self.camera_id}, fourcc={self.camera_fourcc}, "
            f"size={self.camera_width}x{self.camera_height}, fps={self.camera_fps:.1f}"
        )
        self.get_logger().info(
            f"camera actual: fourcc={fourcc}, size={width}x{height}, fps={fps:.1f}"
        )

    def mp_point(self, world, idx):
        lm = world[idx]
        return self.world_scale * np.asarray(
            [float(lm.y), float(lm.x), self.z_sign * float(lm.z)],
            dtype=float,
        )

    def start_realsense(self):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("use_realsense=true but pyrealsense2 is not available") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        threading.Thread(target=self.realsense_loop, daemon=True).start()
        self.get_logger().info("RealSense color+depth capture started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_frame = frame
        except Exception as exc:
            self.get_logger().warning(f"image callback failed: {exc}")

    def capture_loop(self):
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.002)

    def realsense_loop(self):
        while self.running and rclpy.ok():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            with self.lock:
                self.latest_frame = frame
                self.latest_depth = depth_frame
                self.depth_intrinsics = intrinsics

    def infer_loop(self):
        while self.running and rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                depth = self.latest_depth
                intrinsics = self.depth_intrinsics
            if frame is None:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.pose.process(rgb)
            msg = self.make_msg(result, frame.shape[:2], depth, intrinsics)
            image_points_msg = self.make_image_points_msg(result, frame.shape[:2])
            with self.lock:
                self.latest_msg = msg
                self.latest_image_points_msg = image_points_msg
            if self.show_gui:
                self.draw(frame, result, msg)

    def normalize_chain(self, shoulder, elbow, wrist, hand):
        upper = unit(elbow - shoulder)
        elbow = shoulder + upper * self.upper_arm_length
        forearm = unit(wrist - elbow, upper)
        wrist = elbow + forearm * self.forearm_length
        hand_dir = unit(hand - wrist, forearm)
        hand = wrist + hand_dir * self.hand_length
        return shoulder, elbow, wrist, hand

    def depth_point(self, landmarks, world, idx, image_shape, depth, intrinsics):
        if depth is None or intrinsics is None:
            return self.mp_point(world, idx)
        height, width = image_shape
        lm = landmarks[idx]
        x = int(np.clip(float(lm.x) * width, 0, width - 1))
        y = int(np.clip(float(lm.y) * height, 0, height - 1))
        dist = self.sample_depth(depth, x, y, width, height)
        if dist <= 0.0:
            return self.mp_point(world, idx)
        point = self.rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], dist)
        return self.world_scale * np.asarray(
            [float(point[1]), float(point[0]), self.z_sign * float(point[2])],
            dtype=float,
        )

    def sample_depth(self, depth, x, y, width, height):
        values = []
        radius = max(self.depth_sample_radius, 0)
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                value = float(depth.get_distance(xx, yy))
                if value > 0.0:
                    values.append(value)
        if not values:
            return 0.0
        return float(np.median(values))

    def side_block(self, world, landmarks, ids, image_shape, depth, intrinsics):
        shoulder = self.depth_point(landmarks, world, ids["shoulder"], image_shape, depth, intrinsics)
        elbow = self.depth_point(landmarks, world, ids["elbow"], image_shape, depth, intrinsics)
        wrist = self.depth_point(landmarks, world, ids["wrist"], image_shape, depth, intrinsics)
        index = self.depth_point(landmarks, world, ids["index"], image_shape, depth, intrinsics)
        pinky = self.depth_point(landmarks, world, ids["pinky"], image_shape, depth, intrinsics)
        thumb = self.depth_point(landmarks, world, ids["thumb"], image_shape, depth, intrinsics)
        hand = (index + pinky + thumb) / 3.0
        shoulder, elbow, wrist, hand = self.normalize_chain(shoulder, elbow, wrist, hand)

        hand_x = unit(index - wrist, hand - wrist)
        palm_normal = unit(np.cross(index - wrist, pinky - wrist), [0.0, 0.0, 1.0])
        values = []
        for vec in (shoulder, elbow, wrist, hand, palm_normal, hand_x):
            values.extend(vec.astype(float).tolist())
        return values

    def make_msg(self, result, image_shape, depth=None, intrinsics=None):
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return Float32MultiArray(data=[0.0])
        if not self.pose_in_center_roi(result):
            return Float32MultiArray(data=[0.0])
        if not self.arm_points_in_center_roi(result):
            return Float32MultiArray(data=[0.0])

        world = result.pose_world_landmarks.landmark
        landmarks = result.pose_landmarks.landmark
        left_ids = {"shoulder": 11, "elbow": 13, "wrist": 15, "pinky": 17, "index": 19, "thumb": 21}
        right_ids = {"shoulder": 12, "elbow": 14, "wrist": 16, "pinky": 18, "index": 20, "thumb": 22}
        data = [1.0]
        data.extend(self.side_block(world, landmarks, left_ids, image_shape, depth, intrinsics))
        data.extend(self.side_block(world, landmarks, right_ids, image_shape, depth, intrinsics))
        return Float32MultiArray(data=data)

    def make_image_points_msg(self, result, image_shape):
        if not result.pose_landmarks:
            return Float32MultiArray(data=[0.0])
        if not self.arm_points_in_center_roi(result):
            return Float32MultiArray(data=[0.0])

        height, width = image_shape
        data = [1.0, float(width), float(height)]
        for _, x_norm, y_norm, visibility in self.arm_image_point_records(result):
            x_px = float(np.clip(x_norm, 0.0, 1.0) * width)
            y_px = float(np.clip(y_norm, 0.0, 1.0) * height)
            data.extend([x_px, y_px, x_norm, y_norm, visibility])
        return Float32MultiArray(data=data)

    def arm_image_point_records(self, result):
        if not result.pose_landmarks:
            return []

        landmarks = result.pose_landmarks.landmark
        records = []
        for name, mp_ids in self.arm_image_points:
            xs = []
            ys = []
            visibilities = []
            for mp_id in mp_ids:
                landmark = landmarks[mp_id]
                xs.append(float(landmark.x))
                ys.append(float(landmark.y))
                visibilities.append(float(getattr(landmark, "visibility", 0.0)))
            records.append((
                name,
                float(np.mean(xs)),
                float(np.mean(ys)),
                float(min(visibilities)),
            ))
        return records

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
            if left.visibility >= 0.25 and right.visibility >= 0.25:
                candidates.append((
                    0.5 * (float(left.x) + float(right.x)),
                    0.5 * (float(left.y) + float(right.y)),
                ))
        if not candidates:
            for mp_id in (11, 13, 15, 12, 14, 16):
                landmark = landmarks[mp_id]
                if landmark.visibility >= 0.25:
                    candidates.append((float(landmark.x), float(landmark.y)))
        if not candidates:
            return False

        center = np.mean(np.asarray(candidates, dtype=float), axis=0)
        return self.point_in_center_roi(float(center[0]), float(center[1]))

    def arm_points_in_center_roi(self, result):
        if not self.center_roi_enabled or not self.require_all_arm_points_in_roi:
            return True
        if not result.pose_landmarks:
            return False

        for _, x, y, visibility in self.arm_image_point_records(result):
            if visibility < 0.25:
                return False
            if not self.point_in_center_roi(x, y):
                return False
        return True

    def publish_latest(self):
        with self.lock:
            msg = self.latest_msg
            image_points_msg = self.latest_image_points_msg
        if msg is not None:
            self.pub.publish(msg)
        if image_points_msg is not None:
            self.image_points_pub.publish(image_points_msg)

    def draw(self, frame, result, msg=None):
        self.draw_center_roi(frame)
        if result.pose_landmarks:
            height, width = frame.shape[:2]
            records = self.arm_image_point_records(result)
            for side_slice in (records[:4], records[4:8]):
                pts = [
                    (
                        int(np.clip(x, 0.0, 1.0) * width),
                        int(np.clip(y, 0.0, 1.0) * height),
                    )
                    for _, x, y, _ in side_slice
                ]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, (0, 255, 0), 3)
                for pt in pts:
                    cv2.circle(frame, pt, 5, (0, 180, 255), -1)
        if self.show_coords and msg is not None:
            self.draw_coords(frame, msg)
        cv2.imshow("Arm MediaPipe V2", frame)
        cv2.waitKey(1)

    def draw_center_roi(self, frame):
        if not self.center_roi_enabled:
            return
        height, width = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.roi_bounds_normalized()
        p1 = (int(x_min * width), int(y_min * height))
        p2 = (int(x_max * width), int(y_max * height))
        cv2.rectangle(frame, p1, p2, (80, 220, 255), 2)

    def draw_coords(self, frame, msg):
        data = list(msg.data)
        if len(data) != 37 or int(data[0]) <= 0:
            cv2.putText(frame, "pose: not detected", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return

        lines = ["coords: shoulder elbow wrist hand (m)"]
        labels = ("S", "E", "W", "H")
        for side, start in (("L", 1), ("R", 19)):
            block = data[start:start + 12]
            for i, label in enumerate(labels):
                x, y, z = block[i * 3:i * 3 + 3]
                lines.append(f"{side}{label}: {x:+.2f} {y:+.2f} {z:+.2f}")

        x0, y0 = 12, 24
        line_h = 18
        box_w = 300
        box_h = line_h * len(lines) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + box_w, 6 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(lines):
            color = (230, 230, 230) if i == 0 else (80, 255, 180)
            cv2.putText(frame, line, (x0, y0 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def destroy_node(self):
        self.running = False
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
        if hasattr(self, "cap"):
            self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        self.pose.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmMediaPipeV2()
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
    
