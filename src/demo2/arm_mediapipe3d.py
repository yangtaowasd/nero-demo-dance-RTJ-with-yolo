#!/usr/bin/env python3

import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


LANDMARK_IDS = (11, 13, 15, 23, 12, 14, 16, 24)


class ArmMediaPipe3D(Node):
    def __init__(self):
        super().__init__("arm_mediapipe3d")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("camera_fourcc", "MJPG")
        self.declare_parameter("output_topic", "/arm_pose_3d")
        self.declare_parameter("model_complexity", 1)
        self.declare_parameter("min_detection_confidence", 0.55)
        self.declare_parameter("min_tracking_confidence", 0.55)
        self.declare_parameter("min_visibility", 0.50)
        self.declare_parameter("show_gui", True)
        self.declare_parameter("mirror_preview", True)

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.camera_fourcc = str(
            self.get_parameter("camera_fourcc").value
        ).strip().upper()
        self.output_topic = self.get_parameter("output_topic").value
        self.min_visibility = float(
            self.get_parameter("min_visibility").value
        )
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.mirror_preview = bool(self.get_parameter("mirror_preview").value)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(
                np.clip(
                    int(self.get_parameter("model_complexity").value), 0, 2
                )
            ),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(
                self.get_parameter("min_detection_confidence").value
            ),
            min_tracking_confidence=float(
                self.get_parameter("min_tracking_confidence").value
            ),
        )
        self.drawing = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.publisher = self.create_publisher(
            Float32MultiArray, self.output_topic, 10
        )
        self.timer = self.create_timer(0.05, self.publish_latest)

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)
        if self.camera_fourcc:
            fourcc = self.camera_fourcc[:4].ljust(4)
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.camera_id}")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_sequence = 0
        self.latest_msg = Float32MultiArray(data=[0.0])
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.infer_loop, daemon=True).start()

        self.get_logger().info(
            f"MediaPipe 3-D pose started: camera={self.camera_id} "
            f"size={self.camera_width}x{self.camera_height} "
            f"output={self.output_topic}"
        )

    def capture_loop(self):
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warning(
                    "failed to capture frame", throttle_duration_sec=1.0
                )
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                self.latest_sequence += 1

    def infer_loop(self):
        processed_sequence = -1
        while self.running and rclpy.ok():
            with self.lock:
                sequence = self.latest_sequence
                frame = (
                    self.latest_frame.copy()
                    if self.latest_frame is not None
                    else None
                )
            if frame is None or sequence == processed_sequence:
                time.sleep(0.002)
                continue
            processed_sequence = sequence

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = self.pose.process(rgb)
                msg = self.make_world_message(result)
                with self.lock:
                    self.latest_msg = msg
                if self.show_gui:
                    self.draw(frame, result, msg.data[0] > 0.0)
            except Exception as exc:
                self.get_logger().error(
                    f"MediaPipe inference error: {exc}",
                    throttle_duration_sec=1.0,
                )
                time.sleep(0.01)

    def make_world_message(self, result):
        world = result.pose_world_landmarks
        if world is None or len(world.landmark) < 25:
            return Float32MultiArray(data=[0.0])

        selected = [world.landmark[index] for index in LANDMARK_IDS]
        if min(float(point.visibility) for point in selected) < self.min_visibility:
            return Float32MultiArray(data=[0.0])

        data = [1.0]
        for point in selected:
            values = (point.x, point.y, point.z, point.visibility)
            if not np.all(np.isfinite(values)):
                return Float32MultiArray(data=[0.0])
            data.extend(float(value) for value in values)
        return Float32MultiArray(data=data)

    def draw(self, frame, result, valid):
        if result.pose_landmarks is not None:
            self.drawing.draw_landmarks(
                frame, result.pose_landmarks, self.pose_connections
            )
        color = (70, 230, 100) if valid else (0, 180, 255)
        text = "3D pose ready" if valid else "show shoulders, elbows, wrists, hips"
        cv2.putText(
            frame,
            text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        preview = cv2.flip(frame, 1) if self.mirror_preview else frame
        cv2.imshow("MediaPipe 3D Arm Pose", preview)
        cv2.waitKey(1)

    def publish_latest(self):
        with self.lock:
            msg = self.latest_msg
        self.publisher.publish(msg)

    def destroy_node(self):
        self.running = False
        self.cap.release()
        self.pose.close()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmMediaPipe3D()
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
