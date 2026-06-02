#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import threading
from collections import deque
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment  # 用于匈牙利匹配
import time


class SimplePoseTracker:
    """基于关键点距离的简单人体跟踪器"""
    def __init__(self, max_age=5, dist_thresh=50):
        self.next_id = 0
        self.tracks = {}          # track_id -> {'kpts': (17,3), 'age': int}
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, keypoints_list):
        """传入当前帧检测到的多人关键点列表 [(17,3) ndarray]，
           返回 [(track_id, kpts)] 的列表"""
        if not keypoints_list:
            # 没有检测到人，所有已有track年龄+1
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return []

        # 提取所有track的最近关键点
        track_ids = list(self.tracks.keys())
        if not track_ids:
            # 首次出现，全部分配新ID
            assigned = []
            for kpts in keypoints_list:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'kpts': kpts.copy(), 'age': 0}
                assigned.append((tid, kpts))
            return assigned

        track_kpts = np.array([self.tracks[tid]['kpts'] for tid in track_ids])  # (N_tracks, 17, 3)
        det_kpts = np.array(keypoints_list)  # (N_dets, 17, 3)

        # 计算距离矩阵：基于所有可见关键点的平均欧氏距离
        cost_matrix = np.zeros((len(track_ids), len(det_kpts)))
        for i, tk in enumerate(track_kpts):
            for j, dk in enumerate(det_kpts):
                # 只考虑置信度>0.5的点
                mask = (tk[:, 2] > 0.6) & (dk[:, 2] > 0.6)
                if np.any(mask):
                    cost_matrix[i, j] = np.mean(np.linalg.norm(tk[mask, :2] - dk[mask, :2], axis=1))
                else:
                    cost_matrix[i, j] = 1e6  # 无可见点，代价极高
        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned = []
        used_track_ids = set()
        used_det_ids = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_thresh:
                tid = track_ids[r]
                self.tracks[tid]['kpts'] = det_kpts[c].copy()
                self.tracks[tid]['age'] = 0
                assigned.append((tid, det_kpts[c]))
                used_track_ids.add(tid)
                used_det_ids.add(c)

        # 未匹配的track增加年龄，超限删除
        for tid in track_ids:
            if tid not in used_track_ids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        # 未匹配的检测，创建新track
        for j in range(len(det_kpts)):
            if j not in used_det_ids:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'kpts': det_kpts[j].copy(), 'age': 0}
                assigned.append((tid, det_kpts[j]))

        return assigned


class BodyPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_body_pose_node')

        # ----- 参数声明 -----
        self.declare_parameter('model_path', '/home/yang/demo_ws/src/demo2/model/yolo26s-pose.pt')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('arm_conf_thres', 0.7)
        self.declare_parameter('use_camera_topic', False)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('body_pose', '/body_pose')
        self.declare_parameter('show_gui', True)
        self.declare_parameter('history_size', 5)   # 历史帧数
        self.declare_parameter('real_shoulder_width', 0.40)  # 米，一般成人肩宽估计值
        self.declare_parameter('real_upper_arm_length', 0.25)  # 米，肩到肘
        self.declare_parameter('real_forearm_length', 0.20)  # 米，肘到腕
        self.declare_parameter('real_torso_length', 0.45)  # 米，肩到髋
        self.declare_parameter('real_thigh_length', 0.42)  # 米，髋到膝
        self.declare_parameter('real_lower_leg_length', 0.43)  # 米，膝到踝
        self.declare_parameter('real_head_length', 0.25)  # 米，肩中心到头部关键点
        self.declare_parameter('upper_body_same_depth', True)  # 5~12 默认按同一身体平面估计
        self.declare_parameter('arm_depth_mode', 'near')  # near: 肩->肘->腕逐段更靠近相机
        self.declare_parameter('camera_fx', 0.0)  # 像素，<=0 时自动估计
        self.declare_parameter('camera_fy', 0.0)  # 像素，<=0 时自动估计
        self.declare_parameter('camera_cx', 0.0)  # 像素，<=0 时使用图像中心
        self.declare_parameter('camera_cy', 0.0)  # 像素，<=0 时使用图像中心
        self.declare_parameter('min_depth', 0.2)  # 米
        self.declare_parameter('max_depth', 3.0)  # 米
        self.declare_parameter('depth_smoothing_alpha', 0.2)
        self.declare_parameter('max_depth_jump', 0.25)  # 米，单帧深度跳变超过此值则沿用上一帧
        self.declare_parameter('point_depth_smoothing_alpha', 0.35)
        self.declare_parameter('max_point_depth_jump', 0.12)  # 米，肘/腕单帧深度跳变限制

        model_path = self.get_parameter('model_path').value
        self.camera_id = self.get_parameter('camera_id').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.arm_conf_thres = self.get_parameter('arm_conf_thres').value
        self.use_camera_topic = self.get_parameter('use_camera_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.body_pose_topic = self.get_parameter('body_pose').value
        self.show_gui = self.get_parameter('show_gui').value
        self.history_size = self.get_parameter('history_size').value
        self.real_shoulder_width = self.get_parameter('real_shoulder_width').value
        self.real_upper_arm_length = float(self.get_parameter('real_upper_arm_length').value)
        self.real_forearm_length = float(self.get_parameter('real_forearm_length').value)
        self.real_torso_length = float(self.get_parameter('real_torso_length').value)
        self.real_thigh_length = float(self.get_parameter('real_thigh_length').value)
        self.real_lower_leg_length = float(self.get_parameter('real_lower_leg_length').value)
        self.real_head_length = float(self.get_parameter('real_head_length').value)
        self.upper_body_same_depth = bool(self.get_parameter('upper_body_same_depth').value)
        self.arm_depth_mode = str(self.get_parameter('arm_depth_mode').value)
        self.camera_fx = float(self.get_parameter('camera_fx').value)
        self.camera_fy = float(self.get_parameter('camera_fy').value)
        self.camera_cx = float(self.get_parameter('camera_cx').value)
        self.camera_cy = float(self.get_parameter('camera_cy').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.depth_smoothing_alpha = float(self.get_parameter('depth_smoothing_alpha').value)
        self.max_depth_jump = float(self.get_parameter('max_depth_jump').value)
        self.point_depth_smoothing_alpha = float(self.get_parameter('point_depth_smoothing_alpha').value)
        self.max_point_depth_jump = float(self.get_parameter('max_point_depth_jump').value)

        # ----- 加载模型 -----
        self.get_logger().info(f"Loading model from {model_path}")
        self.model = YOLO(model_path,task='pose')

        # ----- 身体关键点定义 -----
        self.BODY_PARTS = list(range(0, 17))
        self.BODY_SKELETON = [
            # (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            # (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        # 需要平滑的关键点索引（肘和腕）
        self.SMOOTH_INDICES = {7, 8, 9, 10}

        # ----- 跟踪器与历史缓存 -----
        self.tracker = SimplePoseTracker(max_age=5, dist_thresh=50)
        self.lock = threading.Lock()
        self.latest_frame = None  
        self.latest_tracks = []              # 当前帧带ID的跟踪结果 [(tid, kpts_17x3)]
        self.histories = {}                  # tid -> deque of body kpts (17, 3) maxlen=self.history_size
        self.running = True

        # ----- 发布器与定时器 -----
        self.pose_pub = self.create_publisher(Float32MultiArray, self.body_pose_topic, 10)
        self.timer = self.create_timer(0.05, self.publish_callback)

        # EMA 平滑器存储: key = (track_id, body_part_index)
        self.ema_smoothers = {}       # 存储平滑器实例
        self.prev_raw_kpts = {}       # key: (tid, i) -> 上一帧原始坐标 (x,y)
        self.last_depth = {}          # track_id -> 上一次可用深度 z
        self.last_point_depths = {}   # (track_id, keypoint_id) -> 上一次可用深度 z
        self.alpha_min = 0.2
        self.alpha_max = 0.6
        self.alpha_slope = 0.01      # 位移到 alpha 的斜率

        time.sleep(0.5)

        # ----- 图像获取 -----
        if self.use_camera_topic:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback, 10)
            self.get_logger().info(f"Subscribed to {self.image_topic}")
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 宽度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                self.get_logger().error(f"Cannot open camera {self.camera_id}")
                raise RuntimeError("Camera open failed")
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            self.get_logger().info(f"Direct camera capture started (id={self.camera_id})")

        # 推理线程
        self.infer_thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_thread.start()

        self.get_logger().info("BodyPoseNode with history smoothing started")

    # ---------- 图像获取 ----------
    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            with self.lock:
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def capture_loop(self):
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                self.get_logger().warn("Failed to capture frame", throttle_duration_sec=1)
        if self.cap:
            self.cap.release()

    # ---------- 推理循环 ----------
    def infer_loop(self):
        while self.running and rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                try:
                    results = self.model(frame, conf=self.conf_thres,max_det=1,verbose=False)[0]
                    keypoints_list = []
                    if results.keypoints is not None:
                        for person in results.keypoints.data:
                            keypoints_list.append(person.cpu().numpy())
                    # 跟踪更新
                    tracked = self.tracker.update(keypoints_list)
                    with self.lock:
                        self.latest_tracks = tracked
                    # 更新历史队列
                    for tid, kpts in tracked:
                        body_kpts = kpts[self.BODY_PARTS, :].copy()  # shape (17,3)
                        if tid not in self.histories:
                            self.histories[tid] = deque(maxlen=self.history_size)
                        self.histories[tid].append(body_kpts)
                    # 清理已经消失的track的历史
                    active_ids = {t[0] for t in tracked}
                    for tid in list(self.histories.keys()):
                        if tid not in active_ids:
                            del self.histories[tid]
                            self.last_depth.pop(tid, None)
                            for point_key in list(self.last_point_depths.keys()):
                                if point_key[0] == tid:
                                    del self.last_point_depths[point_key]

                    # 可选可视化
                    if self.show_gui:
                        self.visualize(frame, tracked)
                except Exception as e:
                    self.get_logger().error(f"Inference error: {e}")

    # ---------- 可视化 ----------
    def visualize(self, frame, tracked):
        for tid, kpts in tracked:
            xyz = self.keypoints_to_xyz(tid, kpts, frame.shape)
            left_arm_ok = all(kpts[i][2] > self.arm_conf_thres for i in [5, 7, 9])
            right_arm_ok = all(kpts[i][2] > self.arm_conf_thres for i in [6, 8, 10])
            for i in self.BODY_PARTS:
                x, y, conf = kpts[i]
                if conf < 0.5: continue
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                self.draw_xyz_label(frame, i, int(x), int(y), xyz[i])
            for start, end in self.BODY_SKELETON:
                if (start, end) in [(5, 7), (7, 9)] and not left_arm_ok: continue
                if (start, end) in [(6, 8), (8, 10)] and not right_arm_ok: continue
                if kpts[start][2] > 0.5 and kpts[end][2] > 0.5:
                    pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                    pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            for start, end in [(5, 7), (7, 9), (6, 8), (8, 10)]:
                self.draw_segment_distance(frame, kpts, xyz, start, end)
            depth = xyz[5, 2] if len(xyz) > 5 else float('nan')
            depth_text = f"ID {tid} z={depth:.2f}m" if np.isfinite(depth) else f"ID {tid} z=nan"
            cv2.putText(frame, depth_text, (10, 30 + tid * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('YOLO Pose', frame)
        cv2.waitKey(1)

    def draw_xyz_label(self, frame, keypoint_id, px, py, xyz):
        if not all(np.isfinite(v) for v in xyz):
            text = f"{keypoint_id}: nan"
        else:
            text = f"{keypoint_id}: {xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}"

        x = min(max(px + 6, 0), frame.shape[1] - 1)
        y = min(max(py - 6, 12), frame.shape[0] - 1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_segment_distance(self, frame, kpts, xyz, start, end):
        if kpts[start, 2] <= 0.5 or kpts[end, 2] <= 0.5:
            return
        if not (np.all(np.isfinite(xyz[start])) and np.all(np.isfinite(xyz[end]))):
            return

        dist = np.linalg.norm(xyz[start] - xyz[end])
        mx = int((kpts[start, 0] + kpts[end, 0]) * 0.5)
        my = int((kpts[start, 1] + kpts[end, 1]) * 0.5)
        text = f"{start}-{end}:{dist:.2f}m"
        cv2.putText(frame, text, (mx + 4, my + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (mx + 4, my + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # ---------- 发布回调（20Hz）----------
    def publish_callback(self):
        with self.lock:
            # 深拷贝当前跟踪结果
            tracks_snapshot = [(tid, kpts.copy()) for tid, kpts in self.latest_tracks]
            histories_snapshot = {tid: list(h) for tid, h in self.histories.items()}
            frame_shape = self.latest_frame.shape if self.latest_frame is not None else None

        data = []
        data.append(float(len(tracks_snapshot)))
        for tid, kpts_17 in tracks_snapshot:
            # 获取该人的历史（17个身体点）
            history = histories_snapshot.get(tid, [])
            # 对当前帧的身体点先做手臂完整性判断，不可信点用历史填充
            body_kpts = kpts_17[self.BODY_PARTS, :].copy()  # (17,3)
            body_kpts = self.smooth_with_history(body_kpts, history, self.arm_conf_thres)
            # === 自适应 alpha EMA 平滑 ===
            smoothed = np.zeros_like(body_kpts)
            for i in range(len(body_kpts)):
                key = (tid, i)
                
                # 获取平滑器
                if key not in self.ema_smoothers:
                    self.ema_smoothers[key] = AdaptiveEMAFilter(alpha_min=self.alpha_min,
                                                                alpha_max=self.alpha_max)
                    self.prev_raw_kpts[key] = body_kpts[i, :2].copy()  # 初始化上一次原始位置
                
                current_xy = body_kpts[i, :2]
                prev_xy = self.prev_raw_kpts[key]
                
                # 计算像素位移（欧氏距离）
                dist = np.linalg.norm(current_xy - prev_xy)
                self.prev_raw_kpts[key] = current_xy.copy()  # 更新记录
                
                # 映射距离到 alpha（线性映射 + 钳制）
                alpha = self.alpha_min + dist * self.alpha_slope
                alpha = max(self.alpha_min, min(self.alpha_max, alpha))
                
                # 可选：置信度太低时不更新平滑器，保持上一帧平滑值
                if body_kpts[i, 2] > 0.5:  # 置信度阈值
                    smoothed[i, :2] = self.ema_smoothers[key].update(current_xy, alpha)
                else:
                    # 低置信度：输出平滑器的当前值（不更新）
                    smoothed[i, :2] = self.ema_smoothers[key].value if self.ema_smoothers[key].value is not None else current_xy
                
                smoothed[i, 2] = body_kpts[i, 2]  # 置信度保持不变
            xyz = self.keypoints_to_xyz(tid, smoothed, frame_shape)
            data.extend(xyz.flatten().tolist())
        msg = Float32MultiArray(data=data)
        self.pose_pub.publish(msg)

    def get_camera_intrinsics(self, frame_shape):
        if frame_shape is None:
            width = 1280.0
            height = 720.0
        else:
            height = float(frame_shape[0])
            width = float(frame_shape[1])

        # 没有标定参数时用图像宽度近似焦距，至少保证 z 能由像素比例计算出来。
        fx = self.camera_fx if self.camera_fx > 0.0 else width
        fy = self.camera_fy if self.camera_fy > 0.0 else fx
        cx = self.camera_cx if self.camera_cx > 0.0 else width / 2.0
        cy = self.camera_cy if self.camera_cy > 0.0 else height / 2.0
        return fx, fy, cx, cy

    def estimate_depth_from_shoulders(self, tid, kpts, fx):
        left_shoulder = kpts[5]
        right_shoulder = kpts[6]
        shoulders_ok = (
            left_shoulder[2] > self.arm_conf_thres and
            right_shoulder[2] > self.arm_conf_thres
        )

        if shoulders_ok:
            pixel_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            if pixel_width > 1.0:
                raw_z = fx * self.real_shoulder_width / pixel_width
                raw_z = float(np.clip(raw_z, self.min_depth, self.max_depth))

                last_z = self.last_depth.get(tid)
                if last_z is not None and np.isfinite(last_z):
                    if abs(raw_z - last_z) > self.max_depth_jump:
                        z = last_z
                    else:
                        alpha = np.clip(self.depth_smoothing_alpha, 0.0, 1.0)
                        z = alpha * raw_z + (1.0 - alpha) * last_z
                else:
                    z = raw_z

                self.last_depth[tid] = z
                return z

        return self.last_depth.get(tid, float('nan'))

    def keypoints_to_xyz(self, tid, kpts, frame_shape):
        fx, fy, cx, cy = self.get_camera_intrinsics(frame_shape)
        base_z = self.estimate_depth_from_shoulders(tid, kpts, fx)

        xyz = np.full((len(kpts), 3), np.nan, dtype=float)
        if not np.isfinite(base_z):
            return xyz

        # 先给所有可见点一个基准深度，后面再用骨架长度逐点修正各自的 z。
        for i in range(len(kpts)):
            if kpts[i, 2] > 0.5:
                xyz[i] = self.pixel_to_xyz(kpts[i, 0], kpts[i, 1], base_z, fx, fy, cx, cy)

        for shoulder_id in (5, 6):
            if kpts[shoulder_id, 2] > 0.5:
                xyz[shoulder_id] = self.pixel_to_xyz(
                    kpts[shoulder_id, 0], kpts[shoulder_id, 1], base_z, fx, fy, cx, cy
                )
                self.last_point_depths[(tid, shoulder_id)] = base_z

        if self.upper_body_same_depth:
            for point_id in (5, 6, 11, 12):
                if kpts[point_id, 2] > 0.5:
                    xyz[point_id] = self.pixel_to_xyz(
                        kpts[point_id, 0], kpts[point_id, 1], base_z, fx, fy, cx, cy
                    )
                    self.last_point_depths[(tid, point_id)] = base_z
            arm_edges = [
                (5, 7, self.real_upper_arm_length),
                (7, 9, self.real_forearm_length),
                (6, 8, self.real_upper_arm_length),
                (8, 10, self.real_forearm_length),
            ]
            if self.arm_depth_mode != 'flat':
                for parent_id, child_id, length_m in arm_edges:
                    self.solve_keypoint_depth(tid, kpts, xyz, parent_id, child_id,
                                              length_m, fx, fy, cx, cy, base_z,
                                              depth_mode=self.arm_depth_mode)

            segment_edges = [
                (11, 13, self.real_thigh_length),
                (13, 15, self.real_lower_leg_length),
                (12, 14, self.real_thigh_length),
                (14, 16, self.real_lower_leg_length),
            ]
        else:
            segment_edges = [
                (5, 7, self.real_upper_arm_length),
                (7, 9, self.real_forearm_length),
                (6, 8, self.real_upper_arm_length),
                (8, 10, self.real_forearm_length),
                (5, 11, self.real_torso_length),
                (6, 12, self.real_torso_length),
                (11, 13, self.real_thigh_length),
                (13, 15, self.real_lower_leg_length),
                (12, 14, self.real_thigh_length),
                (14, 16, self.real_lower_leg_length),
            ]

        for parent_id, child_id, length_m in segment_edges:
            self.solve_keypoint_depth(tid, kpts, xyz, parent_id, child_id,
                                      length_m, fx, fy, cx, cy, base_z)

        self.solve_head_depths(tid, kpts, xyz, fx, fy, cx, cy, base_z)
        return xyz

    def pixel_to_ray(self, u, v, fx, fy, cx, cy):
        return np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)

    def pixel_to_xyz(self, u, v, z, fx, fy, cx, cy):
        ray = self.pixel_to_ray(u, v, fx, fy, cx, cy)
        return ray * z

    def solve_depth_from_parent(self, parent_xyz, child_uv, length_m,
                                fx, fy, cx, cy, prefer_z, depth_mode='previous'):
        ray = self.pixel_to_ray(child_uv[0], child_uv[1], fx, fy, cx, cy)
        a = float(np.dot(ray, ray))
        b = float(-2.0 * np.dot(ray, parent_xyz))
        c = float(np.dot(parent_xyz, parent_xyz) - length_m ** 2)
        disc = b ** 2 - 4.0 * a * c
        if disc < 0.0:
            return float(np.clip(prefer_z, self.min_depth, self.max_depth))

        sqrt_disc = np.sqrt(disc)
        candidates = [
            (-b + sqrt_disc) / (2.0 * a),
            (-b - sqrt_disc) / (2.0 * a),
        ]
        candidates = [
            float(z) for z in candidates
            if np.isfinite(z) and self.min_depth <= z <= self.max_depth
        ]
        if not candidates:
            return float(np.clip(prefer_z, self.min_depth, self.max_depth))

        if depth_mode == 'near':
            return min(candidates)
        if depth_mode == 'far':
            return max(candidates)
        if depth_mode == 'projected':
            expected_z = self.estimate_projected_depth(parent_xyz, child_uv, length_m,
                                                       fx, fy, cx, cy, prefer_z)
            return min(candidates, key=lambda z: abs(z - expected_z))
        return min(candidates, key=lambda z: abs(z - prefer_z))

    def solve_keypoint_depth(self, tid, kpts, xyz, parent_id, child_id,
                             length_m, fx, fy, cx, cy, fallback_z, depth_mode='previous'):
        if child_id >= len(kpts) or parent_id >= len(kpts):
            return
        if kpts[child_id, 2] <= 0.5 or not np.all(np.isfinite(xyz[parent_id])):
            return

        previous_z = self.last_point_depths.get((tid, child_id))
        prefer_z = previous_z if previous_z is not None else xyz[parent_id, 2]
        z = self.solve_depth_from_parent(
            xyz[parent_id],
            kpts[child_id, :2],
            length_m,
            fx, fy, cx, cy,
            prefer_z if np.isfinite(prefer_z) else fallback_z,
            depth_mode=depth_mode
        )

        if previous_z is not None and np.isfinite(previous_z):
            if abs(z - previous_z) > self.max_point_depth_jump:
                z = previous_z + np.sign(z - previous_z) * self.max_point_depth_jump
            alpha = np.clip(self.point_depth_smoothing_alpha, 0.0, 1.0)
            z = alpha * z + (1.0 - alpha) * previous_z

        xyz[child_id] = self.pixel_to_xyz(kpts[child_id, 0], kpts[child_id, 1], z, fx, fy, cx, cy)
        self.last_point_depths[(tid, child_id)] = z

    def estimate_projected_depth(self, parent_xyz, child_uv, length_m,
                                 fx, fy, cx, cy, prefer_z):
        same_depth_child = self.pixel_to_xyz(child_uv[0], child_uv[1],
                                             parent_xyz[2], fx, fy, cx, cy)
        same_depth_dist = np.linalg.norm(same_depth_child[:2] - parent_xyz[:2])
        if same_depth_dist >= length_m * 0.95:
            return parent_xyz[2]

        dz = np.sqrt(max(length_m ** 2 - same_depth_dist ** 2, 0.0))
        if self.arm_depth_mode == 'far':
            return parent_xyz[2] + dz
        if self.arm_depth_mode == 'previous':
            near_z = parent_xyz[2] - dz
            far_z = parent_xyz[2] + dz
            return near_z if abs(near_z - prefer_z) <= abs(far_z - prefer_z) else far_z
        return parent_xyz[2] - dz

    def solve_head_depths(self, tid, kpts, xyz, fx, fy, cx, cy, fallback_z):
        if not (np.all(np.isfinite(xyz[5])) and np.all(np.isfinite(xyz[6]))):
            return

        shoulder_mid = (xyz[5] + xyz[6]) * 0.5
        for point_id in (0, 1, 2, 3, 4):
            if kpts[point_id, 2] <= 0.5:
                continue
            prefer_z = self.last_point_depths.get((tid, point_id), shoulder_mid[2])
            z = self.solve_depth_from_parent(
                shoulder_mid,
                kpts[point_id, :2],
                self.real_head_length,
                fx, fy, cx, cy,
                prefer_z if np.isfinite(prefer_z) else fallback_z
            )
            xyz[point_id] = self.pixel_to_xyz(kpts[point_id, 0], kpts[point_id, 1], z, fx, fy, cx, cy)
            self.last_point_depths[(tid, point_id)] = z

    def smooth_with_history(self, current_body, history, arm_thresh):
        # current_body 现在索引 = 关键点 ID
        left_arm_ok = all(current_body[i, 2] > arm_thresh for i in [5, 7, 9])
        right_arm_ok = all(current_body[i, 2] > arm_thresh for i in [6, 8, 10])

        smoothed = current_body.copy()
        if not left_arm_ok:
            smoothed[7] = self.get_historical_point(7, history, arm_thresh)
            smoothed[9] = self.get_historical_point(9, history, arm_thresh)
        if not right_arm_ok:
            smoothed[8] = self.get_historical_point(8, history, arm_thresh)
            smoothed[10] = self.get_historical_point(10, history, arm_thresh)
        return smoothed

    def get_historical_point(self, body_part_idx, history, thresh):
        """
        从历史队列中寻找最近一帧可信的点，找不到则返回当前点但置置信度为0
        history 最新帧在最后
        body_part_idx: 在body_kpts (12,) 中的索引
        """
        # 倒序遍历历史（从最新到最旧）
        for frame_kpts in reversed(history):
            conf = frame_kpts[body_part_idx, 2]
            if conf > thresh:
                return frame_kpts[body_part_idx].copy()
        # 历史中也没有可信点，返回当前点坐标但置置信度0
        point = np.array([0.0, 0.0, 0.0])
        return point

    def destroy_node(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

class AdaptiveEMAFilter:
    """支持动态 alpha 的指数移动平均滤波器"""
    def __init__(self, alpha_min=0.15, alpha_max=0.7):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.value = None

    def update(self, new_val, alpha):
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))  # 钳制范围
        if self.value is None:
            self.value = new_val.copy()
        else:
            self.value = alpha * new_val + (1 - alpha) * self.value
        return self.value.copy()
    

def main(args=None):
    rclpy.init(args=args)
    node = BodyPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
