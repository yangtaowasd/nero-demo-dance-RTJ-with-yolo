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
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('show_gui', True)
        self.declare_parameter('history_size', 5)   # 历史帧数
        self.declare_parameter('real_shoulder_width', 0.35)  # 米

        model_path = self.get_parameter('model_path').value
        self.camera_id = self.get_parameter('camera_id').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.arm_conf_thres = self.get_parameter('arm_conf_thres').value
        self.use_camera_topic = self.get_parameter('use_camera_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.show_gui = self.get_parameter('show_gui').value
        self.history_size = self.get_parameter('history_size').value
        self.real_shoulder_width = self.get_parameter('real_shoulder_width').value

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
        self.histories = {}                  # tid -> deque of body kpts (12, 3) maxlen=self.history_size
        self.running = True

        # ----- 发布器与定时器 -----
        self.pose_pub = self.create_publisher(Float32MultiArray, '/image_raw', 10)
        self.timer = self.create_timer(0.05, self.publish_callback)

        # EMA 平滑器存储: key = (track_id, body_part_index)
        self.ema_smoothers = {}       # 存储平滑器实例
        self.prev_raw_kpts = {}       # key: (tid, i) -> 上一帧原始坐标 (x,y)
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
                        body_kpts = kpts[self.BODY_PARTS, :].copy()  # shape (12,3)
                        if tid not in self.histories:
                            self.histories[tid] = deque(maxlen=self.history_size)
                        self.histories[tid].append(body_kpts)
                    # 清理已经消失的track的历史
                    active_ids = {t[0] for t in tracked}
                    for tid in list(self.histories.keys()):
                        if tid not in active_ids:
                            del self.histories[tid]

                    # 可选可视化
                    if self.show_gui:
                        self.visualize(frame, tracked)
                except Exception as e:
                    self.get_logger().error(f"Inference error: {e}")

    # ---------- 可视化 ----------
    def visualize(self, frame, tracked):
        for tid, kpts in tracked:
            left_arm_ok = all(kpts[i][2] > self.arm_conf_thres for i in [5, 7, 9])
            right_arm_ok = all(kpts[i][2] > self.arm_conf_thres for i in [6, 8, 10])
            for i in self.BODY_PARTS:
                x, y, conf = kpts[i]
                if conf < 0.5: continue
                if i in (7, 9) and not left_arm_ok: continue
                if i in (8, 10) and not right_arm_ok: continue
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            for start, end in self.BODY_SKELETON:
                if (start, end) in [(5, 7), (7, 9)] and not left_arm_ok: continue
                if (start, end) in [(6, 8), (8, 10)] and not right_arm_ok: continue
                if kpts[start][2] > 0.5 and kpts[end][2] > 0.5:
                    pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                    pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        cv2.imshow('YOLO Pose', frame)
        cv2.waitKey(1)

    # ---------- 发布回调（20Hz）----------
    def publish_callback(self):
        with self.lock:
            # 深拷贝当前跟踪结果
            tracks_snapshot = [(tid, kpts.copy()) for tid, kpts in self.latest_tracks]
            histories_snapshot = {tid: list(h) for tid, h in self.histories.items()}

        data = []
        data.append(float(len(tracks_snapshot)))
        for tid, kpts_17 in tracks_snapshot:
            # 获取该人的历史（12个身体点）
            history = histories_snapshot.get(tid, [])
            # 对当前帧的身体点先做手臂完整性判断，不可信点用历史填充
            body_kpts = kpts_17[self.BODY_PARTS, :].copy()  # (12,3)
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
            data.extend(smoothed.flatten().tolist())
            # data.extend(body_kpts.flatten().tolist())
        msg = Float32MultiArray(data=data)
        self.pose_pub.publish(msg)

    # def smooth_with_history(self, current_body, history, arm_thresh):
    #     """
    #     current_body: (12,3) 当前帧身体关键点
    #     history: list of (12,3) arrays, 最新帧在最后
    #     arm_thresh: 手臂完整性阈值
    #     返回平滑后的 (12,3) 关键点
    #     """
    #     # 判断左右臂完整性（基于原始17点，这里我们需要肩肘腕的置信度）
    #     # 注意：current_body中索引与BODY_PARTS对应：5肩6肩7肘8肘9腕10腕...
    #     # 左臂：肩(索引0? 需要映射) 让我们用绝对值：左肩索引5，左肘7，左腕9
    #     # 在body_kpts中，BODY_PARTS = [5,6,7,8,9,10,11,12,13,14,15,16]
    #     # 所以左肩在body_kpts的索引0，右肩1，左肘2，右肘3，左腕4，右腕5
    #     left_shoulder_idx = 0   # 对应5
    #     right_shoulder_idx = 1  # 对应6
    #     left_elbow_idx = 2      # 对应7
    #     right_elbow_idx = 3     # 对应8
    #     left_wrist_idx = 4      # 对应9
    #     right_wrist_idx = 5     # 对应10

    #     left_arm_ok = all(current_body[i, 2] > arm_thresh for i in 
    #                       [left_shoulder_idx, left_elbow_idx, left_wrist_idx])
    #     right_arm_ok = all(current_body[i, 2] > arm_thresh for i in 
    #                        [right_shoulder_idx, right_elbow_idx, right_wrist_idx])

    #     smoothed = current_body.copy()
    #     # 对肘和腕进行平滑
    #     # 左肘(2) 左腕(4) 右肘(3) 右腕(5)
    #     if not left_arm_ok:
    #         smoothed[left_elbow_idx] = self.get_historical_point(left_elbow_idx, history, arm_thresh)
    #         smoothed[left_wrist_idx] = self.get_historical_point(left_wrist_idx, history, arm_thresh)
    #     if not right_arm_ok:
    #         smoothed[right_elbow_idx] = self.get_historical_point(right_elbow_idx, history, arm_thresh)
    #         smoothed[right_wrist_idx] = self.get_historical_point(right_wrist_idx, history, arm_thresh)
    #     return smoothed
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