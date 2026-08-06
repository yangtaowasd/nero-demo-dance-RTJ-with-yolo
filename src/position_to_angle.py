#!/usr/bin/env python3
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import numpy as np
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD, ROBOT_MDH_PRESET
from scipy.spatial.transform import Rotation as R

# ==================== pyAgxArm 官方 Nero MDH 参数 ====================
NERO_MDH = tuple(tuple(link) for link in ROBOT_MDH_PRESET["nero"])
NERO_JOINT_LIMITS = [
    tuple(ROBOT_JOINT_LIMIT_PRESET_RAD["nero"][f"joint{i}"])
    for i in range(1, 8)
]


def dh_transform(a, alpha, d, theta):
    """标准 DH 变换矩阵"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct,    -st,   0,     a],
        [st*ca,  ct*ca, -sa, -d*sa],
        [st*sa,  ct*sa,  ca,  d*ca],
        [0,      0,     0,    1]
    ])


# ==================== 卡尔曼滤波器 ====================
class KalmanFilter3D:
    def __init__(self, dt, sigma_a=0.5, sigma_pos=0.005, sigma_z=0.01):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 0.1
        self.initialized = False

        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        q = np.array([
            [dt**4 / 4, 0, 0, dt**3 / 2, 0, 0],
            [0, dt**4 / 4, 0, 0, dt**3 / 2, 0],
            [0, 0, dt**4 / 4, 0, 0, dt**3 / 2],
            [dt**3 / 2, 0, 0, dt**2, 0, 0],
            [0, dt**3 / 2, 0, 0, dt**2, 0],
            [0, 0, dt**3 / 2, 0, 0, dt**2]
        ]) * sigma_a**2
        self.Q = q

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.diag([sigma_pos**2, sigma_pos**2, sigma_z**2])

    def initialize(self, pos):
        self.x[:3] = np.array(pos, dtype=float).reshape(3, 1)
        self.x[3:] = 0.0
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].flatten()

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x[:3].flatten(), False
        d = np.sqrt(y.T @ S_inv @ y).item()
        if d > 5.0:
            return self.x[:3].flatten(), False
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].flatten(), True

    def get_position(self):
        return self.x[:3].flatten()


# ==================== 滤波节点 ====================
class ArmPointFilterNode(Node):
    def __init__(self):
        super().__init__("arm_point_filter_node")

        self.declare_parameter("input_topic", "/body_pose")
        self.declare_parameter("left_output_topic", "/left_arm")
        self.declare_parameter("right_output_topic", "/right_arm")
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("sigma_a", 0.5)
        self.declare_parameter("sigma_pos", 0.005)
        self.declare_parameter("sigma_z", 0.01)
        self.declare_parameter("max_missing_frames", 5)

        self.input_topic = self.get_parameter("input_topic").value
        self.left_output_topic = self.get_parameter("left_output_topic").value
        self.right_output_topic = self.get_parameter("right_output_topic").value
        self.dt = float(self.get_parameter("dt").value)
        self.sigma_a = float(self.get_parameter("sigma_a").value)
        self.sigma_pos = float(self.get_parameter("sigma_pos").value)
        self.sigma_z = float(self.get_parameter("sigma_z").value)
        self.max_missing_frames = int(self.get_parameter("max_missing_frames").value)

        self.num_keypoints = 17
        self.expected_len = 1 + self.num_keypoints * 3

        self.target_points = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
        }

        self.left_arm_order = ["left_shoulder", "left_elbow", "left_wrist"]
        self.right_arm_order = ["right_shoulder", "right_elbow", "right_wrist"]

        self.filters = {}
        self.missing_counts = {}
        for name in self.target_points.keys():
            self.filters[name] = KalmanFilter3D(
                dt=self.dt,
                sigma_a=self.sigma_a,
                sigma_pos=self.sigma_pos,
                sigma_z=self.sigma_z
            )
            self.missing_counts[name] = self.max_missing_frames + 1

        self.sub = self.create_subscription(
            Float32MultiArray, self.input_topic, self.callback, 10
        )
        self.left_pub = self.create_publisher(Float32MultiArray, self.left_output_topic, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, self.right_output_topic, 10)

        self.get_logger().info("arm_point_filter_node started")

    def get_keypoint_xyz(self, data, keypoint_id):
        base = 1 + keypoint_id * 3
        return [data[base], data[base+1], data[base+2]]

    def is_valid_xyz(self, xyz):
        return all(np.isfinite(v) for v in xyz)

    def filter_one_point(self, name, raw_pos):
        kf = self.filters[name]
        if not kf.initialized:
            kf.initialize(raw_pos)
            self.missing_counts[name] = 0
            return np.asarray(raw_pos, dtype=float), True
        kf.predict()
        filtered_pos, valid = kf.update(raw_pos)
        if valid:
            self.missing_counts[name] = 0
        return filtered_pos, valid

    def use_predicted_point(self, name, already_predicted=False):
        kf = self.filters[name]
        if not kf.initialized:
            return None
        if not already_predicted:
            kf.predict()
        self.missing_counts[name] += 1
        if self.missing_counts[name] > self.max_missing_frames:
            return None
        return kf.get_position()

    def make_arm_msg(self, filtered_points, arm_order):
        msg = Float32MultiArray()
        out = []
        for name in arm_order:
            xyz = filtered_points[name]
            out.extend([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        msg.data = out
        return msg

    def make_relative_arm_msg(self, filtered_points, arm_order):
        msg = Float32MultiArray()
        shoulder = np.asarray(filtered_points[arm_order[0]], dtype=float)
        out = []
        for name in arm_order:
            xyz = np.asarray(filtered_points[name], dtype=float) - shoulder
            out.extend([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        msg.data = out
        return msg

    def has_ready_points(self, filtered_points, arm_order):
        for name in arm_order:
            kf = self.filters[name]
            xyz = filtered_points.get(name)
            if (
                xyz is None
                or not kf.initialized
                or not self.is_valid_xyz(xyz)
                or self.missing_counts[name] > self.max_missing_frames
            ):
                return False
        return True

    def callback(self, msg):
        data = list(msg.data)

        if len(data) < 1:
            self.get_logger().warning("empty data", throttle_duration_sec=1.0)
            return

        person_count = int(data[0])

        if person_count == 0:
            self.get_logger().warning("no person detected", throttle_duration_sec=1.0)
            return

        if len(data) != self.expected_len:
            self.get_logger().warning(
                f"length error: {len(data)}, expected: {self.expected_len}",
                throttle_duration_sec=1.0
            )
            return

        filtered_points = {}
        for name, kp_id in self.target_points.items():
            raw_pos = self.get_keypoint_xyz(data, kp_id)
            if not self.is_valid_xyz(raw_pos):
                self.get_logger().warning(f"{name} invalid xyz: {raw_pos}", throttle_duration_sec=1.0)
                predicted_pos = self.use_predicted_point(name)
                if predicted_pos is not None:
                    filtered_points[name] = predicted_pos
                continue
            filtered_pos, valid = self.filter_one_point(name, raw_pos)
            if not valid:
                self.get_logger().warning(f"{name} rejected by Kalman gate", throttle_duration_sec=1.0)
                predicted_pos = self.use_predicted_point(name, already_predicted=True)
                if predicted_pos is not None:
                    filtered_points[name] = predicted_pos
                continue
            filtered_points[name] = filtered_pos

        if self.has_ready_points(filtered_points, self.left_arm_order):
            self.left_pub.publish(self.make_relative_arm_msg(filtered_points, self.left_arm_order))
        else:
            self.get_logger().warning("left arm keypoints are not ready", throttle_duration_sec=1.0)

        if self.has_ready_points(filtered_points, self.right_arm_order):
            self.right_pub.publish(self.make_relative_arm_msg(filtered_points, self.right_arm_order))
        else:
            self.get_logger().warning("right arm keypoints are not ready", throttle_duration_sec=1.0)


# ==================== 逆运动学求解器 ====================
class IKSolver:
    def __init__(self, fk_func, jac_func, elbow_pos_func,
                 joint_limits=None, damping=0.05, tol=0.02,
                 max_iter=100, alpha_sec=0.1, max_step=0.05,
                 max_output_delta=0.15, logger=None):
        self.fk = fk_func
        self.jacobian = jac_func
        self.elbow_pos = elbow_pos_func
        self.joint_limits = joint_limits
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_sec = alpha_sec
        self.max_step = max_step
        self.max_output_delta = max_output_delta
        self.logger = logger

    def pose_error(self, T_current, T_target):
        p_err = T_target[:3, 3] - T_current[:3, 3]
        R_err = T_target[:3, :3] @ T_current[:3, :3].T
        rot_err = R.from_matrix(R_err).as_rotvec()
        return np.concatenate([p_err, rot_err])

    def check_joint_limits(self, q):
        if self.joint_limits is None:
            return q
        return np.clip(q,
                       [l[0] for l in self.joint_limits],
                       [l[1] for l in self.joint_limits])

    def solve(self, T_target, q_init, elbow_target=None):
        q = q_init.copy()
        target_pos = T_target[:3, 3]
        best_q = q.copy()
        best_err = np.linalg.norm(target_pos - self.fk(q)[:3, 3])
        converged = False
        for i in range(self.max_iter):
            T_curr = self.fk(q)
            pos_err = target_pos - T_curr[:3, 3]
            err_norm = np.linalg.norm(pos_err)
            if err_norm < best_err:
                best_q = q.copy()
                best_err = err_norm
            if err_norm < self.tol:
                converged = True
                if self.logger:
                    self.logger.info(f"IK converged in {i} iterations")
                break

            J = self.jacobian(q)
            J_pos = J[:3, :]
            lambda_sq = self.damping ** 2
            J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * np.eye(3))
            dq_task = J_dls @ pos_err

            if elbow_target is not None:
                grad = self._elbow_gradient(q, elbow_target)
                N = np.eye(7) - np.linalg.pinv(J_pos) @ J_pos
                dq_sec = N @ (self.alpha_sec * grad)
            else:
                dq_sec = 0.0

            dq = dq_task + dq_sec
            dq_norm = np.linalg.norm(dq)
            if dq_norm > self.max_step:
                dq = dq / dq_norm * self.max_step

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                q_candidate = self.check_joint_limits(q + scale * dq)
                candidate_err = np.linalg.norm(target_pos - self.fk(q_candidate)[:3, 3])
                if candidate_err < err_norm:
                    q = q_candidate
                    accepted = True
                    break

            if not accepted:
                break

        if not converged and self.logger:
            self.logger.warning(f"IK did not converge, best position error: {best_err:.4f} m")
        return self.limit_output_delta(best_q if not converged else q, q_init), converged

    def limit_output_delta(self, q, q_init):
        q = self.check_joint_limits(q)
        delta = np.clip(q - q_init, -self.max_output_delta, self.max_output_delta)
        return self.check_joint_limits(q_init + delta)

    def _elbow_gradient(self, q, elbow_target):
        eps = 1e-6
        p0 = self.elbow_pos(q)
        grad = np.zeros(7)
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            p1 = self.elbow_pos(q + dq)
            grad[i] = np.dot(p1 - p0, elbow_target - p0) / eps
        return grad


# ==================== IK 节点 ====================
class ArmIKNode(Node):
    def __init__(self):
        super().__init__("arm_ik_node")

        self.declare_parameter("left_input_topic", "/left_arm")
        self.declare_parameter("right_input_topic", "/right_arm")
        self.declare_parameter("left_output_topic", "/left_joint_angles")
        self.declare_parameter("right_output_topic", "/right_joint_angles")
        self.declare_parameter("left_joint_status_topic", "/left_joint_status")
        self.declare_parameter("right_joint_status_topic", "/right_joint_status")
        self.declare_parameter("command_output_mode", "float_array")
        self.declare_parameter("ik_backend", "moveit")
        self.declare_parameter("motion_scale", 0.7)
        self.declare_parameter("ik_tolerance", 0.05)
        self.declare_parameter("max_target_delta", 0.08)
        self.declare_parameter("arm_points_are_relative", True)
        self.declare_parameter(
            "camera_to_robot_matrix",
            [
                0.0, 0.0, -1.0,
                -1.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
            ],
        )
        self.declare_parameter("moveit_ik_service", "/compute_ik")
        self.declare_parameter("moveit_fk_service", "/compute_fk")
        self.declare_parameter("moveit_frame_id", "base_link")
        self.declare_parameter("moveit_ik_timeout", 0.05)
        self.declare_parameter("moveit_ik_attempts", 4)
        self.declare_parameter("left_moveit_group", "arm")
        self.declare_parameter("right_moveit_group", "arm")
        self.declare_parameter("left_moveit_tip_link", "tcp_link")
        self.declare_parameter("right_moveit_tip_link", "tcp_link")
        self.declare_parameter("left_moveit_joint_names", "joint1,joint2,joint3,joint4,joint5,joint6,joint7")
        self.declare_parameter("right_moveit_joint_names", "joint1,joint2,joint3,joint4,joint5,joint6,joint7")
        self.declare_parameter("moveit_target_orientation", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("left_agx_joint_state_topic", "")
        self.declare_parameter("right_agx_joint_state_topic", "")
        self.declare_parameter("left_agx_tcp_pose_topic", "")
        self.declare_parameter("right_agx_tcp_pose_topic", "")
        self.declare_parameter("left_agx_move_j_topic", "/left/control/move_j")
        self.declare_parameter("right_agx_move_j_topic", "/right/control/move_j")
        self.declare_parameter("left_agx_move_p_topic", "/left/control/move_p")
        self.declare_parameter("right_agx_move_p_topic", "/right/control/move_p")

        # ==================== 测试专用参数 START ====================
        # test_mode=True 时，不等待真实 /left_joint_status 和 /right_joint_status。
        # 只用于离线验证 IK 输出，接真实机械臂时保持默认 False。
        self.declare_parameter("test_mode", False)
        self.declare_parameter("test_left_initial_joints", [0.3, 0.2, -0.2, 0.3, -0.1, -0.2, 0.1])
        self.declare_parameter("test_right_initial_joints", [0.6, 0.0, -0.4, 0.0, 0.0, -0.2, 0.0])
        self.declare_parameter("test_publish_near_limits", False)
        self.declare_parameter("test_debug_log", True)
        # ==================== 测试专用参数 END ====================

        left_in = self.get_parameter("left_input_topic").value
        right_in = self.get_parameter("right_input_topic").value
        left_out = self.get_parameter("left_output_topic").value
        right_out = self.get_parameter("right_output_topic").value
        left_joint_status = self.get_parameter("left_joint_status_topic").value
        right_joint_status = self.get_parameter("right_joint_status_topic").value
        self.command_output_mode = str(self.get_parameter("command_output_mode").value).strip().lower()
        self.ik_backend = str(self.get_parameter("ik_backend").value).strip().lower()
        self.motion_scale = float(self.get_parameter("motion_scale").value)
        self.ik_tolerance = float(self.get_parameter("ik_tolerance").value)
        self.max_target_delta = float(self.get_parameter("max_target_delta").value)
        self.arm_points_are_relative = self.read_bool_param("arm_points_are_relative")
        self.camera_to_robot_matrix = self.read_matrix_param("camera_to_robot_matrix")
        self.moveit_ik_service = self.get_parameter("moveit_ik_service").value
        self.moveit_fk_service = self.get_parameter("moveit_fk_service").value
        self.moveit_frame_id = self.get_parameter("moveit_frame_id").value
        self.moveit_ik_timeout = float(self.get_parameter("moveit_ik_timeout").value)
        self.moveit_ik_attempts = int(self.get_parameter("moveit_ik_attempts").value)
        self.left_moveit_group = self.get_parameter("left_moveit_group").value
        self.right_moveit_group = self.get_parameter("right_moveit_group").value
        self.left_moveit_tip_link = self.get_parameter("left_moveit_tip_link").value
        self.right_moveit_tip_link = self.get_parameter("right_moveit_tip_link").value
        self.left_moveit_joint_names = self.read_csv_param("left_moveit_joint_names")
        self.right_moveit_joint_names = self.read_csv_param("right_moveit_joint_names")
        self.moveit_target_orientation = self.read_quat_param("moveit_target_orientation")
        self.left_agx_joint_state_topic = self.get_parameter("left_agx_joint_state_topic").value
        self.right_agx_joint_state_topic = self.get_parameter("right_agx_joint_state_topic").value
        self.left_agx_tcp_pose_topic = self.get_parameter("left_agx_tcp_pose_topic").value
        self.right_agx_tcp_pose_topic = self.get_parameter("right_agx_tcp_pose_topic").value
        self.left_agx_move_j_topic = self.get_parameter("left_agx_move_j_topic").value
        self.right_agx_move_j_topic = self.get_parameter("right_agx_move_j_topic").value
        self.left_agx_move_p_topic = self.get_parameter("left_agx_move_p_topic").value
        self.right_agx_move_p_topic = self.get_parameter("right_agx_move_p_topic").value

        # ==================== 测试专用配置 START ====================
        self.test_mode = self.read_bool_param("test_mode")
        self.test_publish_near_limits = self.read_bool_param("test_publish_near_limits")
        self.test_debug_log = self.read_bool_param("test_debug_log")
        self.test_left_initial_joints = self.read_test_joints("test_left_initial_joints")
        self.test_right_initial_joints = self.read_test_joints("test_right_initial_joints")
        # ==================== 测试专用配置 END ====================

        # 机器人模型：基于 pyAgxArm 官方 Nero MDH 参数。
        def my_fk(q):
            T = np.eye(4)
            for i, (d_i, a_i, alpha_i, theta_offset_i) in enumerate(NERO_MDH):
                theta = q[i] + theta_offset_i
                T_i = dh_transform(a_i, alpha_i, d_i, theta)
                T = T @ T_i
            return T

        def my_jacobian(q):
            T = np.eye(4)
            T_list = [T]
            for i, (d_i, a_i, alpha_i, theta_offset_i) in enumerate(NERO_MDH):
                theta = q[i] + theta_offset_i
                T_i = dh_transform(a_i, alpha_i, d_i, theta)
                T = T @ T_i
                T_list.append(T)
            p_n = T_list[7][:3, 3]
            J = np.zeros((6, 7))
            for i in range(7):
                z = T_list[i][:3, 2]
                p = T_list[i][:3, 3]
                J[:3, i] = np.cross(z, p_n - p)
                J[3:, i] = z
            return J

        def my_elbow(q):
            T = np.eye(4)
            for i in range(4):  # 关节 1~4，索引 0~3
                d_i, a_i, alpha_i, theta_offset_i = NERO_MDH[i]
                theta = q[i] + theta_offset_i
                T_i = dh_transform(a_i, alpha_i, d_i, theta)
                T = T @ T_i
            return T[:3, 3]

        limits = NERO_JOINT_LIMITS

        # 创建 IK 求解器
        self.ik_solver_left = IKSolver(
            fk_func=my_fk,
            jac_func=my_jacobian,
            elbow_pos_func=my_elbow,
            joint_limits=limits,
            damping=0.1,
            tol=self.ik_tolerance,
            max_iter=200,
            logger=self.get_logger()
        )
        self.ik_solver_right = IKSolver(
            fk_func=my_fk,
            jac_func=my_jacobian,
            elbow_pos_func=my_elbow,
            joint_limits=limits,
            damping=0.1,
            tol=self.ik_tolerance,
            max_iter=200,
            logger=self.get_logger()
        )

        # 初始关节角（最好从真实机器人读取当前状态）
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)
        self.limit_margin = 0.03
        self.left_home_tcp = None
        self.right_home_tcp = None
        self.left_joint_ready = False
        self.right_joint_ready = False
        self.moveit_cb_group = ReentrantCallbackGroup()
        self.moveit_ik_client = None
        self.moveit_fk_client = None
        if self.ik_backend == "moveit":
            self.moveit_ik_client = self.create_client(
                GetPositionIK, self.moveit_ik_service, callback_group=self.moveit_cb_group
            )
            self.moveit_fk_client = self.create_client(
                GetPositionFK, self.moveit_fk_service, callback_group=self.moveit_cb_group
            )

        # ==================== 测试专用初始化 START ====================
        if self.test_mode:
            self.q_left = self.test_left_initial_joints.copy()
            self.q_right = self.test_right_initial_joints.copy()
            self.left_joint_ready = True
            self.right_joint_ready = True
            self.get_logger().warning(
                "TEST MODE enabled: using fake initial joint angles, not waiting for joint_status"
            )
        # ==================== 测试专用初始化 END ====================

        # 订阅和发布
        self.left_sub = self.create_subscription(
            Float32MultiArray, left_in, self.left_callback, 10,
            callback_group=self.moveit_cb_group
        )
        self.right_sub = self.create_subscription(
            Float32MultiArray, right_in, self.right_callback, 10,
            callback_group=self.moveit_cb_group
        )
        self.left_pub = self.create_publisher(Float32MultiArray, left_out, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, right_out, 10)
        self.left_agx_move_j_pub = self.create_publisher(JointState, self.left_agx_move_j_topic, 10)
        self.right_agx_move_j_pub = self.create_publisher(JointState, self.right_agx_move_j_topic, 10)
        self.left_agx_move_p_pub = self.create_publisher(PoseStamped, self.left_agx_move_p_topic, 10)
        self.right_agx_move_p_pub = self.create_publisher(PoseStamped, self.right_agx_move_p_topic, 10)
        self.left_joint_status_sub = self.create_subscription(
            Float32MultiArray, left_joint_status, self.left_joint_status_callback, 10
        )
        self.right_joint_status_sub = self.create_subscription(
            Float32MultiArray, right_joint_status, self.right_joint_status_callback, 10
        )
        if self.left_agx_joint_state_topic:
            self.create_subscription(
                JointState, self.left_agx_joint_state_topic,
                lambda msg: self.agx_joint_state_callback("left", msg), 10
            )
        if self.right_agx_joint_state_topic:
            self.create_subscription(
                JointState, self.right_agx_joint_state_topic,
                lambda msg: self.agx_joint_state_callback("right", msg), 10
            )
        if self.left_agx_tcp_pose_topic:
            self.create_subscription(
                PoseStamped, self.left_agx_tcp_pose_topic,
                lambda msg: self.agx_tcp_pose_callback("left", msg), 10
            )
        if self.right_agx_tcp_pose_topic:
            self.create_subscription(
                PoseStamped, self.right_agx_tcp_pose_topic,
                lambda msg: self.agx_tcp_pose_callback("right", msg), 10
            )
        self.get_logger().info("arm_ik_node started with DH model")
        self.get_logger().info(
            f"IK backend={self.ik_backend}, command_output_mode={self.command_output_mode}, "
            f"arm_points_are_relative={self.arm_points_are_relative}, "
            f"ik_tolerance={self.ik_tolerance}, max_target_delta={self.max_target_delta}, "
            f"camera_to_robot_matrix={self.camera_to_robot_matrix.tolist()}"
        )
        if self.ik_backend == "moveit":
            self.get_logger().warning(
                "MoveIt IK backend selected. agx_arm_ros currently configures group=arm, "
                "tip=tcp_link and KDLKinematicsPlugin for /compute_ik."
            )
        elif self.ik_backend == "custom":
            self.get_logger().warning(
                "CUSTOM IK backend selected. This solver is only for debugging and is not validated for real motion."
            )
        else:
            self.get_logger().error(f"unknown ik_backend: {self.ik_backend}")
        if self.command_output_mode == "agx_move_p":
            self.get_logger().warning(
                "AGX move_p mode selected. Waiting for TCP feedback topics: "
                f"left='{self.left_agx_tcp_pose_topic}', right='{self.right_agx_tcp_pose_topic}'. "
                "Keep agx_control_enabled=false for dry feedback, set true only when ready to move."
            )

    # ==================== 测试专用函数 START ====================
    def read_test_joints(self, param_name):
        joints = list(self.get_parameter(param_name).value)
        if len(joints) != 7:
            self.get_logger().warning(f"{param_name} length error, use zeros: {joints}")
            return np.zeros(7)
        return np.asarray(joints, dtype=float)

    def read_matrix_param(self, param_name):
        values = list(self.get_parameter(param_name).value)
        if len(values) != 9:
            self.get_logger().warning(f"{param_name} length error, use identity: {values}")
            return np.eye(3)
        return np.asarray(values, dtype=float).reshape(3, 3)

    def read_bool_param(self, param_name):
        value = self.get_parameter(param_name).value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def read_csv_param(self, param_name):
        value = self.get_parameter(param_name).value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item) for item in value]

    def read_quat_param(self, param_name):
        values = list(self.get_parameter(param_name).value)
        if len(values) != 4:
            self.get_logger().warning(f"{param_name} length error, use identity quaternion: {values}")
            return [0.0, 0.0, 0.0, 1.0]
        return [float(v) for v in values]

    def side_moveit_config(self, side):
        if side == "left":
            return self.left_moveit_group, self.left_moveit_tip_link, self.left_moveit_joint_names
        return self.right_moveit_group, self.right_moveit_tip_link, self.right_moveit_joint_names

    def wait_future(self, future, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        return future.done()

    def fill_robot_state_seed(self, robot_state, q_seed, joint_names):
        if not joint_names:
            return True
        if len(joint_names) != len(q_seed):
            self.get_logger().warning(
                f"MoveIt joint name count mismatch: names={len(joint_names)}, q={len(q_seed)}",
                throttle_duration_sec=1.0
            )
            return False
        robot_state.joint_state.name = list(joint_names)
        robot_state.joint_state.position = [float(v) for v in q_seed]
        return True

    def call_moveit_fk(self, side, q_seed):
        group_name, tip_link, joint_names = self.side_moveit_config(side)
        if not tip_link or not joint_names:
            self.get_logger().warning(
                f"{side} MoveIt FK needs {side}_moveit_tip_link and {side}_moveit_joint_names",
                throttle_duration_sec=1.0
            )
            return None
        if not self.moveit_fk_client.service_is_ready():
            self.get_logger().warning(
                f"MoveIt FK service is not ready: {self.moveit_fk_service}",
                throttle_duration_sec=1.0
            )
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = self.moveit_frame_id
        req.fk_link_names = [tip_link]
        if not self.fill_robot_state_seed(req.robot_state, q_seed, joint_names):
            return None

        future = self.moveit_fk_client.call_async(req)
        if not self.wait_future(future, max(self.moveit_ik_timeout, 0.05)):
            self.get_logger().warning("MoveIt FK request timed out", throttle_duration_sec=1.0)
            return None

        resp = future.result()
        if resp is None or resp.error_code.val != 1 or not resp.pose_stamped:
            code = None if resp is None else resp.error_code.val
            self.get_logger().warning(f"MoveIt FK failed: error_code={code}", throttle_duration_sec=1.0)
            return None

        pos = resp.pose_stamped[0].pose.position
        return np.array([pos.x, pos.y, pos.z], dtype=float)

    def get_moveit_home_tcp(self, side, q_seed):
        attr = f"{side}_home_tcp"
        home_tcp = getattr(self, attr)
        if home_tcp is None:
            home_tcp = self.call_moveit_fk(side, q_seed)
            if home_tcp is None:
                return None
            setattr(self, attr, home_tcp)
            self.get_logger().info(f"{side} MoveIt home tcp set to {home_tcp.tolist()}")
        return home_tcp

    def make_pose_stamped(self, target_pos):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.moveit_frame_id
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(target_pos[0])
        pose_stamped.pose.position.y = float(target_pos[1])
        pose_stamped.pose.position.z = float(target_pos[2])
        qx, qy, qz, qw = self.moveit_target_orientation
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        return pose_stamped

    def joint_names_for_side(self, side):
        _, _, joint_names = self.side_moveit_config(side)
        if joint_names:
            return joint_names
        return [f"joint{i}" for i in range(1, 8)]

    def publish_joint_command(self, side, q_result):
        if self.command_output_mode == "agx_move_j":
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names_for_side(side)
            msg.position = [float(v) for v in q_result]
            if side == "left":
                self.left_agx_move_j_pub.publish(msg)
            else:
                self.right_agx_move_j_pub.publish(msg)
            return

        msg = Float32MultiArray()
        msg.data = q_result.tolist()
        if side == "left":
            self.left_pub.publish(msg)
        else:
            self.right_pub.publish(msg)

    def publish_pose_command(self, side, target_pos):
        msg = self.make_pose_stamped(target_pos)
        if side == "left":
            self.left_agx_move_p_pub.publish(msg)
        else:
            self.right_agx_move_p_pub.publish(msg)

    def agx_joint_state_callback(self, side, msg):
        joint_names = self.joint_names_for_side(side)
        index = {name: i for i, name in enumerate(msg.name)}
        missing = [name for name in joint_names if name not in index]
        if missing:
            self.get_logger().warning(
                f"{side} agx joint feedback missing joints: {missing}",
                throttle_duration_sec=1.0
            )
            return

        q = np.asarray([msg.position[index[name]] for name in joint_names], dtype=float)
        if side == "left":
            self.q_left = q
            self.left_joint_ready = True
            solver = self.ik_solver_left
        else:
            self.q_right = q
            self.right_joint_ready = True
            solver = self.ik_solver_right

        attr = f"{side}_home_tcp"
        if getattr(self, attr) is None:
            home_tcp = solver.fk(q)[:3, 3].copy()
            setattr(self, attr, home_tcp)
            self.get_logger().info(
                f"{side} home tcp set from agx joint feedback FK: {home_tcp.tolist()}"
            )

    def agx_tcp_pose_callback(self, side, msg):
        pos = msg.pose.position
        home_tcp = np.asarray([pos.x, pos.y, pos.z], dtype=float)
        attr = f"{side}_home_tcp"
        if getattr(self, attr) is None:
            setattr(self, attr, home_tcp)
            self.get_logger().info(f"{side} agx home tcp set to {home_tcp.tolist()}")

    def extract_moveit_solution(self, side, response):
        _, _, joint_names = self.side_moveit_config(side)
        names = list(response.solution.joint_state.name)
        positions = list(response.solution.joint_state.position)
        if joint_names:
            index = {name: i for i, name in enumerate(names)}
            missing = [name for name in joint_names if name not in index]
            if missing:
                self.get_logger().warning(f"MoveIt IK missing joints for {side}: {missing}")
                return None
            return np.asarray([positions[index[name]] for name in joint_names], dtype=float)
        if len(positions) < 7:
            self.get_logger().warning(f"MoveIt IK returned too few joints for {side}: {len(positions)}")
            return None
        self.get_logger().warning(
            f"{side}_moveit_joint_names is empty; using first 7 joints from MoveIt response",
            throttle_duration_sec=1.0
        )
        return np.asarray(positions[:7], dtype=float)

    def solve_moveit_ik(self, side, q_seed, T_target):
        group_name, tip_link, joint_names = self.side_moveit_config(side)
        if not group_name or not tip_link:
            self.get_logger().warning(
                f"{side} MoveIt IK needs {side}_moveit_group and {side}_moveit_tip_link",
                throttle_duration_sec=1.0
            )
            return None, False
        if not self.moveit_ik_client.service_is_ready():
            self.get_logger().warning(
                f"MoveIt IK service is not ready: {self.moveit_ik_service}",
                throttle_duration_sec=1.0
            )
            return None, False

        req = GetPositionIK.Request()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = tip_link
        req.ik_request.pose_stamped = self.make_pose_stamped(T_target[:3, 3])
        req.ik_request.avoid_collisions = False
        req.ik_request.attempts = self.moveit_ik_attempts
        req.ik_request.timeout.sec = int(self.moveit_ik_timeout)
        req.ik_request.timeout.nanosec = int((self.moveit_ik_timeout % 1.0) * 1e9)
        if not self.fill_robot_state_seed(req.ik_request.robot_state, q_seed, joint_names):
            return None, False

        future = self.moveit_ik_client.call_async(req)
        if not self.wait_future(future, max(self.moveit_ik_timeout + 0.05, 0.1)):
            self.get_logger().warning("MoveIt IK request timed out", throttle_duration_sec=1.0)
            return None, False

        resp = future.result()
        if resp is None or resp.error_code.val != 1:
            code = None if resp is None else resp.error_code.val
            self.get_logger().warning(f"MoveIt IK failed for {side}: error_code={code}", throttle_duration_sec=1.0)
            return None, False

        q_result = self.extract_moveit_solution(side, resp)
        return q_result, q_result is not None

    def maybe_log_test_ik(self, side, shoulder, elbow, wrist,
                          elbow_rel, wrist_rel, elbow_delta, wrist_delta,
                          target, q_result, converged, pos_error):
        if not (self.test_mode and self.test_debug_log):
            return
        self.get_logger().info(
            f"TEST {side}: shoulder={shoulder.tolist()}, elbow={elbow.tolist()}, "
            f"wrist={wrist.tolist()}, elbow_rel={elbow_rel.tolist()}, "
            f"wrist_rel={wrist_rel.tolist()}, elbow_robot_delta={elbow_delta.tolist()}, "
            f"wrist_robot_delta={wrist_delta.tolist()}, target={target.tolist()}, "
            f"q={q_result.tolist()}, converged={converged}, pos_error={pos_error:.4f}"
        )
    # ==================== 测试专用函数 END ====================

    def is_near_joint_limits(self, q):
        near_count = 0
        for value, (lo, hi) in zip(q, self.ik_solver_left.joint_limits):
            if value <= lo + self.limit_margin or value >= hi - self.limit_margin:
                near_count += 1
        return near_count >= 3

    def get_home_tcp(self, side, q, solver):
        attr = f"{side}_home_tcp"
        home_tcp = getattr(self, attr)
        if home_tcp is None:
            home_tcp = solver.fk(q)[:3, 3].copy()
            setattr(self, attr, home_tcp)
            self.get_logger().info(f"{side} home tcp set to {home_tcp.tolist()}")
        return home_tcp

    def get_relative_arm_points(self, shoulder, elbow, wrist):
        """
        /left_arm 和 /right_arm 默认已经是以肩膀为原点的相对坐标。
        如果外部测试节点直接发绝对坐标，可以把 arm_points_are_relative 设为 False。
        """
        if self.arm_points_are_relative:
            return elbow, wrist
        return elbow - shoulder, wrist - shoulder

    def limit_target_delta(self, delta):
        if self.max_target_delta <= 0.0:
            return delta
        norm = np.linalg.norm(delta)
        if norm <= self.max_target_delta:
            return delta
        return delta / norm * self.max_target_delta

    def build_target_pose(self, shoulder, elbow, wrist, home_tcp):
        """
        肩肘腕输入默认是相机坐标下的人体相对肩膀坐标。
        先映射到机器人坐标，再以 home TCP 为中心生成 IK 目标。
        """
        elbow_rel, wrist_rel = self.get_relative_arm_points(shoulder, elbow, wrist)
        wrist_delta = self.limit_target_delta(
            self.motion_scale * (self.camera_to_robot_matrix @ wrist_rel)
        )
        elbow_delta = self.limit_target_delta(
            self.motion_scale * (self.camera_to_robot_matrix @ elbow_rel)
        )

        target_wrist = home_tcp + wrist_delta
        target_elbow = home_tcp + elbow_delta

        T = np.eye(4)
        T[:3, 3] = target_wrist
        return T, target_elbow, elbow_rel, wrist_rel, elbow_delta, wrist_delta

    def left_callback(self, msg):
        if self.command_output_mode != "agx_move_p" and not self.left_joint_ready:
            self.get_logger().warning("left joint status is not ready, skip IK", throttle_duration_sec=1.0)
            return

        if len(msg.data) != 9:
            self.get_logger().warning(f"Left arm data length error: {len(msg.data)}")
            return

        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        q_seed = self.q_left
        if self.command_output_mode == "agx_move_p":
            home_tcp = self.left_home_tcp
            if home_tcp is None:
                self.get_logger().warning(
                    "left AGX feedback is not ready, skip pose command. "
                    f"Check topic '{self.left_agx_tcp_pose_topic}' or '{self.left_agx_joint_state_topic}', "
                    "and start agx_ctrl with namespace:=left.",
                    throttle_duration_sec=1.0
                )
                return
        elif self.ik_backend == "moveit":
            home_tcp = self.get_moveit_home_tcp("left", q_seed)
        elif self.ik_backend == "custom":
            home_tcp = self.get_home_tcp("left", q_seed, self.ik_solver_left)
        else:
            self.get_logger().error(f"unknown ik_backend: {self.ik_backend}", throttle_duration_sec=1.0)
            return
        if home_tcp is None:
            return

        T_target, elbow_target, e_rel, w_rel, e_delta, w_delta = self.build_target_pose(s, e, w, home_tcp)
        if self.command_output_mode == "agx_move_p":
            self.publish_pose_command("left", T_target[:3, 3])
            self.maybe_log_test_ik("left", s, e, w, e_rel, w_rel, e_delta, w_delta,
                                   T_target[:3, 3], q_seed, True, float("nan"))
            return

        if self.ik_backend == "moveit":
            q_result, converged = self.solve_moveit_ik("left", q_seed, T_target)
            if q_result is None:
                return
            pos_error = float("nan")
        else:
            q_result, converged = self.ik_solver_left.solve(T_target, q_seed, elbow_target=elbow_target)
            pos_error = np.linalg.norm(T_target[:3, 3] - self.ik_solver_left.fk(q_result)[:3, 3])
        self.maybe_log_test_ik("left", s, e, w, e_rel, w_rel, e_delta, w_delta,
                               T_target[:3, 3], q_result, converged, pos_error)

        if not converged and not self.test_mode:
            self.get_logger().warning("left IK did not converge, skip publish", throttle_duration_sec=1.0)
            return

        if self.is_near_joint_limits(q_result) and not self.test_publish_near_limits:
            self.get_logger().warning(f"left IK near joint limits, skip publish: {q_result.tolist()}")
            return

        self.q_left = q_result
        self.publish_joint_command("left", q_result)

    def right_callback(self, msg):
        if self.command_output_mode != "agx_move_p" and not self.right_joint_ready:
            self.get_logger().warning("right joint status is not ready, skip IK", throttle_duration_sec=1.0)
            return

        if len(msg.data) != 9:
            self.get_logger().warning(f"Right arm data length error: {len(msg.data)}")
            return

        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        q_seed = self.q_right
        if self.command_output_mode == "agx_move_p":
            home_tcp = self.right_home_tcp
            if home_tcp is None:
                self.get_logger().warning(
                    "right AGX feedback is not ready, skip pose command. "
                    f"Check topic '{self.right_agx_tcp_pose_topic}' or '{self.right_agx_joint_state_topic}', "
                    "and start agx_ctrl with namespace:=right.",
                    throttle_duration_sec=1.0
                )
                return
        elif self.ik_backend == "moveit":
            home_tcp = self.get_moveit_home_tcp("right", q_seed)
        elif self.ik_backend == "custom":
            home_tcp = self.get_home_tcp("right", q_seed, self.ik_solver_right)
        else:
            self.get_logger().error(f"unknown ik_backend: {self.ik_backend}", throttle_duration_sec=1.0)
            return
        if home_tcp is None:
            return

        T_target, elbow_target, e_rel, w_rel, e_delta, w_delta = self.build_target_pose(s, e, w, home_tcp)
        if self.command_output_mode == "agx_move_p":
            self.publish_pose_command("right", T_target[:3, 3])
            self.maybe_log_test_ik("right", s, e, w, e_rel, w_rel, e_delta, w_delta,
                                   T_target[:3, 3], q_seed, True, float("nan"))
            return

        if self.ik_backend == "moveit":
            q_result, converged = self.solve_moveit_ik("right", q_seed, T_target)
            if q_result is None:
                return
            pos_error = float("nan")
        else:
            q_result, converged = self.ik_solver_right.solve(T_target, q_seed, elbow_target=elbow_target)
            pos_error = np.linalg.norm(T_target[:3, 3] - self.ik_solver_right.fk(q_result)[:3, 3])
        self.maybe_log_test_ik("right", s, e, w, e_rel, w_rel, e_delta, w_delta,
                               T_target[:3, 3], q_result, converged, pos_error)

        if not converged and not self.test_mode:
            self.get_logger().warning("right IK did not converge, skip publish", throttle_duration_sec=1.0)
            return

        if self.is_near_joint_limits(q_result) and not self.test_publish_near_limits:
            self.get_logger().warning(f"right IK near joint limits, skip publish: {q_result.tolist()}")
            return

        self.q_right = q_result
        self.publish_joint_command("right", q_result)

    def left_joint_status_callback(self, msg):
        if len(msg.data) == 7:
            self.q_left = np.array(msg.data[0:7])
            self.left_joint_ready = True
            return

        self.get_logger().warning(f"Left joint status data length error: {len(msg.data)}")

    def right_joint_status_callback(self, msg):
        if len(msg.data) == 7:
            self.q_right = np.array(msg.data[0:7])
            self.right_joint_ready = True
            return

        self.get_logger().warning(f"Right joint status data length error: {len(msg.data)}")


def main(args=None):
    rclpy.init(args=args)

    filter_node = ArmPointFilterNode()
    ik_node = ArmIKNode()

    executor = MultiThreadedExecutor()
    executor.add_node(filter_node)
    executor.add_node(ik_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        filter_node.destroy_node()
        ik_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
