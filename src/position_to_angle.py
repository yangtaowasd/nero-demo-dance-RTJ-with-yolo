#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==================== DH 参数与变换 ====================
a =     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                 # 米
alpha = np.deg2rad([0.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]) # 弧度
d =     [0.138, 0.0, 0.31, 0.0, 0.27001, 0.0, 0.0235]        # 米
offset = np.deg2rad([0.0, 180.0, 180.0, 180.0, 90.0, 90.0, 0.0]) # 弧度


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

        self.input_topic = self.get_parameter("input_topic").value
        self.left_output_topic = self.get_parameter("left_output_topic").value
        self.right_output_topic = self.get_parameter("right_output_topic").value
        self.dt = float(self.get_parameter("dt").value)
        self.sigma_a = float(self.get_parameter("sigma_a").value)
        self.sigma_pos = float(self.get_parameter("sigma_pos").value)
        self.sigma_z = float(self.get_parameter("sigma_z").value)

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
        for name in self.target_points.keys():
            self.filters[name] = KalmanFilter3D(
                dt=self.dt,
                sigma_a=self.sigma_a,
                sigma_pos=self.sigma_pos,
                sigma_z=self.sigma_z
            )

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
            return np.asarray(raw_pos, dtype=float), True
        kf.predict()
        return kf.update(raw_pos)

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
            if xyz is None or not kf.initialized or not self.is_valid_xyz(xyz):
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
        valid_points = set()
        for name, kp_id in self.target_points.items():
            raw_pos = self.get_keypoint_xyz(data, kp_id)
            if not self.is_valid_xyz(raw_pos):
                self.get_logger().warning(f"{name} invalid xyz: {raw_pos}", throttle_duration_sec=1.0)
                if self.filters[name].initialized:
                    filtered_points[name] = self.filters[name].get_position()
                continue
            filtered_pos, valid = self.filter_one_point(name, raw_pos)
            if not valid:
                self.get_logger().warning(f"{name} rejected by Kalman gate", throttle_duration_sec=1.0)
                if self.filters[name].initialized:
                    filtered_points[name] = self.filters[name].get_position()
                continue
            valid_points.add(name)
            filtered_points[name] = filtered_pos

        if self.has_ready_points(filtered_points, self.left_arm_order) and all(
            name in valid_points for name in self.left_arm_order
        ):
            self.left_pub.publish(self.make_relative_arm_msg(filtered_points, self.left_arm_order))
        else:
            self.get_logger().warning("left arm keypoints are not ready", throttle_duration_sec=1.0)

        if self.has_ready_points(filtered_points, self.right_arm_order) and all(
            name in valid_points for name in self.right_arm_order
        ):
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
        self.motion_scale = float(self.get_parameter("motion_scale").value)
        self.ik_tolerance = float(self.get_parameter("ik_tolerance").value)
        self.max_target_delta = float(self.get_parameter("max_target_delta").value)
        self.arm_points_are_relative = self.read_bool_param("arm_points_are_relative")
        self.camera_to_robot_matrix = self.read_matrix_param("camera_to_robot_matrix")

        # ==================== 测试专用配置 START ====================
        self.test_mode = self.read_bool_param("test_mode")
        self.test_publish_near_limits = self.read_bool_param("test_publish_near_limits")
        self.test_debug_log = self.read_bool_param("test_debug_log")
        self.test_left_initial_joints = self.read_test_joints("test_left_initial_joints")
        self.test_right_initial_joints = self.read_test_joints("test_right_initial_joints")
        # ==================== 测试专用配置 END ====================

        # 机器人模型：基于 DH 参数
        def my_fk(q):
            T = np.eye(4)
            for i in range(7):
                theta = q[i] + offset[i]
                T_i = dh_transform(a[i], alpha[i], d[i], theta)
                T = T @ T_i
            return T

        def my_jacobian(q):
            T = np.eye(4)
            T_list = [T]
            for i in range(7):
                theta = q[i] + offset[i]
                T_i = dh_transform(a[i], alpha[i], d[i], theta)
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
                theta = q[i] + offset[i]
                T_i = dh_transform(a[i], alpha[i], d[i], theta)
                T = T @ T_i
            return T[:3, 3]

        # 关节限位
        limits = [(-2.9, 2.9)] * 7

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
            Float32MultiArray, left_in, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            Float32MultiArray, right_in, self.right_callback, 10
        )
        self.left_pub = self.create_publisher(Float32MultiArray, left_out, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, right_out, 10)
        self.left_joint_status_sub = self.create_subscription(
            Float32MultiArray, left_joint_status, self.left_joint_status_callback, 10
        )
        self.right_joint_status_sub = self.create_subscription(
            Float32MultiArray, right_joint_status, self.right_joint_status_callback, 10
        )
        self.get_logger().info("arm_ik_node started with DH model")
        self.get_logger().info(
            f"IK arm_points_are_relative={self.arm_points_are_relative}, "
            f"ik_tolerance={self.ik_tolerance}, max_target_delta={self.max_target_delta}, "
            f"camera_to_robot_matrix={self.camera_to_robot_matrix.tolist()}"
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
        if not self.left_joint_ready:
            self.get_logger().warning("left joint status is not ready, skip IK", throttle_duration_sec=1.0)
            return

        if len(msg.data) != 9:
            self.get_logger().warning(f"Left arm data length error: {len(msg.data)}")
            return

        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        q_seed = self.q_left
        home_tcp = self.get_home_tcp("left", q_seed, self.ik_solver_left)
        T_target, elbow_target, e_rel, w_rel, e_delta, w_delta = self.build_target_pose(s, e, w, home_tcp)
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

        joint_msg = Float32MultiArray()
        joint_msg.data = q_result.tolist()
        self.left_pub.publish(joint_msg)

    def right_callback(self, msg):
        if not self.right_joint_ready:
            self.get_logger().warning("right joint status is not ready, skip IK", throttle_duration_sec=1.0)
            return

        if len(msg.data) != 9:
            self.get_logger().warning(f"Right arm data length error: {len(msg.data)}")
            return

        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        q_seed = self.q_right
        home_tcp = self.get_home_tcp("right", q_seed, self.ik_solver_right)
        T_target, elbow_target, e_rel, w_rel, e_delta, w_delta = self.build_target_pose(s, e, w, home_tcp)
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

        joint_msg = Float32MultiArray()
        joint_msg.data = q_result.tolist()
        self.right_pub.publish(joint_msg)

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
