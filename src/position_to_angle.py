#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R


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


# ==================== 滤波节点（发布稳定后的手臂点）====================
class ArmPointFilterNode(Node):
    def __init__(self):
        super().__init__("arm_point_filter_node")

        self.declare_parameter("input_topic", "/image_raw")
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

    def callback(self, msg):
        data = list(msg.data)
        if len(data) != self.expected_len:
            self.get_logger().warning(
                f"length error: {len(data)}, expected: {self.expected_len}",
                throttle_duration_sec=1.0
            )
            return

        person_count = int(data[0])
        if person_count == 0:
            self.get_logger().warning("no person detected", throttle_duration_sec=1.0)
            return
        if person_count > 1:
            self.get_logger().warning(
                f"multiple persons ({person_count}), only one supported",
                throttle_duration_sec=1.0
            )
            return

        filtered_points = {}
        for name, kp_id in self.target_points.items():
            raw_pos = self.get_keypoint_xyz(data, kp_id)
            if not self.is_valid_xyz(raw_pos):
                self.get_logger().warning(f"{name} invalid xyz: {raw_pos}", throttle_duration_sec=1.0)
                filtered_points[name] = self.filters[name].get_position() if self.filters[name].initialized else np.zeros(3)
                continue
            filtered_pos, valid = self.filter_one_point(name, raw_pos)
            if not valid:
                self.get_logger().warning(f"{name} rejected by Kalman gate", throttle_duration_sec=1.0)
            filtered_points[name] = filtered_pos

        self.left_pub.publish(self.make_arm_msg(filtered_points, self.left_arm_order))
        self.right_pub.publish(self.make_arm_msg(filtered_points, self.right_arm_order))


# ==================== 逆运动学求解器（你需要实现三个模型函数）====================
class IKSolver:
    def __init__(self, fk_func, jac_func, elbow_pos_func,
                 joint_limits=None, damping=0.05, tol=1e-4,
                 max_iter=100, alpha_sec=0.1):
        self.fk = fk_func
        self.jacobian = jac_func
        self.elbow_pos = elbow_pos_func
        self.joint_limits = joint_limits
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_sec = alpha_sec

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
        for i in range(self.max_iter):
            T_curr = self.fk(q)
            err = self.pose_error(T_curr, T_target)
            if np.linalg.norm(err[:3]) < self.tol and np.linalg.norm(err[3:]) < self.tol:
                self.get_logger().info(f"IK converged in {i} iterations")
                return self.check_joint_limits(q)

            J = self.jacobian(q)
            lambda_sq = self.damping ** 2
            J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
            dq_task = J_dls @ err

            if elbow_target is not None:
                p_elbow = self.elbow_pos(q)
                grad = self._elbow_gradient(q, elbow_target)
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq_sec = N @ (self.alpha_sec * grad)
            else:
                dq_sec = 0.0

            q = q + dq_task + dq_sec
            q = self.check_joint_limits(q)

        self.get_logger().warning("IK did not converge, returning current solution")
        return self.check_joint_limits(q)

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


# ==================== IK 节点：将手臂点转为关节角 ====================
class ArmIKNode(Node):
    def __init__(self):
        super().__init__("arm_ik_node")

        # ------- 参数 -------
        self.declare_parameter("left_input_topic", "/left_arm")
        self.declare_parameter("right_input_topic", "/right_arm")
        self.declare_parameter("left_output_topic", "/left_joint_angles")
        self.declare_parameter("right_output_topic", "/right_joint_angles")

        left_in = self.get_parameter("left_input_topic").value
        right_in = self.get_parameter("right_input_topic").value
        left_out = self.get_parameter("left_output_topic").value
        right_out = self.get_parameter("right_output_topic").value

        # ------- 机器人模型（你需要实现这三个函数）-------
        # 示例：仅作为占位，请替换为你自己的模型
        def my_fk(q):
            return np.eye(4)  # TODO: 实现正运动学
        def my_jac(q):
            return np.zeros((6, 7))  # TODO: 实现雅可比
        def my_elbow(q):
            return np.zeros(3)  # TODO: 返回肘部在基座下的坐标

        # ------- 关节限位（示例，按实际修改）-------
        limits = [(-2.9, 2.9)] * 7

        # 创建左右臂各自独立的 IK 求解器（通常同一机器人左右臂对称）
        self.ik_solver_left = IKSolver(
            fk_func=my_fk,
            jac_func=my_jac,
            elbow_pos_func=my_elbow,
            joint_limits=limits,
            damping=0.1,
            max_iter=200
        )
        self.ik_solver_right = IKSolver(
            fk_func=my_fk,
            jac_func=my_jac,
            elbow_pos_func=my_elbow,
            joint_limits=limits,
            damping=0.1,
            max_iter=200
        )

        # 初始关节角（机器人当前角度，最好从真实机器人获取）
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

        # ------- 订阅 & 发布 -------
        self.left_sub = self.create_subscription(
            Float32MultiArray, left_in, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            Float32MultiArray, right_in, self.right_callback, 10
        )
        self.left_pub = self.create_publisher(Float32MultiArray, left_out, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, right_out, 10)

        self.get_logger().info("arm_ik_node started")

    # ---------- 工具：由肩、肘、腕构建末端目标位姿 ----------
    def build_target_pose(self, shoulder, elbow, wrist):
        """
        简易位姿构建：
        - 末端位置 = 腕部位置
        - 姿态：Z 轴指向从腕部到手的延伸方向（这里简单设为指向下方，可按需修改）
                X 轴指向肩到腕的方向（投影到水平面）
        注意：这只是一个示例，实际应用中应根据抓取任务设计姿态。
        """
        T = np.eye(4)
        T[:3, 3] = wrist  # 位置

        # 方向向量（基座坐标系）
        v_sw = wrist - shoulder
        # 如果肩腕距离过小，返回单位阵
        if np.linalg.norm(v_sw) < 0.01:
            return T

        # X轴：肩到腕的水平投影方向
        x_dir = v_sw.copy()
        x_dir[2] = 0.0  # 忽略 Z 分量，保持水平
        x_norm = np.linalg.norm(x_dir)
        if x_norm < 0.001:
            x_dir = np.array([1.0, 0.0, 0.0])
        else:
            x_dir /= x_norm

        # Z轴：指向下方（可根据任务改为腕→指尖方向等）
        z_dir = np.array([0.0, 0.0, -1.0])

        # Y轴：右手定则
        y_dir = np.cross(z_dir, x_dir)
        y_norm = np.linalg.norm(y_dir)
        if y_norm < 0.001:
            # 处理退化情况
            y_dir = np.array([0.0, 1.0, 0.0])
            z_dir = np.cross(x_dir, y_dir)
        else:
            y_dir /= y_norm
            z_dir = np.cross(x_dir, y_dir)

        T[:3, 0] = x_dir
        T[:3, 1] = y_dir
        T[:3, 2] = z_dir
        return T

    # ---------- 回调：处理左臂 ----------
    def left_callback(self, msg):
        if len(msg.data) != 9:
            self.get_logger().warning(f"Left arm data length error: {len(msg.data)}")
            return

        # 解析: shoulder_x, shoulder_y, shoulder_z, elbow_x, ..., wrist_z
        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        T_target = self.build_target_pose(s, e, w)
        q_result = self.ik_solver_left.solve(T_target, self.q_left, elbow_target=e)
        self.q_left = q_result  # 保存作为下一帧初值

        # 发布关节角度
        joint_msg = Float32MultiArray()
        joint_msg.data = q_result.tolist()
        self.left_pub.publish(joint_msg)

    # ---------- 回调：处理右臂 ----------
    def right_callback(self, msg):
        if len(msg.data) != 9:
            self.get_logger().warning(f"Right arm data length error: {len(msg.data)}")
            return

        s = np.array(msg.data[0:3])
        e = np.array(msg.data[3:6])
        w = np.array(msg.data[6:9])

        T_target = self.build_target_pose(s, e, w)
        q_result = self.ik_solver_right.solve(T_target, self.q_right, elbow_target=e)
        self.q_right = q_result

        joint_msg = Float32MultiArray()
        joint_msg.data = q_result.tolist()
        self.right_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)

    filter_node = ArmPointFilterNode()
    ik_node = ArmIKNode()

    executor = rclpy.executors.MultiThreadedExecutor()
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