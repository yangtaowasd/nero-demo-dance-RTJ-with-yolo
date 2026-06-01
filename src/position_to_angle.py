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

        # 状态量:
        # [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))

        # 状态协方差，表示当前状态的不确定性
        self.P = np.eye(6) * 0.1

        self.initialized = False

        # 状态转移矩阵
        # x_new = x + vx * dt
        # y_new = y + vy * dt
        # z_new = z + vz * dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # 过程噪声
        # sigma_a 越大，反应越快，但更容易抖
        # sigma_a 越小，更平滑，但延迟更大
        q = np.array([
            [dt**4 / 4, 0, 0, dt**3 / 2, 0, 0],
            [0, dt**4 / 4, 0, 0, dt**3 / 2, 0],
            [0, 0, dt**4 / 4, 0, 0, dt**3 / 2],
            [dt**3 / 2, 0, 0, dt**2, 0, 0],
            [0, dt**3 / 2, 0, 0, dt**2, 0],
            [0, 0, dt**3 / 2, 0, 0, dt**2]
        ]) * sigma_a**2
        self.Q = q

        # 观测矩阵
        # 输入只有 [x, y, z]
        # 速度 vx, vy, vz 不是直接观测出来的
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 观测噪声
        # sigma_pos 是 x/y 的测量噪声
        # sigma_z 是 z 的测量噪声
        # 一般 z 方向更不稳定，所以 sigma_z 可以大一点
        self.R = np.diag([
            sigma_pos**2,
            sigma_pos**2,
            sigma_z**2
        ])

    def initialize(self, pos):
        """
        第一次收到位置时初始化滤波器
        pos: [x, y, z]
        """
        self.x[:3] = np.array(pos, dtype=float).reshape(3, 1)
        self.x[3:] = 0.0
        self.initialized = True

    def predict(self):
        """
        根据上一帧状态预测当前状态
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].flatten()

    def update(self, z):
        """
        使用当前检测值修正预测结果
        z: [x, y, z]
        """
        z = np.asarray(z, dtype=float).reshape(3, 1)

        # 新息，也就是：检测值 - 预测值
        y = z - self.H @ self.x

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 马氏距离，用来判断是否是异常跳点
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x[:3].flatten(), False

        d = np.sqrt(y.T @ S_inv @ y).item()

        # 阈值越小，越容易拒绝跳点
        # 阈值越大，越容易接受检测值
        if d > 5.0:
            return self.x[:3].flatten(), False

        # 卡尔曼增益
        K = self.P @ self.H.T @ S_inv

        # 状态更新
        self.x = self.x + K @ y

        # 协方差更新
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3].flatten(), True

    def get_position(self):
        return self.x[:3].flatten()


# ==================== ROS2 节点 ====================
class ArmPointFilterNode(Node):
    def __init__(self):
        super().__init__("arm_point_filter_node")

        # ==================== 参数 ====================
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

        # ==================== 输入数据格式 ====================
        # data[0] = person_count
        # 后面是 17 个关键点，每个关键点 3 个数: x, y, z
        #
        # 总长度:
        # 1 + 17 * 3 = 52
        self.num_keypoints = 17
        self.expected_len = 1 + self.num_keypoints * 3

        # ==================== 需要处理的关键点 ====================
        # YOLO Pose 常用编号:
        # 5  = left_shoulder
        # 6  = right_shoulder
        # 7  = left_elbow
        # 8  = right_elbow
        # 9  = left_wrist
        # 10 = right_wrist
        self.target_points = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
        }

        # 左臂输出顺序
        self.left_arm_order = [
            "left_shoulder",
            "left_elbow",
            "left_wrist",
        ]

        # 右臂输出顺序
        self.right_arm_order = [
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        ]

        # ==================== 为每个点创建一个滤波器 ====================
        self.filters = {}

        for name in self.target_points.keys():
            self.filters[name] = KalmanFilter3D(
                dt=self.dt,
                sigma_a=self.sigma_a,
                sigma_pos=self.sigma_pos,
                sigma_z=self.sigma_z
            )

        # ==================== ROS2 订阅和发布 ====================
        self.sub = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.callback,
            10
        )

        self.left_pub = self.create_publisher(
            Float32MultiArray,
            self.left_output_topic,
            10
        )

        self.right_pub = self.create_publisher(
            Float32MultiArray,
            self.right_output_topic,
            10
        )

        self.get_logger().info("arm_point_filter_node started")
        self.get_logger().info(f"input_topic: {self.input_topic}")
        self.get_logger().info(f"left_output_topic: {self.left_output_topic}")
        self.get_logger().info(f"right_output_topic: {self.right_output_topic}")
        self.get_logger().info(f"dt: {self.dt}")

    def get_keypoint_xyz(self, data, keypoint_id):
        """
        从 Float32MultiArray 中取出某个关键点的 [x, y, z]

        数据格式:
        data[0] = person_count

        data[1 + keypoint_id * 3 + 0] = x
        data[1 + keypoint_id * 3 + 1] = y
        data[1 + keypoint_id * 3 + 2] = z
        """
        base = 1 + keypoint_id * 3

        x = data[base + 0]
        y = data[base + 1]
        z = data[base + 2]

        return [x, y, z]

    def is_valid_xyz(self, xyz):
        """
        检查点是否有效
        防止 NaN / Inf 进入滤波器
        """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        if not np.isfinite(x):
            return False
        if not np.isfinite(y):
            return False
        if not np.isfinite(z):
            return False

        return True

    def filter_one_point(self, name, raw_pos):
        """
        对一个点做滤波
        name: 点名，例如 left_wrist
        raw_pos: 原始 [x, y, z]
        """
        kf = self.filters[name]

        # 第一次收到这个点，先初始化
        if not kf.initialized:
            kf.initialize(raw_pos)
            return np.asarray(raw_pos, dtype=float), True

        # 先预测，再更新
        kf.predict()
        filtered_pos, valid = kf.update(raw_pos)

        return filtered_pos, valid

    def make_arm_msg(self, filtered_points, arm_order):
        """
        根据指定顺序生成 Float32MultiArray

        arm_order 例子:
        [
            "left_shoulder",
            "left_elbow",
            "left_wrist"
        ]

        输出:
        [
            shoulder_x, shoulder_y, shoulder_z,
            elbow_x,    elbow_y,    elbow_z,
            wrist_x,    wrist_y,    wrist_z
        ]
        """
        msg = Float32MultiArray()

        out = []

        for name in arm_order:
            xyz = filtered_points[name]

            out.append(float(xyz[0]))
            out.append(float(xyz[1]))
            out.append(float(xyz[2]))

        msg.data = out
        return msg

    def callback(self, msg):
        data = list(msg.data)

        # ==================== 长度检查 ====================
        if len(data) != self.expected_len:
            self.get_logger().warn(
                f"length error: {len(data)}, expected: {self.expected_len}",
                throttle_duration_sec=1.0
            )
            return

        # ==================== 人数检查 ====================
        person_count = int(data[0])

        if person_count == 0:
            self.get_logger().warn(
                "cannot recognize human",
                throttle_duration_sec=1.0
            )
            return

        if person_count > 1:
            self.get_logger().warn(
                f"person_count error: {person_count}, only one person is allowed",
                throttle_duration_sec=1.0
            )
            return

        # ==================== 读取并滤波所有目标点 ====================
        filtered_points = {}

        for name, keypoint_id in self.target_points.items():
            raw_pos = self.get_keypoint_xyz(data, keypoint_id)

            if not self.is_valid_xyz(raw_pos):
                self.get_logger().warn(
                    f"{name} invalid xyz: {raw_pos}",
                    throttle_duration_sec=1.0
                )

                # 如果这个点已经初始化过，就继续使用上一帧的位置
                if self.filters[name].initialized:
                    filtered_points[name] = self.filters[name].get_position()
                else:
                    # 如果还没有初始化过，就用 0
                    filtered_points[name] = np.zeros(3)

                continue

            filtered_pos, valid = self.filter_one_point(name, raw_pos)

            if not valid:
                self.get_logger().warn(
                    f"{name} rejected by Kalman gate",
                    throttle_duration_sec=1.0
                )

            filtered_points[name] = filtered_pos

        # ==================== 发布左臂 ====================
        left_msg = self.make_arm_msg(
            filtered_points,
            self.left_arm_order
        )
        self.left_pub.publish(left_msg)

        # ==================== 发布右臂 ====================
        right_msg = self.make_arm_msg(
            filtered_points,
            self.right_arm_order
        )
        self.right_pub.publish(right_msg)


class pointtoangle:
    def __init__(self):
        self.right_direction = [1] * 7 
        self.left_direction = [1] * 7

    def position_to_angle(self):
        angle_data = [0.0 for _ in range(7)]

        
        return angle_data

class IKSolver:
    def __init__(self, fk_func, jac_func, elbow_pos_func,
                 joint_limits=None, damping=0.05, tol=1e-4,
                 max_iter=100, alpha_sec=0.1):
        """
        fk_func(q) -> 4x4 齐次矩阵 (np.array)
        jac_func(q) -> 6x7 雅可比矩阵 (np.array)
        elbow_pos_func(q) -> 3维肘部位置 (np.array)
        joint_limits: [(min, max), ...] 7对限位
        alpha_sec: 零空间优化步长
        """
        self.fk = fk_func
        self.jacobian = jac_func
        self.elbow_pos = elbow_pos_func
        self.joint_limits = joint_limits
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_sec = alpha_sec

    def pose_error(self, T_current, T_target):
        """返回 6 维误差向量 [dx, dy, dz, rx, ry, rz] (位置+轴角)"""
        p_curr = T_current[:3, 3]
        p_targ = T_target[:3, 3]
        pos_err = p_targ - p_curr

        R_curr = T_current[:3, :3]
        R_targ = T_target[:3, :3]
        R_err = R_targ @ R_curr.T
        rot_err = R.from_matrix(R_err).as_rotvec()  # 轴角

        return np.concatenate([pos_err, rot_err])

    def check_joint_limits(self, q):
        """将关节角度限制在允许范围内"""
        if self.joint_limits is None:
            return q
        q_clipped = np.clip(q,
                            [l[0] for l in self.joint_limits],
                            [l[1] for l in self.joint_limits])
        return q_clipped

    def solve(self, T_target, q_init, elbow_target=None):
        """
        主求解函数
        T_target: 4x4 目标末端位姿
        q_init: 初始关节角度 (7,)
        elbow_target: 可选，3维肘部目标位置 (来自YOLO)
        return: 求解出的关节角度 q (7,)
        """
        q = q_init.copy()
        for i in range(self.max_iter):
            # 正运动学
            T_curr = self.fk(q)
            err = self.pose_error(T_curr, T_target)

            # 收敛判断
            pos_err_norm = np.linalg.norm(err[:3])
            rot_err_norm = np.linalg.norm(err[3:])
            if pos_err_norm < self.tol and rot_err_norm < self.tol:
                print(f"IK 收敛于第 {i} 次迭代")
                return self.check_joint_limits(q)

            # 雅可比 (6x7)
            J = self.jacobian(q)

            # 阻尼最小二乘 (处理奇异)
            lambda_sq = self.damping ** 2
            J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
            dq_task = J_dls @ err

            # 零空间投影：次要任务 (肘部位置逼近)
            if elbow_target is not None:
                p_elbow_curr = self.elbow_pos(q)
                grad = self.elbow_pos_gradient(q, elbow_target)
                # 零空间投影矩阵
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq_sec = N @ (self.alpha_sec * grad)
            else:
                dq_sec = 0.0

            dq = dq_task + dq_sec

            # 更新并限位
            q = q + dq
            q = self.check_joint_limits(q)

        print("警告: IK 未收敛，返回当前解")
        return self.check_joint_limits(q)

    def elbow_pos_gradient(self, q, elbow_target):
        """
        计算肘部位置相对于关节角的梯度 ∂p_elbow/∂q，
        用于使肘部移向目标位置。
        这里使用有限差分，实际可替换为解析雅可比对应行。
        """
        eps = 1e-6
        p0 = self.elbow_pos(q)
        grad = np.zeros(7)
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            p1 = self.elbow_pos(q + dq)
            grad[i] = np.dot(p1 - p0, elbow_target - p0) / eps
        return grad


# ============ 使用示例：与 YOLO 肘部信息结合 ============
if __name__ == "__main__":
    # 假设用户已实现了自己的机器人模型函数
    # 此处用简易的伪模型演示结构

    def my_fk(q):
        # 实际应用中替换为真实的 DH 正运动学
        # 返回 4x4 矩阵
        return np.eye(4)  # 占位

    def my_jacobian(q):
        return np.zeros((6, 7))  # 占位

    def my_elbow_pos(q):
        return np.zeros(3)  # 占位

    # 关节限位 (示例)
    limits = [(-2.9, 2.9)] * 7

    solver = IKSolver(fk_func=my_fk,
                      jac_func=my_jacobian,
                      elbow_pos_func=my_elbow_pos,
                      joint_limits=limits,
                      damping=0.1,
                      max_iter=200)

    # 假设从 YOLO 获得的肘部 3D 位置 (已转换到机器人基座)
    elbow_3d_target = np.array([0.5, 0.2, 0.4])

    # 末端目标位姿 (例如根据腕部位置和抓取姿态构建)
    T_des = np.eye(4)
    T_des[:3, 3] = [0.6, 0.0, 0.3]  # 腕部位置
    # 姿态部分按需设置
    T_des[:3, :3] = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()

    q_init = np.zeros(7)
    q_result = solver.solve(T_des, q_init, elbow_target=elbow_3d_target)

    print("求解关节角：", q_result)
    print("J4 (肘关节):", q_result[3])
    print("J7 (末端姿态关节):", q_result[6])


def main(args=None):
    rclpy.init(args=args)

    node = ArmPointFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()