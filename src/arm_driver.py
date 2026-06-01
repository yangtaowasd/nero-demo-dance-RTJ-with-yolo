#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from arm_and_revo2 import *

class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")

        # 参数声明
        self.declare_parameter("angle_topic", "/angle_topic")
        self.declare_parameter("can_interface", "can0")      # 新增：CAN 接口名
        self.declare_parameter("joint_limits", [             # 新增：关节限位（可选）
            [-2.96, 2.96], [-2.96, 2.96], [-2.96, 2.96],
            [-2.96, 2.96], [-2.96, 2.96], [-2.96, 2.96],
            [-2.96, 2.96]
        ])

        self.sub_topic = self.get_parameter("angle_topic").value
        self.can_iface = self.get_parameter("can_interface").value
        self.limits = self.get_parameter("joint_limits").value

        # 订阅关节角
        self.angle_sub = self.create_subscription(
            Float32MultiArray,
            self.sub_topic,
            self.pose_sub_callback,
            10
        )

        # 使用参数化的 CAN 接口初始化机械臂
        self.arm = Nero(self.can_iface)
        self.arm_enable()

        self.get_logger().info(f"机械臂驱动节点启动")
        self.get_logger().info(f"  CAN 接口: {self.can_iface}")
        self.get_logger().info(f"  订阅话题: {self.sub_topic}")

    def arm_enable(self):
        try:
            self.arm.connect()
            self.arm.read_all()
            self.arm.enable()
            self.get_logger().info("机械臂已连接并使能")
        except Exception as e:
            self.get_logger().error(f"机械臂使能失败: {e}")

    def arm_disable(self):
        try:
            self.arm.disable()
            self.get_logger().info("机械臂已去使能")
        except Exception as e:
            self.get_logger().error(f"去使能失败: {e}")

    def pose_sub_callback(self, msg):
        angle_data = list(msg.data)

        if len(angle_data) != 7:
            self.get_logger().warn(f"关节角数量错误: {len(angle_data)}，期望7")
            return

        # 限位检查与钳位
        for i, (ang, (lo, hi)) in enumerate(zip(angle_data, self.limits)):
            if ang < lo or ang > hi:
                self.get_logger().warn(f"关节{i+1}角度 {ang:.3f} 超限，已钳位")
                angle_data[i] = max(lo, min(hi, ang))

        try:
            self.arm.move_j(angle_data)
            self.get_logger().info(
                f"执行关节角: {[f'{a:.2f}' for a in angle_data]}",
                throttle_duration_sec=0.5
            )
        except Exception as e:
            self.get_logger().error(f"运动指令失败: {e}")

    def destroy_node(self):
        self.arm_disable()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from arm_and_revo2 import *

class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")

        # 参数声明
        self.declare_parameter("angle_topic", "/angle_topic")
        self.declare_parameter("can_interface", "can0")      # 新增：CAN 接口名
        self.declare_parameter("joint_limits", [             # 新增：关节限位（可选）
            [-2.96, 2.96], [-2.96, 2.96], [-2.96, 2.96],
            [-2.96, 2.96], [-2.96, 2.96], [-2.96, 2.96],
            [-2.96, 2.96]
        ])

        self.sub_topic = self.get_parameter("angle_topic").value
        self.can_iface = self.get_parameter("can_interface").value
        self.limits = self.get_parameter("joint_limits").value

        # 订阅关节角
        self.angle_sub = self.create_subscription(
            Float32MultiArray,
            self.sub_topic,
            self.pose_sub_callback,
            10
        )

        # 使用参数化的 CAN 接口初始化机械臂
        self.arm = Nero(self.can_iface)
        self.arm_enable()

        self.get_logger().info(f"机械臂驱动节点启动")
        self.get_logger().info(f"  CAN 接口: {self.can_iface}")
        self.get_logger().info(f"  订阅话题: {self.sub_topic}")

    def arm_enable(self):
        try:
            self.arm.connect()
            self.arm.read_all()
            self.arm.enable()
            self.get_logger().info("机械臂已连接并使能")
        except Exception as e:
            self.get_logger().error(f"机械臂使能失败: {e}")

    def arm_disable(self):
        try:
            self.arm.disable()
            self.get_logger().info("机械臂已去使能")
        except Exception as e:
            self.get_logger().error(f"去使能失败: {e}")

    def pose_sub_callback(self, msg):
        angle_data = list(msg.data)

        if len(angle_data) != 7:
            self.get_logger().warn(f"关节角数量错误: {len(angle_data)}，期望7")
            return

        # 限位检查与钳位
        for i, (ang, (lo, hi)) in enumerate(zip(angle_data, self.limits)):
            if ang < lo or ang > hi:
                self.get_logger().warn(f"关节{i+1}角度 {ang:.3f} 超限，已钳位")
                angle_data[i] = max(lo, min(hi, ang))

        try:
            self.arm.move_j(angle_data)
            self.get_logger().info(
                f"执行关节角: {[f'{a:.2f}' for a in angle_data]}",
                throttle_duration_sec=0.5
            )
        except Exception as e:
            self.get_logger().error(f"运动指令失败: {e}")

    def destroy_node(self):
        self.arm_disable()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()