#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class IKTestPublisher(Node):
    def __init__(self):
        super().__init__("ik_test_publisher")

        # ==================== 测试专用参数 START ====================
        # 这个节点只负责给 position_to_angle.py 发布假的 /left_arm 和 /right_arm。
        # 数据格式: [shoulder xyz, elbow xyz, wrist xyz]，单位保持为米。
        self.declare_parameter("left_arm_topic", "/left_arm")
        self.declare_parameter("right_arm_topic", "/right_arm")
        self.declare_parameter("publish_left", True)
        self.declare_parameter("publish_right", False)
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter(
            "left_arm_data",
            [0.0, 0.0, 0.0, 0.15, 0.1, 0.1, 0.35, 0.1, 0.15],
        )
        self.declare_parameter(
            "right_arm_data",
            [0.0, 0.0, 0.0, 0.15, -0.1, 0.1, 0.35, -0.1, 0.15],
        )
        # ==================== 测试专用参数 END ====================

        self.left_arm_topic = self.get_parameter("left_arm_topic").value
        self.right_arm_topic = self.get_parameter("right_arm_topic").value
        self.publish_left = bool(self.get_parameter("publish_left").value)
        self.publish_right = bool(self.get_parameter("publish_right").value)
        publish_rate = float(self.get_parameter("publish_rate").value)

        self.left_arm_data = self.read_arm_data("left_arm_data")
        self.right_arm_data = self.read_arm_data("right_arm_data")

        self.left_pub = self.create_publisher(Float32MultiArray, self.left_arm_topic, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, self.right_arm_topic, 10)

        period = 1.0 / publish_rate if publish_rate > 0.0 else 1.0
        self.timer = self.create_timer(period, self.publish_callback)

        self.get_logger().warning("IK TEST publisher started")
        self.get_logger().info(f"left: enabled={self.publish_left}, topic={self.left_arm_topic}, data={self.left_arm_data}")
        self.get_logger().info(f"right: enabled={self.publish_right}, topic={self.right_arm_topic}, data={self.right_arm_data}")

    def read_arm_data(self, param_name):
        data = list(self.get_parameter(param_name).value)
        if len(data) != 9:
            self.get_logger().warning(f"{param_name} length error, use zeros: {data}")
            return [0.0] * 9
        return [float(x) for x in data]

    def publish_callback(self):
        if self.publish_left:
            self.left_pub.publish(Float32MultiArray(data=self.left_arm_data))
        if self.publish_right:
            self.right_pub.publish(Float32MultiArray(data=self.right_arm_data))


def main(args=None):
    rclpy.init(args=args)
    node = IKTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
