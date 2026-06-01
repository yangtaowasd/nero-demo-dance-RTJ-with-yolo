#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from arm_and_revo2 import *


class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")

        self.declare_parameter("angle_topic", "/angle_topic")
        self.declare_parameter("joint_status_topic", "/joint_status_topic")
        self.declare_parameter("can_interface", "can0")

        self.sub_topic = self.get_parameter("angle_topic").value
        self.joint_status_topic = self.get_parameter("joint_status_topic").value
        self.can_iface = self.get_parameter("can_interface").value

        # 关节限位，单位 rad
        self.limits = [
            [-2.96, 2.96],
            [-2.96, 2.96],
            [-2.96, 2.96],
            [-2.96, 2.96],
            [-2.96, 2.96],
            [-2.96, 2.96],
            [-2.96, 2.96],
        ]

        self.angle_sub = self.create_subscription(
            Float32MultiArray,
            self.sub_topic,
            self.pose_sub_callback,
            10
        )

        self.joint_status_pub = self.create_publisher(
            Float32MultiArray,
            self.joint_status_topic,
            10
        )

        self.arm_ready = False
        self.arm = Nero(channel=self.can_iface)
        self.arm_enable()

        self.timer = self.create_timer(0.05, self.arm_read)

        self.get_logger().info("arm_driver_node started")
        self.get_logger().info(f"CAN: {self.can_iface}")
        self.get_logger().info(f"sub: {self.sub_topic}")
        self.get_logger().info(f"pub: {self.joint_status_topic}")

    def arm_enable(self):
        try:
            self.arm.connect()
            self.arm.enable()
            self.arm_ready = True
            self.get_logger().info("arm connected and enabled")
        except Exception as e:
            self.arm_ready = False
            self.get_logger().error(f"arm enable failed: {e}")

    def arm_disable(self):
        if not self.arm_ready:
            return
        try:
            self.arm.disable()
            self.arm_ready = False
            self.get_logger().info("arm disabled")
        except Exception as e:
            self.get_logger().error(f"arm disable failed: {e}")

    def pose_sub_callback(self, msg):
        if not self.arm_ready:
            self.get_logger().warn("arm is not ready, skip move_j", throttle_duration_sec=1.0)
            return

        angle_data = list(msg.data)

        if len(angle_data) != 7:
            self.get_logger().warn(f"angle_data length error: {len(angle_data)}")
            return

        for i in range(7):
            lo = self.limits[i][0]
            hi = self.limits[i][1]

            if angle_data[i] < lo:
                self.get_logger().warn(f"joint {i + 1} too small, clamp")
                angle_data[i] = lo

            if angle_data[i] > hi:
                self.get_logger().warn(f"joint {i + 1} too large, clamp")
                angle_data[i] = hi

        try:
            self.arm.move_j(angle_data)
            self.get_logger().info(f"move_j: {angle_data}")
        except Exception as e:
            self.get_logger().error(f"move_j failed: {e}")

    def arm_read(self):
        if not self.arm_ready:
            return

        try:
            joint_angles = self.arm.get_joint_angles().msg

            if joint_angles is None:
                self.get_logger().warn("joint_angles is None")
                return

            status_msg = Float32MultiArray()
            status_msg.data = list(joint_angles)

            self.joint_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"read joint failed: {e}")

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


if __name__ == "__main__":
    main()
