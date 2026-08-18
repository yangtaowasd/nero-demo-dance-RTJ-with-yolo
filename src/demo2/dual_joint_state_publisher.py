#!/usr/bin/env python3
"""Publish a fallback dual-Nero pose for RViz-only visualization."""

import os
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from demo2.arm_sides import DEFAULT_JOINT_STATE_TOPICS
from demo2.instance_guard import acquire_instance_lock


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
DEFAULT_POSITIONS = (0.0, 1.5707963, 0.0, 0.0, 0.0, 0.0, 0.0)


def external_joint_source_present(publisher_counts):
    """Return whether another source is publishing either joint topic."""
    return any(int(count) > 1 for count in publisher_counts)


def joint_state_message(stamp, positions=DEFAULT_POSITIONS):
    """Build one complete Nero joint-state message."""
    positions = tuple(float(value) for value in positions)
    if len(positions) != len(JOINT_NAMES):
        raise ValueError("expected seven Nero joint positions")
    message = JointState()
    message.header.stamp = stamp
    message.name = list(JOINT_NAMES)
    message.position = list(positions)
    return message


class DualJointStatePublisher(Node):
    """Publish a static pose only when no tracking controller is active."""

    def __init__(self):
        """Configure fallback topics and collision detection."""
        super().__init__("dual_joint_state_publisher")
        self.declare_parameter(
            "left_topic", DEFAULT_JOINT_STATE_TOPICS["left"]
        )
        self.declare_parameter(
            "right_topic", DEFAULT_JOINT_STATE_TOPICS["right"]
        )
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("startup_grace_sec", 0.75)
        self.declare_parameter(
            "initial_positions", list(DEFAULT_POSITIONS)
        )
        self.positions = tuple(
            self.get_parameter("initial_positions").value
        )
        if len(self.positions) != len(JOINT_NAMES):
            raise ValueError("initial_positions must contain seven values")
        self.joint_topics = [
            str(self.get_parameter(name).value)
            for name in ("left_topic", "right_topic")
        ]
        self.joint_publishers = [
            self.create_publisher(JointState, topic, 10)
            for topic in self.joint_topics
        ]
        rate = max(
            float(self.get_parameter("publish_rate_hz").value), 1.0
        )
        self.startup_grace = max(
            float(self.get_parameter("startup_grace_sec").value), 0.0
        )
        self.start_time = time.monotonic()
        self.yielding_to_external_source = False
        self.create_timer(1.0 / rate, self.publish_pose)
        self.get_logger().info(
            "RViz fallback pose ready; no robot commands are published"
        )

    def publish_pose(self):
        """Publish unless an external tracking source owns the topics."""
        if time.monotonic() - self.start_time < self.startup_grace:
            return
        counts = [
            self.count_publishers(topic) for topic in self.joint_topics
        ]
        external_source = external_joint_source_present(counts)
        if external_source:
            if not self.yielding_to_external_source:
                self.get_logger().warning(
                    "tracking joint-state source detected; fallback pose "
                    "is yielding to prevent RViz switching"
                )
            self.yielding_to_external_source = True
            return
        if self.yielding_to_external_source:
            self.get_logger().info(
                "tracking source stopped; fallback RViz pose resumed"
            )
        self.yielding_to_external_source = False
        stamp = self.get_clock().now().to_msg()
        for publisher in self.joint_publishers:
            publisher.publish(joint_state_message(stamp, self.positions))


def main(args=None):
    """Run the conflict-safe RViz fallback publisher."""
    try:
        lock_descriptor = acquire_instance_lock()
    except RuntimeError as exc:
        print(f"dual display not started: {exc}", file=sys.stderr)
        return
    rclpy.init(args=args)
    node = None
    try:
        node = DualJointStatePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        os.close(lock_descriptor)


if __name__ == "__main__":
    main()
