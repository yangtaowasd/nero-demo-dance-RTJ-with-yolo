#!/usr/bin/env python3
"""Compatibility entry point running split RGB-D detector/controller nodes."""

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from demo2.depth_arm_controller import DepthArmController
from demo2.depth_pose_detector import DepthPoseDetector


def main(args=None):
    """Run both split nodes for users of the former monolithic entry point."""
    rclpy.init(args=args)
    detector = DepthPoseDetector()
    controller = DepthArmController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(detector)
    executor.add_node(controller)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        detector.destroy_node()
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
