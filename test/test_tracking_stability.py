"""Tests for controller and RViz output stability."""

from collections import deque
import threading
import time
from types import SimpleNamespace

from builtin_interfaces.msg import Time
import numpy as np

from demo2.arm_sides import REQUIRED_LANDMARK_INDICES
from demo2.depth_arm_controller import (
    DEFAULT_DISPLAY_JOINTS,
    DepthArmController,
    robust_bone_length_baseline,
    smooth_joint_target,
)
from demo2.dual_joint_state_publisher import (
    external_joint_source_present,
)


def test_static_pose_yields_to_external_tracking_source():
    """The RViz fallback cannot fight with a tracking controller."""
    assert not external_joint_source_present([1, 1])
    assert external_joint_source_present([2, 1])


def test_uninitialized_rviz_pose_is_not_all_zero():
    """Before tracking, RViz uses the visible default pose rather than zero."""
    assert not np.allclose(DEFAULT_DISPLAY_JOINTS, 0.0)
    assert np.isclose(DEFAULT_DISPLAY_JOINTS[1], 1.5707963)


def test_joint_filter_applies_deadband_and_time_smoothing():
    """Small jitter is held while intentional joint motion stays responsive."""
    previous = np.zeros(2)
    tiny = np.deg2rad([0.1, -0.1])
    proposal = np.deg2rad([10.0, -10.0])

    held = smooth_joint_target(
        previous, tiny, 0.1, 0.2, np.deg2rad(0.35), np.deg2rad(120.0)
    )
    moved = smooth_joint_target(
        previous,
        proposal,
        0.1,
        0.2,
        np.deg2rad(0.35),
        np.deg2rad(120.0),
    )

    np.testing.assert_allclose(held, 0.0)
    assert np.all(np.abs(moved) > 0.0)
    assert np.all(np.abs(moved) < np.abs(proposal))


def test_point_filter_uses_a_short_median_before_ema():
    """Each arm attenuates noise without sharing filter state."""
    controller = object.__new__(DepthArmController)
    controller.side_previous_points = {"left": None, "right": None}
    controller.side_point_history = {
        side: deque(maxlen=3) for side in ("left", "right")
    }
    controller.max_point_jump = 1.0
    controller.smoothing_alpha = 0.5
    zeros = np.zeros((8, 3), dtype=float)
    noisy = np.full((8, 3), 0.2, dtype=float)

    controller.filter_side_points(zeros, "left")
    filtered = controller.filter_side_points(noisy, "left")

    left_indices = list(REQUIRED_LANDMARK_INDICES["left"])
    np.testing.assert_allclose(filtered[left_indices], 0.05)
    assert controller.side_previous_points["right"] is None


def test_bone_baseline_waits_for_multiple_frames_and_uses_median():
    """One bad first depth frame cannot poison all future tracking."""
    good = np.asarray([0.30, 0.27, 0.31, 0.28])
    samples = [good + 0.002 * index for index in range(7)]
    assert robust_bone_length_baseline(samples) is None

    samples.insert(0, np.asarray([0.55, 0.50, 0.52, 0.48]))
    baseline = robust_bone_length_baseline(samples)

    np.testing.assert_allclose(baseline, good + 0.006, atol=0.01)


def test_rejected_ik_keeps_unpublished_warm_start_progress():
    """A large valid motion can converge instead of rejecting forever."""
    class Solver:
        def __init__(self):
            self.previous = np.zeros(7)

        def solve(self, _upper, _forearm):
            self.previous = np.ones(7)
            return self.previous.copy(), np.asarray([35.0, 30.0])

    controller = SimpleNamespace(
        reference_origin=np.zeros(3),
        reference_basis=np.eye(3),
        corrections={"left": np.eye(3)},
        solvers={"left": Solver()},
        max_direction_error=25.0,
        set_invalid=lambda *_args: None,
        publish_side_status=lambda *_args: None,
    )
    points = np.asarray([
        [-0.2, 0.0, 2.0],
        [-0.4, 0.1, 2.0],
        [-0.6, 0.2, 2.0],
        [-0.15, 0.5, 2.0],
        [0.2, 0.0, 2.0],
        [0.4, 0.1, 2.0],
        [0.6, 0.2, 2.0],
        [0.15, 0.5, 2.0],
    ])

    assert not DepthArmController.solve_side(controller, points, "left")
    np.testing.assert_allclose(controller.solvers["left"].previous, 1.0)


def test_left_and_right_command_gates_are_independent():
    """One invalid side cannot stop the opposite side's command topic."""
    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    controller = SimpleNamespace()
    controller.data_lock = threading.Lock()
    controller.latest_valid = {"left": True, "right": False}
    controller.last_valid_time = {
        "left": time.monotonic(),
        "right": None,
    }
    controller.pose_timeout = 0.35
    controller.latest_joints = {
        "left": np.ones(7),
        "right": np.full(7, 2.0),
    }
    controller.target_joints = {
        side: joints.copy()
        for side, joints in controller.latest_joints.items()
    }
    controller.has_joint_solution = {"left": True, "right": True}
    controller.last_publish_filter_time = time.monotonic() - 0.05
    controller.joint_smoothing_tau = 0.2
    controller.joint_deadband = np.deg2rad(0.35)
    controller.max_joint_speed = np.deg2rad(120.0)
    controller.joint_limits = np.asarray([[-3.0, 3.0]] * 7)
    controller.initial_display_positions = np.zeros(7)
    controller.joint_names = [f"joint{index}" for index in range(1, 8)]
    controller.publish_joint_states_enabled = True
    controller.command_output_enabled = True
    controller.joint_publishers = {
        "left": Publisher(),
        "right": Publisher(),
    }
    controller.command_publishers = {
        "left": Publisher(),
        "right": Publisher(),
    }
    controller.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time())
    )

    DepthArmController.publish_latest(controller)

    assert len(controller.joint_publishers["left"].messages) == 1
    assert len(controller.joint_publishers["right"].messages) == 1
    assert len(controller.command_publishers["left"].messages) == 1
    assert len(controller.command_publishers["right"].messages) == 0
