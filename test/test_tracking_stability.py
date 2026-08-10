"""Tests for controller and RViz output stability."""

from collections import deque
import os
import threading
import time
from types import SimpleNamespace

from builtin_interfaces.msg import Time
import numpy as np
import pytest

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
from demo2.instance_guard import (
    acquire_instance_lock,
    parent_process_changed,
    require_instance_available,
)


def test_launch_preflight_rejects_an_active_controller(tmp_path):
    """A duplicate launch stops before camera and CAN nodes are created."""
    lock_path = str(tmp_path / "controller.lock")
    descriptor = acquire_instance_lock(lock_path)
    try:
        assert str(os.getpid()) in (tmp_path / "controller.lock").read_text()
        with pytest.raises(RuntimeError, match="already running"):
            require_instance_available(lock_path=lock_path)
    finally:
        os.close(descriptor)
    assert require_instance_available(lock_path=lock_path) == []


def test_parent_change_detects_an_orphaned_ros_node():
    """Controller and CAN drivers can release resources with dead launch."""
    assert not parent_process_changed(123, 123)
    assert parent_process_changed(123, 1)


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


def test_joint_filter_reacts_faster_only_for_deliberate_motion():
    """Adaptive smoothing keeps the deadband but reduces motion lag."""
    previous = np.zeros(2)
    proposal = np.deg2rad([10.0, -10.0])
    arguments = (
        previous,
        proposal,
        0.05,
        0.20,
        np.deg2rad(0.35),
        np.deg2rad(120.0),
    )

    fixed = smooth_joint_target(*arguments)
    adaptive = smooth_joint_target(
        *arguments,
        fast_smoothing_tau=0.04,
        adaptive_motion_start_rad=np.deg2rad(1.0),
        adaptive_motion_full_rad=np.deg2rad(8.0),
    )
    tiny = smooth_joint_target(
        previous,
        np.deg2rad([0.1, -0.1]),
        0.05,
        0.20,
        np.deg2rad(0.35),
        np.deg2rad(120.0),
        fast_smoothing_tau=0.04,
        adaptive_motion_start_rad=np.deg2rad(1.0),
        adaptive_motion_full_rad=np.deg2rad(8.0),
    )

    assert np.all(np.abs(adaptive) > np.abs(fixed))
    np.testing.assert_allclose(tiny, 0.0)


def test_point_filter_uses_a_short_median_before_ema():
    """Each arm attenuates noise without sharing filter state."""
    controller = object.__new__(DepthArmController)
    controller.side_previous_points = {"left": None, "right": None}
    controller.side_point_history = {
        side: deque(maxlen=3) for side in ("left", "right")
    }
    controller.max_point_jump = 1.0
    controller.smoothing_alpha = 0.5
    controller.adaptive_point_filter_enabled = False
    controller.point_fast_smoothing_alpha = 0.85
    controller.point_motion_start = 0.015
    controller.point_motion_full = 0.060
    zeros = np.zeros((8, 3), dtype=float)
    noisy = np.full((8, 3), 0.2, dtype=float)

    controller.filter_side_points(zeros, "left")
    filtered = controller.filter_side_points(noisy, "left")

    left_indices = list(REQUIRED_LANDMARK_INDICES["left"])
    np.testing.assert_allclose(filtered[left_indices], 0.05)
    assert controller.side_previous_points["right"] is None


def test_point_filter_bypasses_median_lag_during_real_motion():
    """Fast movement uses the newest Kalman point; static filtering remains."""
    controller = object.__new__(DepthArmController)
    controller.side_previous_points = {"left": None, "right": None}
    controller.side_point_history = {
        side: deque(maxlen=3) for side in ("left", "right")
    }
    controller.max_point_jump = 1.0
    controller.smoothing_alpha = 0.30
    controller.adaptive_point_filter_enabled = True
    controller.point_fast_smoothing_alpha = 0.85
    controller.point_motion_start = 0.015
    controller.point_motion_full = 0.060
    zeros = np.zeros((8, 3), dtype=float)
    moved = np.full((8, 3), 0.10, dtype=float)

    controller.filter_side_points(zeros, "left")
    filtered = controller.filter_side_points(moved, "left")

    left_indices = list(REQUIRED_LANDMARK_INDICES["left"])
    assert np.all(filtered[left_indices] > 0.08)


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


def test_predicted_landmarks_are_visible_in_side_status():
    """Operators can distinguish Kalman motion from measured tracking."""
    class Solver:
        def solve(self, _upper, _forearm):
            return np.ones(7), np.asarray([2.0, 1.0])

    statuses = []
    controller = SimpleNamespace(
        reference_origin=np.zeros(3),
        reference_basis=np.eye(3),
        corrections={"left": np.eye(3)},
        solvers={"left": Solver()},
        max_direction_error=25.0,
        data_lock=threading.Lock(),
        target_joints={"left": np.zeros(7)},
        joint_limits=np.asarray([[-3.0, 3.0]] * 7),
        has_joint_solution={"left": False},
        latest_valid={"left": False},
        last_valid_time={"left": None},
        publish_side_status=lambda side, status: statuses.append(
            (side, status)
        ),
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

    assert DepthArmController.solve_side(
        controller, points, "left", predicted_count=2
    )
    assert statuses[0][0] == "left"
    assert "Kalman predicting 2 point(s)" in statuses[0][1]


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
    controller.adaptive_joint_smoothing_enabled = True
    controller.joint_fast_smoothing_tau = 0.04
    controller.joint_motion_start = np.deg2rad(1.0)
    controller.joint_motion_full = np.deg2rad(8.0)
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
