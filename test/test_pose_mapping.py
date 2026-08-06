import numpy as np
import pytest
import rclpy

from demo2.position_to_angle_v2 import PositionToAngleV2


def arm_points():
    points = np.zeros((8, 5), dtype=float)
    points[:, 4] = 0.99
    points[0, :2] = [600.0, 360.0]
    points[1, :2] = [450.0, 360.0]
    points[2, :2] = [300.0, 360.0]
    points[3, :2] = [250.0, 360.0]
    points[4, :2] = [680.0, 360.0]
    points[5, :2] = [830.0, 360.0]
    points[6, :2] = [980.0, 360.0]
    points[7, :2] = [1030.0, 360.0]
    return points


@pytest.fixture
def solver():
    rclpy.init()
    node = PositionToAngleV2()
    points = arm_points()
    node.yolo_baseline = node.make_yolo_baseline(points[:, :2])
    yield node
    node.destroy_node()
    rclpy.shutdown()


def rotate_chain(points, offset, angle):
    output = points.copy()
    shoulder = output[offset, :2].copy()
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    for index in range(offset + 1, offset + 4):
        output[index, :2] = shoulder + rotation @ (
            output[index, :2] - shoulder
        )
    return output


@pytest.mark.parametrize("side,offset", [("left", 0), ("right", 4)])
def test_rigid_straight_arm_rotation_does_not_create_elbow_flexion(
    solver, side, offset
):
    rotated = rotate_chain(arm_points(), offset, np.deg2rad(30.0))

    joints = solver.solve_yolo_side(side, rotated)

    assert np.rad2deg(joints[3]) == pytest.approx(0.0, abs=0.5)
    assert abs(np.rad2deg(joints[1])) == pytest.approx(30.0, abs=0.5)


@pytest.mark.parametrize("side", ["left", "right"])
def test_global_image_scale_does_not_create_forward_motion(solver, side):
    points = arm_points()
    image_center = np.asarray([640.0, 360.0])
    points[:, :2] = image_center + 0.75 * (points[:, :2] - image_center)

    joints = solver.solve_yolo_side(side, points)

    assert np.rad2deg(joints[0]) == pytest.approx(0.0, abs=0.5)


def test_elbow_flexion_comes_from_angle_between_limb_segments(solver):
    points = arm_points()
    points[2, :2] = points[1, :2] + [0.0, 150.0]
    points[3, :2] = points[2, :2] + [0.0, 50.0]

    joints = solver.solve_yolo_side("left", points)

    assert np.rad2deg(joints[3]) == pytest.approx(90.0, abs=0.5)


@pytest.mark.parametrize(
    "side,offset,expected_sign",
    [("left", 0, -1.0), ("right", 4, 1.0)],
)
def test_single_arm_foreshortening_still_creates_forward_motion(
    solver, side, offset, expected_sign
):
    points = arm_points()
    shoulder = points[offset, :2].copy()
    points[offset + 1:offset + 4, :2] = shoulder + 0.75 * (
        points[offset + 1:offset + 4, :2] - shoulder
    )

    joints = solver.solve_yolo_side(side, points)

    assert expected_sign * np.rad2deg(joints[0]) > 20.0


def test_joint_deadband_rejects_sub_degree_target_jitter(solver, monkeypatch):
    monkeypatch.setattr(
        "demo2.position_to_angle_v2.time.monotonic", lambda: 1.0
    )
    solver.latest["left"] = np.zeros(7, dtype=float)
    solver.last_update_time["left"] = 0.9
    target = np.zeros(7, dtype=float)
    target[1] = np.deg2rad(0.5)

    solver.limit_step("left", target)

    assert np.rad2deg(solver.latest["left"][1]) == pytest.approx(0.0)


def test_joint_smoothing_softens_large_target_step(solver, monkeypatch):
    monkeypatch.setattr(
        "demo2.position_to_angle_v2.time.monotonic", lambda: 1.0
    )
    solver.latest["left"] = np.zeros(7, dtype=float)
    solver.last_update_time["left"] = 0.9
    target = np.zeros(7, dtype=float)
    target[1] = np.deg2rad(20.0)

    solver.limit_step("left", target)

    output_deg = np.rad2deg(solver.latest["left"][1])
    assert 5.0 < output_deg < 15.0
