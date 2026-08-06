import numpy as np
import pytest

from demo2.pose_retargeting import arm_features, retarget_arm


def neutral_pose():
    return np.asarray([
        [-0.20, -0.50, 0.00],
        [-0.50, -0.50, 0.00],
        [-0.80, -0.50, 0.00],
        [-0.15, 0.00, 0.00],
        [0.20, -0.50, 0.00],
        [0.50, -0.50, 0.00],
        [0.80, -0.50, 0.00],
        [0.15, 0.00, 0.00],
    ])


@pytest.mark.parametrize("side", ["left", "right"])
def test_features_are_invariant_to_camera_translation_and_scale(side):
    baseline = arm_features(neutral_pose(), side)
    transformed = 2.5 * neutral_pose() + np.asarray([3.0, -4.0, 7.0])

    current = arm_features(transformed, side)

    assert current.forward == pytest.approx(baseline.forward)
    assert current.elevation == pytest.approx(baseline.elevation)
    assert current.elbow_flex == pytest.approx(baseline.elbow_flex)


@pytest.mark.parametrize(
    "side,elbow_index,wrist_index,expected_sign",
    [("left", 1, 2, -1.0), ("right", 5, 6, 1.0)],
)
def test_arm_moving_forward_drives_only_shoulder_forward_joint(
    side, elbow_index, wrist_index, expected_sign
):
    points = neutral_pose()
    shoulder_index = 0 if side == "left" else 4
    shoulder = points[shoulder_index]
    points[elbow_index] = shoulder + [0.0, 0.0, -0.30]
    points[wrist_index] = shoulder + [0.0, 0.0, -0.60]

    joints = retarget_arm(
        side,
        arm_features(points, side),
        arm_features(neutral_pose(), side),
    )

    assert expected_sign * joints[0] == pytest.approx(np.pi / 2.0)
    assert joints[1] == pytest.approx(0.0)
    assert joints[3] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "side,elbow_index,wrist_index,expected_sign",
    [("left", 1, 2, 1.0), ("right", 5, 6, -1.0)],
)
def test_arm_raising_drives_only_shoulder_elevation_joint(
    side, elbow_index, wrist_index, expected_sign
):
    points = neutral_pose()
    shoulder_index = 0 if side == "left" else 4
    shoulder = points[shoulder_index]
    points[elbow_index] = shoulder + [0.0, -0.30, 0.0]
    points[wrist_index] = shoulder + [0.0, -0.60, 0.0]

    joints = retarget_arm(
        side,
        arm_features(points, side),
        arm_features(neutral_pose(), side),
    )

    assert expected_sign * joints[1] == pytest.approx(np.pi / 2.0)
    assert joints[0] == pytest.approx(0.0)
    assert joints[3] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "side,elbow_index,wrist_index",
    [("left", 1, 2), ("right", 5, 6)],
)
def test_bending_elbow_drives_only_elbow_flexion_joint(
    side, elbow_index, wrist_index
):
    points = neutral_pose()
    elbow = points[elbow_index]
    points[wrist_index] = elbow + [0.0, 0.30, 0.0]

    joints = retarget_arm(
        side,
        arm_features(points, side),
        arm_features(neutral_pose(), side),
    )

    assert joints[0] == pytest.approx(0.0)
    assert joints[1] == pytest.approx(0.0)
    assert joints[3] == pytest.approx(np.pi / 2.0)
