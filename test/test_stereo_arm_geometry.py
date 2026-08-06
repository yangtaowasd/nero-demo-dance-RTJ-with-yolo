import numpy as np
import pytest

from demo2.stereo_arm_geometry import CameraModel, StereoRig


def make_vertical_rig():
    matrix = np.asarray([
        [520.0, 0.0, 320.0],
        [0.0, 520.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    lower = CameraModel("pinhole", matrix, np.zeros(5))
    upper = CameraModel("fisheye", matrix, np.zeros(4))
    # Upper camera center is y=-0.5 in the lower camera frame.
    return StereoRig(lower, upper, np.eye(3), [0.0, 0.5, 0.0])


@pytest.mark.parametrize(
    "point",
    ([0.1, 0.0, 2.0], [-0.4, 0.3, 2.8], [0.25, -0.4, 1.5]),
)
def test_vertical_stereo_recovers_3d_point(point):
    rig = make_vertical_rig()
    point = np.asarray(point, dtype=float)
    lower_pixel = rig.lower.project(point)
    upper_pixel = rig.upper.project(rig.rotation @ point + rig.translation)

    result = rig.triangulate(lower_pixel, upper_pixel)

    assert result.point == pytest.approx(point, abs=1e-6)
    assert result.ray_gap_m == pytest.approx(0.0, abs=1e-8)
    assert result.reprojection_error_px == pytest.approx(0.0, abs=1e-6)


def test_bad_correspondence_is_rejected_by_ray_gap_or_reprojection():
    rig = make_vertical_rig()
    points = np.asarray([
        [-0.2, -0.3, 2.0], [-0.4, -0.1, 2.0], [-0.6, 0.1, 2.0],
        [-0.15, 0.4, 2.0], [0.2, -0.3, 2.0], [0.4, -0.1, 2.0],
        [0.6, 0.1, 2.0], [0.15, 0.4, 2.0],
    ])
    lower = np.asarray([rig.lower.project(point) for point in points])
    upper = np.asarray([
        rig.upper.project(rig.rotation @ point + rig.translation)
        for point in points
    ])
    upper[2] += [80.0, -40.0]

    output, quality = rig.triangulate_landmarks(
        lower, upper, max_ray_gap_m=0.02, max_reprojection_error_px=3.0
    )

    assert output is None
    assert quality is None
