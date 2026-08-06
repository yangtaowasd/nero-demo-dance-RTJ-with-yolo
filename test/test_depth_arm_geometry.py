import numpy as np
import pytest
from types import SimpleNamespace

from demo2.depth_arm_geometry import (
    DepthLandmarkReconstructor,
    PinholeIntrinsics,
    color_message_to_bgr,
    depth_message_to_meters,
    depth_image_to_meters,
    robust_depth_at_pixel,
)


def test_uint16_depth_is_converted_from_millimetres():
    image = np.asarray([[0, 1250], [2000, 3500]], dtype=np.uint16)

    output = depth_image_to_meters(image, "16UC1")

    np.testing.assert_allclose(output, [[0.0, 1.25], [2.0, 3.5]])


def test_float_depth_stays_in_metres():
    image = np.asarray([[1.2, 2.4]], dtype=np.float32)

    output = depth_image_to_meters(image, "32FC1")

    np.testing.assert_allclose(output, image)


def test_depth_hole_uses_foreground_cluster_instead_of_background():
    depth = np.full((15, 15), 2.8, dtype=float)
    depth[4:11, 4:11] = 1.2
    depth[7, 7] = 0.0

    output = robust_depth_at_pixel(depth, [7, 7], radius=4)

    assert output == pytest.approx(1.2)


def test_deprojection_recovers_expected_camera_point():
    intrinsics = PinholeIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)

    point = intrinsics.deproject([420.0, 190.0], 2.0)

    np.testing.assert_allclose(point, [0.4, -0.2, 2.0])


def test_reconstructor_rejects_misaligned_depth_dimensions():
    intrinsics = PinholeIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    reconstructor = DepthLandmarkReconstructor()

    with pytest.raises(ValueError, match="aligned depth dimensions"):
        reconstructor.reconstruct(
            np.zeros((8, 2)), np.ones((240, 320)), intrinsics
        )


def test_partial_reconstruction_keeps_coordinates_around_a_depth_hole():
    intrinsics = PinholeIntrinsics(20, 20, 10.0, 10.0, 10.0, 10.0)
    reconstructor = DepthLandmarkReconstructor(
        radius=0, min_valid_pixels=1
    )
    pixels = np.asarray([[column, 5.0] for column in range(2, 10)])
    depth = np.full((20, 20), 2.0, dtype=float)
    depth[5, 2] = 0.0

    points, depths = reconstructor.reconstruct_partial(
        pixels, depth, intrinsics
    )

    assert np.all(np.isnan(points[0]))
    assert np.isnan(depths[0])
    np.testing.assert_allclose(points[1:, 2], 2.0)
    complete, complete_depths = reconstructor.reconstruct(
        pixels, depth, intrinsics
    )
    assert complete is None
    assert complete_depths is None


def test_padded_rgb_message_is_decoded_to_bgr():
    message = SimpleNamespace(
        encoding="rgb8",
        width=2,
        height=1,
        step=8,
        data=bytes([255, 0, 0, 0, 255, 0, 99, 99]),
    )

    output = color_message_to_bgr(message)

    np.testing.assert_array_equal(output, [[[0, 0, 255], [0, 255, 0]]])


def test_padded_depth_message_is_decoded_to_metres():
    raw = np.asarray([[1000, 2500, 65535]], dtype="<u2")
    message = SimpleNamespace(
        encoding="16UC1",
        width=2,
        height=1,
        step=6,
        is_bigendian=False,
        data=raw.tobytes(),
    )

    output = depth_message_to_meters(message)

    np.testing.assert_allclose(output, [[1.0, 2.5]])
