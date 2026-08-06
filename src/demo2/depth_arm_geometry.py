"""Depth-image sampling and pinhole deprojection for arm landmarks."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = ()
    distortion_model: str = "plumb_bob"

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

    @classmethod
    def from_camera_info(cls, message):
        matrix = list(message.k)
        if len(matrix) != 9:
            raise ValueError("CameraInfo.k must contain nine values")
        return cls(
            int(message.width),
            int(message.height),
            float(matrix[0]),
            float(matrix[4]),
            float(matrix[2]),
            float(matrix[5]),
            tuple(float(value) for value in message.d),
            str(message.distortion_model),
        )

    def deproject(self, pixel, depth_m):
        u, v = np.asarray(pixel, dtype=float)
        depth_m = float(depth_m)
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            raise ValueError("depth must be finite and positive")
        matrix = np.asarray([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])
        distortion = np.asarray(self.distortion, dtype=float)
        if distortion.size and np.any(np.abs(distortion) > 1e-12):
            source = np.asarray([[[u, v]]], dtype=float)
            if self.distortion_model == "equidistant":
                normalized = cv2.fisheye.undistortPoints(
                    source, matrix, distortion.reshape(4, 1)
                )
            else:
                normalized = cv2.undistortPoints(
                    source, matrix, distortion
                )
            x, y = normalized.reshape(2)
        else:
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
        return np.asarray([x * depth_m, y * depth_m, depth_m])


def depth_image_to_meters(image, encoding, uint16_scale=0.001):
    image = np.asarray(image)
    encoding = str(encoding).strip().lower()
    if encoding in ("16uc1", "mono16"):
        return image.astype(np.float32) * float(uint16_scale)
    if encoding == "32fc1":
        return image.astype(np.float32, copy=False)
    raise ValueError(
        f"unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1"
    )


def color_message_to_bgr(message):
    """Decode common sensor_msgs/Image color encodings without cv_bridge."""
    encoding = str(message.encoding).strip().lower()
    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(
            f"unsupported color encoding {encoding!r}; expected BGR/RGB 8-bit"
        )
    row_values = int(message.step)
    required = int(message.width) * channels
    if row_values < required:
        raise ValueError("color image step is smaller than its row width")
    values = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.height) * row_values
    if values.size < expected:
        raise ValueError("color image data is truncated")
    image = values[:expected].reshape(int(message.height), row_values)
    image = image[:, :required].reshape(
        int(message.height), int(message.width), channels
    )
    if encoding in ("rgb8", "rgba8"):
        image = image[..., [2, 1, 0, 3] if channels == 4 else [2, 1, 0]]
    if channels == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def depth_message_to_meters(message, uint16_scale=0.001):
    """Decode 16UC1/32FC1 sensor_msgs/Image including padded rows."""
    encoding = str(message.encoding).strip().lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    else:
        raise ValueError(
            f"unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1"
        )
    if int(message.step) % dtype.itemsize:
        raise ValueError("depth image step is not aligned to its data type")
    row_values = int(message.step) // dtype.itemsize
    if row_values < int(message.width):
        raise ValueError("depth image step is smaller than its row width")
    values = np.frombuffer(message.data, dtype=dtype)
    expected = int(message.height) * row_values
    if values.size < expected:
        raise ValueError("depth image data is truncated")
    image = values[:expected].reshape(int(message.height), row_values)
    image = image[:, :int(message.width)]
    return depth_image_to_meters(image, encoding, uint16_scale)


def robust_depth_at_pixel(
    depth_m,
    pixel,
    radius=4,
    min_depth_m=0.15,
    max_depth_m=5.0,
    min_valid_pixels=4,
    cluster_tolerance_m=0.08,
):
    """Estimate foreground depth around a landmark while rejecting holes.

    The center sample anchors the cluster when available. If the center is a
    depth hole, the lower quartile anchors the likely foreground rather than
    taking a background-dominated window median.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    if depth_m.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    u, v = np.round(np.asarray(pixel, dtype=float)).astype(int)
    height, width = depth_m.shape
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    radius = max(int(radius), 0)
    x0, x1 = max(u - radius, 0), min(u + radius + 1, width)
    y0, y1 = max(v - radius, 0), min(v + radius + 1, height)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[
        np.isfinite(patch)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    ]
    if valid.size < int(min_valid_pixels):
        return None

    center = float(depth_m[v, u])
    if np.isfinite(center) and min_depth_m <= center <= max_depth_m:
        anchor = center
    else:
        anchor = float(np.quantile(valid, 0.25))
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    tolerance = max(float(cluster_tolerance_m), 3.0 * mad)
    cluster = valid[np.abs(valid - anchor) <= tolerance]
    if cluster.size < int(min_valid_pixels):
        return None
    return float(np.median(cluster))


class DepthLandmarkReconstructor:
    def __init__(
        self,
        radius=4,
        min_depth_m=0.15,
        max_depth_m=5.0,
        min_valid_pixels=4,
        cluster_tolerance_m=0.08,
    ):
        self.radius = int(radius)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_valid_pixels = int(min_valid_pixels)
        self.cluster_tolerance_m = float(cluster_tolerance_m)

    @staticmethod
    def validate_inputs(pixels, depth_m, intrinsics):
        """Validate aligned landmark and depth dimensions."""
        pixels = np.asarray(pixels, dtype=float)
        if pixels.shape != (8, 2):
            raise ValueError("expected eight 2-D arm/hip landmarks")
        if depth_m.shape != (intrinsics.height, intrinsics.width):
            raise ValueError(
                "aligned depth dimensions do not match color CameraInfo"
            )
        return pixels

    def reconstruct_partial(self, pixels, depth_m, intrinsics):
        """Return per-landmark XYZ, using NaN where depth is unavailable."""
        pixels = self.validate_inputs(pixels, depth_m, intrinsics)
        points = []
        depths = []
        for pixel in pixels:
            depth = robust_depth_at_pixel(
                depth_m,
                pixel,
                self.radius,
                self.min_depth_m,
                self.max_depth_m,
                self.min_valid_pixels,
                self.cluster_tolerance_m,
            )
            if depth is None:
                points.append(np.full(3, np.nan, dtype=float))
                depths.append(np.nan)
                continue
            points.append(intrinsics.deproject(pixel, depth).astype(float))
            depths.append(depth)
        return np.asarray(points), np.asarray(depths)

    def reconstruct(self, pixels, depth_m, intrinsics):
        """Return all eight XYZ landmarks only when every depth is valid."""
        points, depths = self.reconstruct_partial(
            pixels, depth_m, intrinsics
        )
        if not np.all(np.isfinite(points)):
            return None, None
        return points, depths
