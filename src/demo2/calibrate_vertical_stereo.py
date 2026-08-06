#!/usr/bin/env python3
"""Calibrate a lower pinhole camera and an upper fisheye camera.

Keep both cameras rigidly fixed. Show the same chessboard to both cameras at
many positions, depths and tilts, then press SPACE to accept each good pair.
The board dimensions are the number of *inner* corners.

Example::

    python3 src/demo2/calibrate_vertical_stereo.py \
      --lower-id 0 --upper-id 2 --columns 9 --rows 6 \
      --square-size-m 0.025 --output config/my_vertical_stereo.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower-id", type=int, default=0)
    parser.add_argument("--upper-id", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def open_camera(camera_id, args):
    camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open camera {camera_id}")
    return camera


def board_points(columns, rows, square_size):
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * square_size


def find_corners(frame, pattern):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags,
    )
    if not found:
        return None
    return cv2.cornerSubPix(
        gray,
        corners,
        (7, 7),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
    ).astype(np.float64)


def collect_pairs(args):
    lower = open_camera(args.lower_id, args)
    upper = open_camera(args.upper_id, args)
    pattern = (args.columns, args.rows)
    lower_samples = []
    upper_samples = []
    print("SPACE: accept visible pair; U: undo; Q/ESC: finish (minimum 12 pairs)")
    try:
        while len(lower_samples) < args.samples:
            lower_ok = lower.grab()
            upper_ok = upper.grab()
            if not lower_ok or not upper_ok:
                continue
            lower_ok, lower_frame = lower.retrieve()
            upper_ok, upper_frame = upper.retrieve()
            if not lower_ok or not upper_ok:
                continue
            lower_corners = find_corners(lower_frame, pattern)
            upper_corners = find_corners(upper_frame, pattern)
            if lower_corners is not None:
                cv2.drawChessboardCorners(lower_frame, pattern, lower_corners, True)
            if upper_corners is not None:
                cv2.drawChessboardCorners(upper_frame, pattern, upper_corners, True)
            ready = lower_corners is not None and upper_corners is not None
            text = f"pairs {len(lower_samples)}/{args.samples} {'READY' if ready else 'find board'}"
            color = (70, 230, 100) if ready else (0, 180, 255)
            for frame in (lower_frame, upper_frame):
                cv2.putText(
                    frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA,
                )
            cv2.imshow("Calibration lower laptop", lower_frame)
            cv2.imshow("Calibration upper fisheye", upper_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and ready:
                lower_samples.append(lower_corners.copy())
                upper_samples.append(upper_corners.copy())
                print(f"accepted pair {len(lower_samples)}")
            elif key in (ord("u"), ord("U")) and lower_samples:
                lower_samples.pop()
                upper_samples.pop()
                print("removed last pair")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        lower.release()
        upper.release()
        cv2.destroyAllWindows()
    if len(lower_samples) < 12:
        raise RuntimeError("at least 12 diverse stereo pairs are required")
    return lower_samples, upper_samples


def calibrate_intrinsics(object_points, lower_points, upper_points, image_size):
    pinhole_objects = [points.astype(np.float32) for points in object_points]
    pinhole_images = [points.astype(np.float32) for points in lower_points]
    lower_rms, lower_matrix, lower_distortion, _, _ = cv2.calibrateCamera(
        pinhole_objects, pinhole_images, image_size, None, None
    )

    fisheye_objects = [points.reshape(-1, 1, 3) for points in object_points]
    fisheye_images = [points.reshape(-1, 1, 2) for points in upper_points]
    upper_matrix = np.asarray([
        [image_size[0] * 0.5, 0.0, image_size[0] * 0.5],
        [0.0, image_size[0] * 0.5, image_size[1] * 0.5],
        [0.0, 0.0, 1.0],
    ])
    upper_distortion = np.zeros((4, 1))
    fisheye_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    fisheye_flags |= cv2.fisheye.CALIB_FIX_SKEW
    upper_rms, upper_matrix, upper_distortion, _, _ = cv2.fisheye.calibrate(
        fisheye_objects,
        fisheye_images,
        image_size,
        upper_matrix,
        upper_distortion,
        flags=fisheye_flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )
    return (
        lower_rms,
        lower_matrix,
        lower_distortion.reshape(-1),
        upper_rms,
        upper_matrix,
        upper_distortion.reshape(-1),
    )


def board_pose_pinhole(object_points, image_points, matrix, distortion):
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points, image_points, matrix, distortion
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    return cv2.Rodrigues(rotation_vector)[0], translation.reshape(3)


def board_pose_fisheye(object_points, image_points, matrix, distortion):
    normalized = cv2.fisheye.undistortPoints(
        image_points.reshape(-1, 1, 2), matrix, distortion.reshape(4, 1)
    )
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        normalized,
        np.eye(3),
        np.zeros(5),
    )
    if not ok:
        raise RuntimeError("fisheye solvePnP failed")
    return cv2.Rodrigues(rotation_vector)[0], translation.reshape(3)


def estimate_extrinsics(
    object_points,
    lower_points,
    upper_points,
    lower_matrix,
    lower_distortion,
    upper_matrix,
    upper_distortion,
):
    rotations = []
    translations = []
    for objects, lower, upper in zip(object_points, lower_points, upper_points):
        lower_rotation, lower_translation = board_pose_pinhole(
            objects, lower, lower_matrix, lower_distortion
        )
        upper_rotation, upper_translation = board_pose_fisheye(
            objects, upper, upper_matrix, upper_distortion
        )
        rotation = upper_rotation @ lower_rotation.T
        translation = upper_translation - rotation @ lower_translation
        rotations.append(rotation)
        translations.append(translation)

    rotation_mean = Rotation.from_matrix(rotations).mean().as_matrix()
    translation_median = np.median(np.asarray(translations), axis=0)
    rotation_errors = np.rad2deg(
        (Rotation.from_matrix(rotation_mean).inv() * Rotation.from_matrix(rotations)).magnitude()
    )
    translation_errors = np.linalg.norm(
        np.asarray(translations) - translation_median, axis=1
    )
    keep = (rotation_errors < max(np.median(rotation_errors) * 3.0, 1.0)) & (
        translation_errors < max(np.median(translation_errors) * 3.0, 0.01)
    )
    if int(np.sum(keep)) >= 6:
        rotation_mean = Rotation.from_matrix(np.asarray(rotations)[keep]).mean().as_matrix()
        translation_median = np.median(np.asarray(translations)[keep], axis=0)
    return rotation_mean, translation_median, rotation_errors, translation_errors


def write_yaml(args, calibration):
    (
        lower_rms,
        lower_matrix,
        lower_distortion,
        upper_rms,
        upper_matrix,
        upper_distortion,
        rotation,
        translation,
        rotation_errors,
        translation_errors,
    ) = calibration
    output = {
        "image_width": args.width,
        "image_height": args.height,
        "lower_camera": {
            "model": "pinhole",
            "camera_matrix": lower_matrix.tolist(),
            "distortion": lower_distortion.tolist(),
        },
        "upper_camera": {
            "model": "fisheye",
            "camera_matrix": upper_matrix.tolist(),
            "distortion": upper_distortion.tolist(),
        },
        "upper_from_lower": {
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
        },
        "calibration_quality": {
            "lower_rms_px": float(lower_rms),
            "upper_rms_px": float(upper_rms),
            "median_extrinsic_rotation_spread_deg": float(np.median(rotation_errors)),
            "median_extrinsic_translation_spread_m": float(np.median(translation_errors)),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    print(f"wrote {path}")
    print(yaml.safe_dump(output["calibration_quality"], sort_keys=False))


def main():
    args = parse_args()
    lower_points, upper_points = collect_pairs(args)
    objects = [
        board_points(args.columns, args.rows, args.square_size_m)
        for _ in lower_points
    ]
    intrinsics = calibrate_intrinsics(
        objects, lower_points, upper_points, (args.width, args.height)
    )
    lower_rms, lower_matrix, lower_distortion = intrinsics[:3]
    upper_rms, upper_matrix, upper_distortion = intrinsics[3:]
    extrinsics = estimate_extrinsics(
        objects,
        lower_points,
        upper_points,
        lower_matrix,
        lower_distortion,
        upper_matrix,
        upper_distortion,
    )
    write_yaml(
        args,
        (
            lower_rms,
            lower_matrix,
            lower_distortion,
            upper_rms,
            upper_matrix,
            upper_distortion,
            *extrinsics,
        ),
    )


if __name__ == "__main__":
    main()
