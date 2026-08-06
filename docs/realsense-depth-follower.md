# Intel RealSense depth recognition

The RealSense implementation is split so that camera acquisition and 3-D
recognition can be debugged or moved without any Nero robot code.

## Data flow

```text
realsense2_camera OR realsense_rgbd_viewer_cpp
  color/image_raw --------------------+
  aligned_depth_to_color/image_raw ---+--> depth_pose_detector
  color/camera_info ------------------+       |
                                               +--> /realsense/arm_pose_3d
                                               +--> /realsense/arm_pose_debug

/realsense/arm_pose_3d --> depth_arm_controller
                          |--> person camera/reference poses
                          +--> joint states / gated commands
```

The detector first finds 2-D landmarks with the bundled YOLO pose model. It
then samples a foreground depth cluster around every pixel and deprojects it
with the color-camera intrinsics. MediaPipe can be selected as an optional 2-D
detector without changing the output interface.

RealSense provides its factory-calibrated color/depth intrinsics through
`CameraInfo`. Runtime calibration therefore needs no board, tag, IMU, or
additional pose hardware: the controller averages the visible shoulder/hip
geometry while the person stands naturally and still for three seconds. Arm
position during this step is unrestricted; only both shoulders and both hips
must remain visible. Isolated detector/depth outliers are discarded rather
than restarting the sample window.

## Install

The project-local C++ driver and its librealsense runtime are bundled, so
camera acquisition does not need sudo or `pyrealsense2`. YOLO still needs:

```bash
python3 -m pip install --user 'numpy>=1.23,<1.25'
python3 -m pip install --user torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user ultralytics
```

The official ROS driver is optional. When this command succeeds, the launch
uses it; otherwise the launch automatically uses
`realsense_rgbd_viewer_cpp`:

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix realsense2_camera
```

## Camera-only debugging

Connect the RealSense over USB 3 and start only camera acquisition:

```bash
ros2 launch demo2 realsense_camera.launch.py
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

To see RGB, colorized aligned depth, and their fusion side by side without
starting recognition or robot control:

```bash
ros2 launch demo2 realsense_rgbd_view.launch.py
```

The viewer publishes its composite image on `/realsense/rgbd_view`. Camera
capture, alignment, visualization, and publishing share one C++ process to
avoid Python/DDS image-copy bottlenecks.

To start the C++ camera and YOLO RGB-D recognition without any robot nodes:

```bash
ros2 launch demo2 realsense_yolo.launch.py
```

The YOLO window draws the selected person, arm skeleton, confidence and depth
for all eight landmarks, plus live inference FPS. The annotated ROS image is
published on `/realsense/arm_pose_debug`, while the eight 3-D points are
published on `/realsense/arm_pose_3d`. The diagnostic topic
`/realsense/landmarks_3d` retains valid individual coordinates when another
landmark has missing depth; its `depth_valid` channel identifies each one.

The default profile is 640 x 480 at 30 FPS. When bandwidth is limited:

```bash
ros2 launch demo2 realsense_camera.launch.py \
  color_profile:=640x480x15 depth_profile:=640x480x15
```

For multiple cameras, select one by serial number:

```bash
ros2 launch demo2 realsense_camera.launch.py serial_no:="'123456789'"
```

## Recognition-only debugging

With the camera driver already running:

```bash
ros2 launch demo2 depth_pose_detector.launch.py show_gui:=false
ros2 topic echo /realsense/arm_pose_3d --once
```

Open `/realsense/arm_pose_debug` in `rqt_image_view` to inspect detections and
the measured depth beside each landmark. The recognition node has no URDF, IK,
Nero driver, or CAN dependency.

To replay the exact camera input offline:

```bash
ros2 bag record \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info

ros2 bag play <bag_directory> --clock
ros2 launch demo2 depth_pose_detector.launch.py show_gui:=false
```

Use MediaPipe instead of YOLO when required:

```bash
python3 -m pip install "mediapipe==0.10.14"
ros2 launch demo2 depth_pose_detector.launch.py \
  detector_backend:=mediapipe model_complexity:=0
```

## Portable output contract

`/realsense/arm_pose_3d` is `sensor_msgs/msg/PointCloud` in the color optical
frame. A valid message always contains eight points in this order:

1. left shoulder;
2. left elbow;
3. left wrist;
4. left hip;
5. right shoulder;
6. right elbow;
7. right wrist;
8. right hip.

Coordinates are `(X, Y, Z)` metres in the color optical frame: `X` points
right, `Y` points down, and `Z` points forward from the camera. The
`confidence` channel contains eight detector scores and `landmark_id`
contains IDs 0–7 in the order above. An empty point cloud means that the
current frame is invalid; a consumer must also enforce a timeout so stale
poses never drive hardware.

## Controller-only and complete runs

Test the robot side against a detector or a recorded/published point cloud:

```bash
ros2 launch demo2 depth_arm_control.launch.py \
  command_output_enabled:=false
```

Start everything in one command:

```bash
ros2 launch demo2 realsense_depth_arm.launch.py \
  command_output_enabled:=false
```

On first use, stand naturally and still in view for three seconds. The
reference is saved as
`~/.ros/demo2/realsense_person_calibration.json` and automatically loaded on
later runs. Re-run the software calibration whenever the camera is moved:

```bash
ros2 service call /depth_arm_controller/recalibrate \
  std_srvs/srv/Trigger {}
```

Use these diagnostics to inspect the result:

```bash
ros2 topic echo /realsense/calibration_status
ros2 topic echo /realsense/person_camera_pose
ros2 topic echo /realsense/person_relative_pose
```

`person_camera_pose` is the inferred torso center and orientation in the color
optical frame. Its local axes are `X=person right`, `Y=person down`, and
`Z=person facing direction`. `person_relative_pose` is displacement and
rotation from the saved standing reference. Shoulder–elbow–wrist directions
are evaluated in the person's current torso frame, so turning relative to the
camera is taken into account before Nero IK. Relative translation and
excessive rotation are also used as safety gates. Keep hardware command output
disabled until the debug image and RViz response are stable.

## Move recognition to another package

The portable portion consists of:

- `src/realsense_rgbd_viewer.cpp`;
- `third_party/librealsense2/`;
- `src/demo2/depth_pose_detector.py`;
- `src/demo2/depth_arm_geometry.py`;
- `launch/realsense_camera.launch.py`;
- `launch/realsense_rgbd_view.launch.py`;
- `launch/depth_pose_detector.launch.py`;
- the relevant CMake and `package.xml` dependency entries;
- the chosen pose model under `model/`.

It uses only standard ROS messages (`sensor_msgs`, `geometry_msgs`),
librealsense2, OpenCV C++, NumPy, `rclpy`, and the selected YOLO/MediaPipe
runtime. The bundled librealsense runtime is x86-64; use the target platform's
librealsense package when moving to ARM. Rename the Python package imports when
moving the detector. The downstream project only needs to implement the
documented `PointCloud` contract; no Nero source files need to be copied.

## Main tuning parameters

- `sync_tolerance_sec`: maximum color/depth timestamp difference;
- `depth_window_radius`: depth sampling radius around a landmark;
- `depth_cluster_tolerance_m`: foreground cluster width;
- `min_depth_m` / `max_depth_m`: accepted working range, default 0.15–5.0 m;
- `min_landmark_confidence`: pose keypoint confidence gate;
- `point_smoothing_alpha`: controller-side 3-D smoothing;
- `max_point_jump_m`: controller-side discontinuity rejection.
- `neutral_calibration_sec`: natural-standing calibration duration;
- `calibration_max_consecutive_outliers`: consecutive unstable torso frames
  required before deliberately starting a new sample cluster;
- `max_person_translation_m`: maximum displacement from the reference;
- `max_person_rotation_deg`: maximum rotation from the reference.

Most ROS depth streams use `16UC1` millimetres. Override
`depth_uint16_scale` if another driver uses a different scale.
