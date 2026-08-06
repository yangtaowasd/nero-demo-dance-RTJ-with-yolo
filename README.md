# demo2 RealSense RGB-D arm follower

This ROS 2 package detects a person's arms in Intel RealSense color images,
reconstructs the landmarks with aligned hardware depth, and mirrors the 3-D
directions on two Nero arms. YOLO pose is the default detector; MediaPipe is an
optional fallback.

## Split architecture

The depth-camera path is deliberately separated at ROS topic boundaries:

1. `realsense_camera.launch.py` starts only camera acquisition and
   depth-to-color alignment. It uses the official ROS driver when available,
   otherwise it automatically starts the package-local C++ librealsense
   bridge.
2. `depth_pose_detector.py` consumes aligned RGB-D images and publishes a
   robot-independent `sensor_msgs/PointCloud` on
   `/realsense/arm_pose_3d`, plus an annotated debug image on
   `/realsense/arm_pose_debug`.
3. `depth_arm_controller.py` consumes that point cloud and contains all
   Nero-specific filtering, calibration, IK, and command gating.

This makes camera bring-up, recognition, and arm control independently
testable. See [RealSense depth follower](docs/realsense-depth-follower.md) for
the topic contract, rosbag workflow, tuning, and migration file list.

## Install and build

For ROS 2 Humble, a sudo-free installation is supported:

```bash
python3 -m pip install --user 'numpy>=1.23,<1.25'
python3 -m pip install --user torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user ultralytics

cd ~/demo_ws
colcon build --packages-select demo2
source install/setup.bash
```

Installing `ros-humble-realsense2-camera` remains optional. If it is later
installed, the same camera launch automatically prefers that driver.

The package-local `model/yolo26n-pose.pt` is used by default for lower CPU
latency. Use `yolo26s-pose.pt` when higher accuracy is more important than
speed. To use the MediaPipe backend instead, install `mediapipe==0.10.14` and
pass `detector_backend:=mediapipe`.

The Nero URDF meshes are resolved from the renamed ROS package
`armbycontroller`, which is built from `agxarm_control_by_gamecontroller`.

## Run the complete RealSense pipeline

Connect the camera over USB 3 and keep real-arm commands disabled initially:

```bash
ros2 launch demo2 realsense_depth_arm.launch.py \
  command_output_enabled:=false
```

On the first run, enter the frame and stand naturally and still for three
seconds. No T-pose, calibration board, tag, IMU, or other pose hardware is
required, and the arms may stay naturally at any position. Keep both shoulders
and both hips visible; isolated YOLO/depth outliers are ignored instead of
restarting the calibration. The depth path rejects unsynchronized frames,
depth holes,
implausible limb lengths, large 3-D jumps, stale poses, and poor IK solutions.

The resulting camera-to-person reference is stored in
`~/.ros/demo2/realsense_person_calibration.json`. Current torso pose in the
camera optical frame is published on `/realsense/person_camera_pose`; motion
relative to the saved reference is published on
`/realsense/person_relative_pose`. To recalibrate after moving the camera:

```bash
ros2 service call /depth_arm_controller/recalibrate std_srvs/srv/Trigger {}
```

## Debug each layer independently

Terminal 1 — camera only:

```bash
ros2 launch demo2 realsense_camera.launch.py
```

Camera debugging with RGB, colorized aligned depth, and RGB-D fusion in one
window (no recognition or robot nodes):

```bash
ros2 launch demo2 realsense_rgbd_view.launch.py
```

The same three-panel image is published on `/realsense/rgbd_view`.
Camera capture, alignment, colorization, fusion, ROS publishing, and GUI are
implemented in one C++ process. At 640 x 480 the measured processing rate on
the target machine is approximately 30 FPS. The default displayed and
accepted depth range is 0.15–5.0 m.

Camera plus YOLO RGB-D recognition, without RViz or robot control:

```bash
ros2 launch demo2 realsense_yolo.launch.py
```

This opens an annotated view with the selected person box, arm skeleton,
landmark confidence, metric depth, and inference FPS. It uses the lightweight
`yolo26n-pose.pt` model by default. For higher accuracy at lower speed:

```bash
ros2 launch demo2 realsense_yolo.launch.py \
  model_path:="$(ros2 pkg prefix demo2)/share/demo2/model/yolo26s-pose.pt"
```

Read all per-landmark coordinates, including partial results marked by the
`depth_valid` channel:

```bash
ros2 topic echo /realsense/landmarks_3d
```

Terminal 2 — recognition only:

```bash
ros2 launch demo2 depth_pose_detector.launch.py show_gui:=false
ros2 topic echo /realsense/arm_pose_3d --once
```

View `/realsense/arm_pose_debug` with `rqt_image_view`, Foxglove, or RViz.

Terminal 3 — Nero visualization/control only:

```bash
ros2 launch demo2 depth_arm_control.launch.py \
  command_output_enabled:=false
```

For an aligned-depth camera other than RealSense, run its driver and use
`depth_camera_arm.launch.py`, overriding `color_topic`,
`aligned_depth_topic`, and `camera_info_topic`.

## Existing monocular pipeline

The original USB-camera paths remain available:

```bash
# MediaPipe monocular world landmarks
ros2 launch demo2 dual_nero_yolo_rviz.launch.py

# Legacy YOLO 2-D mapping
ros2 launch demo2 dual_nero_yolo_rviz.launch.py pose_backend:=yolo_2d
```

Only enable `command_output_enabled` after the independent RGB-D output and
RViz motion are stable. The arm drivers must be running separately on the
configured `/left/neroarm/command_joints` and
`/right/neroarm/command_joints` topics.
