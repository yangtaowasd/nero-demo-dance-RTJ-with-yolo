# demo2 RealSense RGB-D arm follower

This ROS 2 package detects a person's arms in Intel RealSense color images,
reconstructs the landmarks with aligned hardware depth, and mirrors the 3-D
directions on two Nero arms. The real-time detector is C++ YOLO/TorchScript.

## Split architecture

The depth-camera path is deliberately separated at ROS topic boundaries:

1. `realsense_camera.launch.py` starts only camera acquisition and
   depth-to-color alignment. It uses the official ROS driver when available,
   otherwise it automatically starts the package-local C++ librealsense
   bridge.
2. `depth_pose_detector_cpp` consumes aligned RGB-D images, runs TorchScript
   YOLO, performs depth reconstruction, and publishes a robot-independent
   `sensor_msgs/PointCloud` on
   `/realsense/landmarks_3d`, plus an annotated debug image on
   `/realsense/arm_pose_debug`.
3. `depth_arm_controller.py` consumes that point cloud and contains all
   Nero-specific filtering, calibration, IK, and command gating.

This makes camera bring-up, recognition, and arm control independently
testable. See [RealSense depth follower](docs/realsense-depth-follower.md) for
the topic contract, rosbag workflow, tuning, and migration file list.

## Install and build

For ROS 2 Humble, a sudo-free installation is supported:

```bash
python3 -m pip install --user torch \
  --index-url https://download.pytorch.org/whl/cpu

cd ~/demo_ws
colcon build --packages-select demo2
source install/setup.bash
```

Installing `ros-humble-realsense2-camera` remains optional. If it is later
installed, the same camera launch automatically prefers that driver.

The default real-time path uses the package-local 384-pixel
`model/yolo26n-pose.torchscript` from C++. Camera capture, YOLO inference,
target association, depth sampling, 3-D reconstruction, annotation, and ROS
publishing therefore run without Python image copies. The Python detector is
not part of the installed pipeline.

The Nero URDF meshes are resolved from the renamed ROS package
`armbycontroller`, which is built from `agxarm_control_by_gamecontroller`.

## Run the complete RealSense pipeline

Connect the camera over USB 3 and keep real-arm commands disabled initially:

```bash
export ROS_LOCALHOST_ONLY=1
ros2 launch demo2 realsense_depth_arm.launch.py \
  command_output_enabled:=false
```

This entry point opens the local D435i directly through librealsense and USB.
It is pinned to serial `108322074190`, publishes camera data under
`/usb_realsense/d435i`, and sets `ROS_LOCALHOST_ONLY=1` for the complete
pipeline. It does not subscribe to `/camera/camera/...` or consume camera
images advertised by another computer over DDS.

This is the recommended tracking entry point; do not also start the static
dual-arm display launch. Both entry points share a singleton guard, so an
accidental second pipeline exits before publishing duplicate TF or joint-state
data.

On the first run, enter the frame and stand naturally and still for three
seconds. No T-pose, calibration board, tag, IMU, or other pose hardware is
required, and the arms may stay naturally at any position. Keep both shoulders
and both hips visible; isolated YOLO/depth outliers are ignored instead of
restarting the calibration. The depth path rejects unsynchronized frames,
depth holes, implausible limb lengths, large 3-D jumps, stale poses, and poor
IK solutions. It waits briefly for the matching aligned-depth timestamp rather
than pairing a color image with the previous 30 Hz depth frame.
Anatomical left landmarks update only `/left/joint_states` and
`/left/neroarm/command_joints`; anatomical right landmarks update only the
matching `/right/...` topics. Each side has an independent validity check,
filter, IK state, timeout, and command gate, so a dropout on one arm does not
freeze the other. Brief dropouts hold the last valid pose for up to
`pose_timeout_sec` (0.35 seconds by default). After calibration, a missing
shoulder/hip depth sample uses the last valid torso for at most 0.25 seconds,
so one short torso hole does not interrupt both arms.

The resulting camera-to-person reference is stored in
`~/.ros/demo2/person_calibration_108322074190.json`. The record is bound to
the local USB camera and cannot be reused by another camera. Current torso pose in the
camera optical frame is published on `/realsense/person_camera_pose`; motion
relative to the saved reference is published on
`/realsense/person_relative_pose`. To recalibrate after moving the camera:

```bash
ros2 service call /depth_arm_controller/recalibrate std_srvs/srv/Trigger {}
```

After the service succeeds, stand naturally and keep both shoulders and hips
stable in view for three seconds. Arm position is unrestricted.

Inspect each side independently:

```bash
ros2 topic echo /left/tracking_status
ros2 topic echo /right/tracking_status
ros2 topic echo /left/joint_states
ros2 topic echo /right/joint_states
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
accepted depth range is 0.15–8.0 m.

Camera plus YOLO RGB-D recognition, without RViz or robot control:

```bash
ros2 launch demo2 realsense_yolo.launch.py
```

This opens an annotated view with the selected person box, arm skeleton,
landmark confidence, metric depth, and inference FPS. It uses the lightweight
384-pixel `yolo26n-pose.torchscript` model by default. The detector locks onto
one person across confidence changes and short occlusions instead of changing
to whichever person scores highest in each frame.

On the current Ryzen 9 target, the C++ camera stays at 30 FPS, C++ inference
measures about 22–24 FPS, and the complete pose/depth output measures about
19–22 FPS depending on debug-image publishing. The previous Python RGB-D
detector measured about 9.5 FPS. The worker always replaces an old pending
frame with the newest camera frame, so load reduces frame rate rather than
creating an increasing delay.

Read all per-landmark coordinates, including partial results marked by the
`depth_valid` channel:

```bash
ros2 topic echo /realsense/landmarks_3d
```

Terminal 2 — recognition only:

```bash
ros2 launch demo2 depth_pose_detector.launch.py show_gui:=false
ros2 topic echo /realsense/landmarks_3d --once
```

View `/realsense/arm_pose_debug` with `rqt_image_view`, Foxglove, or RViz.

Terminal 3 — Nero visualization/control only:

```bash
ros2 launch demo2 depth_arm_control.launch.py \
  command_output_enabled:=false
```

Before the first valid IK result, RViz uses a visible bent-elbow pose instead
of an all-zero pose. A short recognition/depth dropout holds the last valid
joint result. Pixel landmarks, 3-D points, and joint angles are filtered at
separate stages to reduce visible jitter.

For a static model-only check, use:

```bash
ros2 launch demo2 nero_dual_side_mount_display.launch.py
```

Its fallback joint publisher automatically stops publishing when the tracking
controller owns the joint-state topics, preventing RViz from alternating
between the static and tracked poses. Normally run either this static launch
or the complete RealSense launch, not both.

For an aligned-depth camera other than RealSense, run its driver and use
`depth_camera_arm.launch.py`, overriding `color_topic`,
`aligned_depth_topic`, and `camera_info_topic`.

Only enable `command_output_enabled` after the independent RGB-D output and
RViz motion are stable. The included pyAgxArm bridge consumes the configured
`/left/neroarm/command_joints` and `/right/neroarm/command_joints` topics.

## Dual Nero hardware through pyAgxArm

The package includes two isolated SocketCAN drivers. By default anatomical
left owns `can0`, anatomical right owns `can1`, and both drivers are read-only.
They publish measured feedback on
`/left/neroarm/measured_joint_states` and
`/right/neroarm/measured_joint_states`.

Activate both CAN adapters at the pyAgxArm/Nero bitrate:

```bash
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 up
```

Connect both arms without enabling their motors or sending movement:

```bash
ros2 launch demo2 dual_nero_pyagxarm.launch.py
```

This dual-arm installation defaults to the verified v1.11 driver on both
sides. Nero v1.11 starts continuous CAN feedback only after the explicit
enable sequence, so a powered but disabled arm reports
`connected_waiting_for_enable`. Set either firmware argument to `auto` only
when that controller returns `software_version` while disabled.

For the complete camera pipeline, add `start_hardware:=true`. Real movement
requires both command gates and an explicit enable service call:

```bash
export ROS_LOCALHOST_ONLY=1
ros2 launch demo2 realsense_depth_arm.launch.py \
  start_hardware:=true \
  command_output_enabled:=true \
  hardware_execute_motion:=true

ros2 service call /left/nero_pyagxarm_driver/enable \
  std_srvs/srv/SetBool '{data: true}'
ros2 service call /right/nero_pyagxarm_driver/enable \
  std_srvs/srv/SetBool '{data: true}'
```

Each enable request is rejected until that arm is connected and a fresh,
validated vision command is available. It also requires a NORMAL arm status
and positive enable feedback from all seven motors. A 350 ms command timeout holds the
measured pose; lost joint feedback closes the command gate and triggers the
pyAgxArm electronic stop. Emergency-stop services are:

```bash
ros2 service call /left/nero_pyagxarm_driver/estop std_srvs/srv/Trigger '{}'
ros2 service call /right/nero_pyagxarm_driver/estop std_srvs/srv/Trigger '{}'
```

If the physical CAN cables are reversed, swap `left_can_interface` and
`right_can_interface` in the launch command instead of swapping ROS topics.
