# Intel RealSense depth recognition

The RealSense implementation is split so that camera acquisition and 3-D
recognition can be debugged or moved without any Nero robot code.

## Data flow

```text
realsense2_camera OR realsense_rgbd_viewer_cpp
  color/image_raw --------------------+
  aligned_depth_to_color/image_raw ---+--> depth_pose_detector_cpp
  color/camera_info ------------------+       |
                                               +--> /realsense/landmarks_3d
                                               +--> /realsense/arm_pose_debug

/realsense/landmarks_3d --> depth_arm_controller
                          |--> person camera/reference poses
                          +--> joint states / gated commands
```

The default detector is one C++ process. It runs the bundled 384-pixel
TorchScript YOLO pose model, samples a foreground depth cluster around every
landmark, and deprojects the points with the color-camera intrinsics. It keeps
only the newest pending RGB-D frame rather than building a latency-producing
queue. A color frame waits briefly for depth within the configured 20 ms
tolerance; the previous 30 Hz depth frame is rejected instead of being fused
with the wrong RGB image.

YOLO detections are associated over time by person box overlap and motion.
This keeps control locked to one person when another detection temporarily has
a higher score. A short miss keeps the last tracked box visible while the
controller safely holds its last joints; stale pixels are never presented as a
new control measurement. A keypoint EMA, foreground-biased depth clustering,
a short 3-D median plus EMA, and joint deadband/low-pass filtering reduce
detector, depth, and RViz jitter at their respective stages.

RealSense provides its factory-calibrated color/depth intrinsics through
`CameraInfo`. Runtime calibration therefore needs no board, tag, IMU, or
additional pose hardware: the controller averages the visible shoulder/hip
geometry while the person stands naturally and still for three seconds. Arm
position during this step is unrestricted; only both shoulders and both hips
must remain visible. Isolated detector/depth outliers are discarded rather
than restarting the sample window.

## Install

The project-local C++ driver, librealsense runtime, and exported TorchScript
model are bundled, so camera acquisition does not need `pyrealsense2`. The
build links the LibTorch included with the installed CPU PyTorch package:

```bash
python3 -m pip install --user torch \
  --index-url https://download.pytorch.org/whl/cpu
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
published on `/realsense/landmarks_3d`. The diagnostic topic
`/realsense/landmarks_3d` retains valid individual coordinates when another
landmark has missing depth; its `depth_valid` channel identifies each one.
The overlay also reports the actual RGB-D timestamp difference. Every three
seconds the detector log reports cumulative `sync`, `yolo`, `partial`, and
`complete` counts, making synchronization loss distinguishable from YOLO or
landmark loss.
On the current Ryzen 9 target, the 640 x 480 camera path measures 30 FPS, C++
inference measures about 22–24 FPS, and the complete 384-pixel recognition/
depth topic measures about 19–22 FPS depending on debug-image publishing. The
former Python combined detector measured about 9.5 FPS on the same target.

The robot entry point opens the local D435i directly with librealsense over
USB. Its default serial is `108322074190`; its private image topics are
`/usb_realsense/d435i/color/image_raw` and
`/usb_realsense/d435i/aligned_depth_to_color/image_raw`. The composed launch
also sets `ROS_LOCALHOST_ONLY=1`, so a camera advertised by another computer
cannot enter this pipeline.

The default profile is 640 x 480 at 30 FPS. When bandwidth is limited:

```bash
ros2 launch demo2 realsense_camera.launch.py \
  color_profile:=640x480x15 depth_profile:=640x480x15
```

After replacing the physical camera, select its USB serial number explicitly:

```bash
ros2 launch demo2 realsense_camera.launch.py serial_no:="'123456789'"
```

## Recognition-only debugging

With the camera driver already running:

```bash
ros2 launch demo2 depth_pose_detector.launch.py show_gui:=false
ros2 topic echo /realsense/landmarks_3d --once
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

## Portable output contract

`/realsense/landmarks_3d` is `sensor_msgs/msg/PointCloud` in the color optical
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
contains IDs 0–7 in the order above. `depth_valid` marks each usable metric
coordinate; an unavailable point contains NaNs so the other arm can continue.
An empty point cloud means that no person or synchronized RGB-D frame is
available. A consumer must also enforce a timeout so stale poses never drive
hardware.

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

Use this as the only tracking entry point. The complete controller and the
static model-only display share one process lock; accidentally starting both
cannot create competing mount TF or joint-state publishers.

On first use, stand naturally and still in view for three seconds. The
reference is saved as
`~/.ros/demo2/person_calibration_108322074190.json` and automatically loaded
on later runs. Its camera identity is validated before use; legacy records or
records from another camera trigger a fresh natural-standing calibration.
Re-run the software calibration whenever the camera is moved:

```bash
ros2 service call /depth_arm_controller/recalibrate \
  std_srvs/srv/Trigger {}
```

Then stand naturally and keep both shoulders and hips stable in view for three
seconds. The arms do not need any prescribed pose.

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

The two Nero bases are a mirrored pair. Before IK, the controller mirrors
only the anatomical left arm's torso-relative forward/backward component;
outward and vertical components keep their signs. Both arms therefore move
in the same sagittal world direction without swapping their independent
left/right topics.

The anatomical left arm is fixed to `/left/joint_states` and
`/left/neroarm/command_joints`; the anatomical right arm is fixed to the
corresponding `/right/...` outputs. Their confidence/depth checks, filters, IK
state, timeouts, and command gates are independent. Monitor
`/left/tracking_status` and `/right/tracking_status` to see which side is
tracking or holding its last valid pose.

## pyAgxArm hardware boundary

`dual_nero_pyagxarm.launch.py` starts one process per arm. Each process owns
exactly one SocketCAN interface and subscribes to only its matching anatomical
command topic. The defaults are:

| Side | CAN | Command | Measured feedback | Hardware status |
|---|---|---|---|---|
| left | `can0` | `/left/neroarm/command_joints` | `/left/neroarm/measured_joint_states` | `/left/neroarm/hardware_status` |
| right | `can1` | `/right/neroarm/command_joints` | `/right/neroarm/measured_joint_states` | `/right/neroarm/hardware_status` |

Starting the hardware nodes connects and verifies seven-joint feedback but
does not enable motors. Motion additionally requires
`hardware_execute_motion:=true`, `command_output_enabled:=true`, and the
side-specific `~/enable` service. This prevents a camera or launch restart
from enabling a real arm automatically.

Firmware defaults to `auto`: each side first queries `get_firmware()` through
an official `NeroFW.DEFAULT` probe and then reconnects with the matching Nero
driver. The hardware status includes the requested/resolved firmware,
controller `arm_status`, and all seven joint enable flags.

The hardware boundary independently rejects non-finite, malformed, and
out-of-URDF-limit commands. Enabling also requires NORMAL controller status
and seven positive motor-enable flags. It applies a second 30 deg/s joint-rate limit and
a 350 ms command watchdog. On command timeout it replaces the old trajectory
with the current measured pose. On feedback loss it closes that side's gate
and requests pyAgxArm's damped electronic stop. A process lock prevents a
second driver from owning the same CAN interface.

The controller publishes a nonzero bent-elbow display pose until it has a
valid IK solution. After tracking starts, brief invalid or missing frames
retain the last valid result; a side closes its hardware command gate only
after its independent timeout and never snaps RViz back to all zeros. Do not
run a second joint-state publisher on
`/left/joint_states` or `/right/joint_states` while diagnosing tracking.

## Move recognition to another package

The portable portion consists of:

- `src/realsense_rgbd_viewer.cpp`;
- `src/depth_pose_detector.cpp`;
- `third_party/librealsense2/`;
- `model/yolo26n-pose.torchscript`;
- `launch/depth_pose_detector.launch.py`;
- the relevant CMake and `package.xml` dependency entries.

Camera launch/migration also uses `launch/realsense_camera.launch.py` and
`launch/realsense_rgbd_view.launch.py`.

The default path uses standard ROS messages (`sensor_msgs`), librealsense2,
OpenCV C++, and LibTorch. The bundled librealsense and TorchScript artifacts
are built for the current x86-64 software stack; re-export/rebuild them for a
different target. The downstream project only needs to implement the
documented `PointCloud` contract; no Nero source files need to be copied.

## Main tuning parameters

- `sync_tolerance_sec`: maximum color/depth timestamp difference, default
  0.02 seconds;
- `sync_wait_sec`: bounded wait for the matching depth frame, default 0.02
  seconds;
- `depth_window_radius`: depth sampling radius around a landmark;
- `depth_cluster_tolerance_m`: foreground cluster width;
- `min_depth_m` / `max_depth_m`: accepted working range, default 0.15–5.0 m;
- `min_landmark_confidence`: pose keypoint confidence gate;
- `torso_hold_sec`: calibrated torso hold across a brief shoulder/hip dropout,
  default 0.25 seconds;
- `target_lock_max_missed_frames`: frames held before selecting a new person;
- `keypoint_smoothing_alpha`: detector-side 2-D keypoint smoothing;
- `point_smoothing_alpha`: controller-side 3-D smoothing;
- `point_median_window`: short 3-D outlier suppression window;
- `max_point_jump_m`: controller-side discontinuity rejection;
- `joint_smoothing_tau_sec`: joint low-pass time constant;
- `joint_deadband_deg`: joint changes smaller than this are held;
- `neutral_calibration_sec`: natural-standing calibration duration;
- `calibration_max_consecutive_outliers`: consecutive unstable torso frames
  required before deliberately starting a new sample cluster;
- `max_person_translation_m`: maximum displacement from the reference;
- `max_person_rotation_deg`: maximum rotation from the reference.

Most ROS depth streams use `16UC1` millimetres. Override
`depth_uint16_scale` if another driver uses a different scale.
