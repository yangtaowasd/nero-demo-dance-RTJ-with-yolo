# demo2v2

MediaPipe body pose to mirrored dual Nero arm control.

## Flow

1. `arm_mediapipe`: camera or image topic to six arm world landmarks on `/arm_pose`.
2. `position_to_angle`: `/arm_pose` to filtered `/left_arm` and `/right_arm`, then AGX `move_p` pose commands.
3. `agx_arm_ctrl`: AGX SDK ROS control topics for Nero arms.

The two arms are treated as mirrored side-mounted arms. The default launch uses separate left and right camera-to-robot matrices:

- left: `[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]`
- right: `[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]`

## Build

```bash
colcon build --packages-select demo2v2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
```

`agx_arm_description` must be in the sourced ROS environment because the Nero
URDF meshes use `package://agx_arm_description/...` resource paths.

## Sim First

The official Nero URDF was copied from `agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf`.
The dual-arm display launch runs two `robot_state_publisher` instances with `left/` and `right/` frame prefixes because the official URDF has fixed link names.

```bash
ros2 launch demo2v2 nero_dual_side_mount_display.launch.py
```

For RViz validation with MediaPipe disabled in the OpenCV window:

```bash
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
ros2 launch demo2v2 dual_nero_mediapipe_rviz.launch.py show_gui:=false
```

By default, the MediaPipe launch files only accept a person whose body center is
inside the center of the image. The v2 launch also requires all eight arm image
points, shoulder/elbow/wrist/hand for both arms, to stay inside the ROI. Tune or
disable it with:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_rviz.launch.py \
  center_roi_enabled:=true center_roi_fraction:=0.67
```

For v2, the default ROI is wider/taller:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_v2_rviz.launch.py \
  center_roi_enabled:=true center_roi_fraction:=0.80
```

The v2 launch can use either MediaPipe or YOLO26s while keeping the same
`/arm_pose_v2` output for `position_to_angle_v2`:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_v2_rviz.launch.py pose_backend:=mediapipe

ros2 launch demo2v2 dual_nero_mediapipe_v2_rviz.launch.py \
  pose_backend:=yolo26s yolo_model_path:=/home/yang/demo_ws/src/demo2/model/yolo26s-pose.pt
```

Use `center_roi_enabled:=false` to accept the full image.

## Dry Run

This starts MediaPipe arm tracking and the mapper, but does not start AGX control:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_agx.launch.xml start_agx_ctrl:=false
```

Start AGX feedback while keeping physical control disabled:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_agx.launch.xml start_agx_ctrl:=true agx_control_enabled:=false
```

Only after verifying topics, calibration, and target motion, enable AGX control:

```bash
ros2 launch demo2v2 dual_nero_mediapipe_agx.launch.xml start_agx_ctrl:=true agx_control_enabled:=true
```
