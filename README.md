# demo2v2

MediaPipe body pose to mirrored dual Nero arm control.

## Flow

1. `body_yolo`: camera or image topic to MediaPipe world landmarks on `/body_pose`.
2. `position_to_angle`: `/body_pose` to filtered `/left_arm` and `/right_arm`, then AGX `move_p` pose commands.
3. `agx_arm_ctrl`: AGX SDK ROS control topics for Nero arms.

The two arms are treated as mirrored side-mounted arms. The default launch uses separate left and right camera-to-robot matrices:

- left: `[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]`
- right: `[0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]`

## Build

```bash
colcon build --packages-select demo2v2
source install/setup.bash
```

## Sim First

The official Nero URDF was copied from `agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf`.
The dual-arm display launch runs two `robot_state_publisher` instances with `left/` and `right/` frame prefixes because the official URDF has fixed link names.

```bash
ros2 launch demo2v2 nero_dual_side_mount_display.launch.py
```

## Dry Run

This starts YOLO and the mapper, but does not start AGX control:

```bash
ros2 launch demo2v2 dual_nero_yolo_agx.launch.xml start_agx_ctrl:=false
```

Start AGX feedback while keeping physical control disabled:

```bash
ros2 launch demo2v2 dual_nero_yolo_agx.launch.xml start_agx_ctrl:=true agx_control_enabled:=false
```

Only after verifying topics, calibration, and target motion, enable AGX control:

```bash
ros2 launch demo2v2 dual_nero_yolo_agx.launch.xml start_agx_ctrl:=true agx_control_enabled:=true
```

The copied YOLO model is kept local by `.gitignore`. If the package is cloned elsewhere, copy `yolo26s-pose.pt` into `model/` or pass `model_path:=...`.
