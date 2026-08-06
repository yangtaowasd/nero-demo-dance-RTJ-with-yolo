# demo2

CPU-friendly monocular 3-D pose mirroring for dual Nero arm control.

This repository combines the original `demo2` YOLO/IK pipeline with the
newer MediaPipe 3-D retargeting and RViz pipeline. The original AGX `move_p`
implementation formerly named `position_to_angle_v2.py` is retained as
`position_to_angle_agx.py`.

## Flow

1. `arm_mediapipe3d`: USB camera frames to `/arm_pose_3d`.
2. `position_to_angle_v2`: body-relative 3-D arm directions to left/right
   Nero joint commands.
3. Optional `neroarm_control` real-arm drivers receive `/left/neroarm/command_joints`
   and `/right/neroarm/command_joints`.

MediaPipe is the default backend and runs without YOLO. The legacy YOLO 2-D
backend remains available as a fallback.

## Build

```bash
cd ~/demo_ws
python3 -m pip install "mediapipe==0.10.14"
colcon build --packages-select demo2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
```

`agx_arm_description` must be in the sourced ROS environment because the Nero
URDF meshes use `package://agx_arm_description/...` resource paths.

This package installs its own YOLO model, URDFs, and RViz configuration. See
[URDF origin](docs/urdf-origin.md) for the upstream source and local
modifications.

## RViz Dry Run

```bash
ros2 launch demo2 dual_nero_yolo_rviz.launch.py show_gui:=true
```

The default CPU profile is 640×480 with MediaPipe Pose Full. If it cannot keep
up in real time, select the Lite model:

```bash
ros2 launch demo2 dual_nero_yolo_rviz.launch.py \
  mediapipe_model_complexity:=0
```

The project-local YOLO model can still be selected explicitly:

```bash
ros2 launch demo2 dual_nero_yolo_rviz.launch.py \
  pose_backend:=yolo_2d
```

## Pose mapping

The default calibrated mapping uses MediaPipe world landmarks:

- J1: upper-arm forward direction in the torso coordinate frame.
- J2: upper-arm elevation in the torso coordinate frame.
- J3: neutral, because axial upper-arm rotation is not observable from a
  monocular skeleton.
- J4: three-dimensional elbow flexion.
- J5–J7: neutral by default.

Camera translation, image scale, and the person's distance from the camera are
removed by constructing the coordinate frame from shoulders and hips. On
startup, hold a neutral pose with both arms straight for two seconds.

`joint_smoothing_tau` controls response smoothing; larger values are steadier
but add lag. `joint_deadband_deg` suppresses small target jitter.

## Real Arm Dry Run

Start the real drivers without motion:

```bash
ros2 launch demo2 dual_nero_yolo_rviz.launch.py \
  start_pose:=false start_real_driver:=true command_output_enabled:=true execute_motion:=false \
  left_can_interface:=can0 right_can_interface:=can1
```

## Real Arm Motion

Only after RViz and dry-run look correct:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
source /home/yang/agx_arm_ws/install/setup.bash
source /home/yang/demo_ws/install/setup.bash

ros2 launch demo2 dual_nero_yolo_rviz.launch.py \
  show_gui:=true start_real_driver:=true command_output_enabled:=true \
  execute_motion:=true speed_percent:=30 max_command_delta:=0.25 \
  left_can_interface:=can0 right_can_interface:=can1
```
