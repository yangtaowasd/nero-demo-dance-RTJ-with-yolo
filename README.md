# demo2v2

YOLO26s pose to mirrored dual Nero arm control.

## Flow

1. `arm_yolo26s_v2`: camera frames to `/arm_pose_v2` and `/arm_pose_v2/image_points`.
2. `position_to_angle_v2`: YOLO image points to left/right Nero joint commands.
3. Optional `neroarm_control` real-arm drivers receive `/left/neroarm/command_joints`
   and `/right/neroarm/command_joints`.

## Build

```bash
cd ~/demo_ws
colcon build --packages-select demo2v2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
```

`agx_arm_description` must be in the sourced ROS environment because the Nero
URDF meshes use `package://agx_arm_description/...` resource paths.

## RViz Dry Run

```bash
ros2 launch demo2v2 dual_nero_yolo_rviz.launch.py show_gui:=true
```

Use a custom model path if needed:

```bash
ros2 launch demo2v2 dual_nero_yolo_rviz.launch.py \
  yolo_model_path:=/home/yang/demo_ws/src/demo2/model/yolo26s-pose.pt
```

The launch uses YOLO directly; there is no backend switch.

## Real Arm Dry Run

Start the real drivers without motion:

```bash
ros2 launch demo2v2 dual_nero_yolo_rviz.launch.py \
  start_real_driver:=true command_output_enabled:=true execute_motion:=false \
  left_can_interface:=can0 right_can_interface:=can1
```

## Real Arm Motion

Only after RViz and dry-run look correct:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
source /home/yang/agx_arm_ws/install/setup.bash
source /home/yang/demo_ws/install/setup.bash

ros2 launch demo2v2 dual_nero_yolo_rviz.launch.py \
  show_gui:=true start_real_driver:=true command_output_enabled:=true \
  execute_motion:=true speed_percent:=30 max_command_delta:=0.25 \
  left_can_interface:=can0 right_can_interface:=can1
```
