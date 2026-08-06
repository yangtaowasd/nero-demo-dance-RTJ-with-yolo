# Intel RealSense depth follower

This wrapper starts the official RealSense ROS 2 driver with synchronized
color and depth streams, aligns depth to color, and then starts the independent
depth-based Nero follower.

## Install the driver

```bash
sudo apt-get install -y \
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description
```

Confirm that ROS can see it:

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix realsense2_camera
```

## Build and start

Connect the RealSense camera over USB 3, then run:

```bash
cd /home/yang/demo_ws
colcon build --packages-select nero_assets demo2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash

ros2 launch demo2 realsense_depth_arm.launch.py \
  command_output_enabled:=false
```

The default profile is 640×480 at 30 FPS for both color and depth. The wrapper
enables:

- `enable_color`;
- `enable_depth`;
- `enable_sync`;
- `align_depth.enable`;
- the temporal depth filter.

It consumes these generated topics:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

## Useful overrides

Select a camera by serial number when multiple devices are connected:

```bash
ros2 launch demo2 realsense_depth_arm.launch.py \
  serial_no:="'123456789'" \
  command_output_enabled:=false
```

Use a lighter profile if USB bandwidth or CPU is limited:

```bash
ros2 launch demo2 realsense_depth_arm.launch.py \
  color_profile:=640x480x15 \
  depth_profile:=640x480x15 \
  command_output_enabled:=false
```

Keep real-arm commands disabled until the RViz motion and depth-tracking window
are both stable.
