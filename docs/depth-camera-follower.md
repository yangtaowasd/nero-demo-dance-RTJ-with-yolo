# Depth-camera arm follower

This independent pipeline replaces monocular or stereo-estimated Z with an
aligned hardware depth image. Existing pose nodes remain unchanged.

## Required camera topics

The camera driver must publish:

- a BGR/RGB color image;
- a depth image spatially aligned to that color image;
- the color camera `CameraInfo`.

Supported depth encodings are `16UC1` (millimetres by default) and `32FC1`
(metres). Color, aligned depth and `CameraInfo` must have identical dimensions.

Default topic names are suitable for a common RealSense ROS 2 configuration:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

For another driver, inspect its topics and override all three names.

## Build

```bash
cd /home/yang/demo_ws
colcon build --packages-select nero_assets demo2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
```

The depth-camera driver must already be running with depth-to-color alignment
enabled. Confirm the actual names first:

```bash
ros2 topic list | grep -E 'color|depth|camera_info'
```

## RViz dry run

```bash
ros2 launch demo2 depth_camera_arm.launch.py \
  color_topic:=/camera/camera/color/image_raw \
  aligned_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/camera/color/camera_info \
  command_output_enabled:=false
```

Hold a straight-arm neutral pose for two seconds. Keep hardware commands off
until both RViz arms follow correctly.

## Safety and rejection

The node stops hardware command publication when any of these checks fails:

- color/depth timestamp mismatch;
- missing or invalid depth around a landmark;
- color, depth and camera-info dimensions do not match;
- excessive 3-D point jump or bone-length change;
- excessive IK direction error;
- stale pose data.

When using a non-millimetre `16UC1` source, override `depth_uint16_scale` on the
node. Most ROS depth drivers use `0.001` metre per unit.
