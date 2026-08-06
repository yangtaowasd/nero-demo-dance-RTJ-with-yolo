# Generic aligned-depth camera follower

`depth_camera_arm.launch.py` runs the split recognition and Nero controller
against any driver that provides:

- a BGR/RGB color `sensor_msgs/Image`;
- a depth image spatially aligned to that color image;
- the color-camera `sensor_msgs/CameraInfo`.

Supported depth encodings are `16UC1` (millimetres by default) and `32FC1`
(metres). Color, aligned depth, and `CameraInfo` must have identical
dimensions.

```bash
ros2 launch demo2 depth_camera_arm.launch.py \
  color_topic:=/camera/color/image_raw \
  aligned_depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  command_output_enabled:=false
```

To isolate a fault, launch only `depth_pose_detector.launch.py` with the same
three topic overrides and inspect `/realsense/arm_pose_debug`. Then launch
`depth_arm_control.launch.py` separately once the eight-point output is stable.

The detector rejects timestamp mismatches, invalid depth, and missing pose
landmarks. The controller independently rejects excessive 3-D jumps,
implausible limb-length changes, stale input, and poor IK solutions. An invalid
frame closes the hardware-command gate.

See [RealSense depth recognition](realsense-depth-follower.md) for the portable
point-cloud contract and migration instructions.
