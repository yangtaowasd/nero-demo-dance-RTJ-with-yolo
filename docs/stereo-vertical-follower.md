# Vertical stereo arm follower

This is an independent experimental pipeline. It does not replace or import
the existing monocular pose-to-angle node.

## Camera placement

- Keep the laptop camera below and the fisheye camera above.
- Use a rigid vertical separation of roughly 0.4–0.8 m.
- Aim both cameras at the torso and keep both devices fixed after calibration.
- Use the same image resolution for calibration and tracking.

## 1. Calibrate

Use a printed chessboard and measure one square in metres. `columns` and
`rows` mean inner corners, not printed squares. Capture at least 20–25 pairs
with the board at different depths, positions and tilts.

```bash
cd /home/yang/demo_ws/src/demo2
python3 src/demo2/calibrate_vertical_stereo.py \
  --lower-id 0 --upper-id 2 \
  --columns 9 --rows 6 --square-size-m 0.025 \
  --samples 25 --output config/my_vertical_stereo.yaml
```

Press Space to accept a pair, `U` to undo, and `Q` to finish after at least 12
pairs. Do not use `stereo_vertical_example.yaml`: it is deliberately marked as
a non-runnable template.

## 2. RViz dry run

```bash
cd /home/yang/demo_ws
colcon build --packages-select demo2
source /home/yang/agx_arm_ws/install/setup.bash
source install/setup.bash
ros2 launch demo2 stereo_vertical_arm.launch.py \
  calibration_file:=/home/yang/demo_ws/src/demo2/config/my_vertical_stereo.yaml \
  lower_camera_id:=0 upper_camera_id:=2 \
  command_output_enabled:=false
```

Hold a straight-arm neutral pose for two seconds. The node triangulates the
eight shoulder/elbow/wrist/hip landmarks, rejects bad ray geometry and bone
length jumps, and solves direction-constrained IK against the Nero URDF.

## 3. Hardware output

Keep `command_output_enabled:=false` until both arms move correctly in RViz.
When enabled, commands are published to:

- `/left/neroarm/command_joints`
- `/right/neroarm/command_joints`

The experimental launch file intentionally does not start a physical arm
driver. Start and configure the driver separately only after dry-run testing.
