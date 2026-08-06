# URDF origin

`demo2` installs and loads its own Nero URDF files from `urdf/`. The mesh
files referenced by the model are supplied by the renamed `armbycontroller`
ROS package.

## Upstream

The Nero model originated from:

```text
agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf
```

The upstream `agx_arm_urdf` repository is MIT licensed. The two
`nero_with_{left,right}_revo2_description.xacro` files that were previously in
this repository matched the upstream files byte for byte.

The original `nero_description.urdf` also matched upstream when it entered this
repository. It was later adjusted only in the lower, upper, and velocity limits
of joints 1 through 7.

`pyAgxArm` is a separate Python/CAN driver library. It contains no URDF or
Xacro files, so the model was not copied from `pyAgxArm`.

## Dependency boundary

- `demo2`: runtime logic plus its YOLO model, URDF, and RViz files.
- `armbycontroller`: upstream mesh resources used by the URDF.
- `pyAgxArm`: Nero hardware API used by the driver code.

The bundled librealsense2 runtime under `third_party/librealsense2` is Apache
2.0 licensed; its package copyright notice is retained alongside the binary.
