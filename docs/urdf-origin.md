# URDF origin

`demo2` installs and loads its own Nero URDF files from `urdf/`. The mesh
files referenced by the model are supplied by the renamed `armbycontroller`
ROS package.

## Upstream

The Nero model originated from:

```text
agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf
```

The upstream `agx_arm_urdf` repository is MIT licensed. The original
`nero_description.urdf` matched upstream when it entered this repository. It
was later adjusted only in the lower, upper, and velocity limits of joints 1
through 7.

## Dependency boundary

- `demo2`: RealSense runtime logic plus its YOLO model, URDF, and RViz files.
- `armbycontroller`: upstream mesh resources used by the URDF.

The bundled librealsense2 runtime under `third_party/librealsense2` is Apache
2.0 licensed; its package copyright notice is retained alongside the binary.
