# 一键启动 / ワンクリック起動 / One-click startup

## 中文

### 环境

- Ubuntu 22.04、ROS 2 Humble、Intel RealSense D435i。
- 本包位于 ROS 2 工作区的 `src` 目录，例如
  `~/demo_ws/src/nero-demo-dance-RTJ-with-yolo`。
- 首次构建前安装 CPU 版 PyTorch：

```bash
python3 -m pip install --user torch \
  --index-url https://download.pytorch.org/whl/cpu
```

### 一键运行

```bash
cd ~/demo_ws/src/nero-demo-dance-RTJ-with-yolo
./run.sh
```

脚本会自动加载 ROS 环境、增量编译 `demo2`、加载工作区环境，并启动
RealSense、YOLO 姿态识别和双臂 RViz。默认不会连接机械臂，也不会下发
运动指令。按 `Ctrl+C` 停止。

跳过已有构建，或关闭 GUI/RViz：

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

只有确认识别、深度和 RViz 动作稳定后，才可显式启用真机：

```bash
./run.sh start_hardware:=true \
  command_output_enabled:=true \
  hardware_execute_motion:=true \
  left_can_interface:=can1 \
  right_can_interface:=can0
```

真机启动前必须确认急停可用、CAN 口对应正确且机械臂周围无人。若 ROS
安装位置不同，可用 `ROS_SETUP=/path/to/setup.bash ./run.sh`。

## 日本語

### 動作環境

- Ubuntu 22.04、ROS 2 Humble、Intel RealSense D435i。
- このパッケージを ROS 2 ワークスペースの `src` 配下に置きます。例：
  `~/demo_ws/src/nero-demo-dance-RTJ-with-yolo`。
- 初回ビルド前に CPU 版 PyTorch をインストールします。

```bash
python3 -m pip install --user torch \
  --index-url https://download.pytorch.org/whl/cpu
```

### ワンクリック起動

```bash
cd ~/demo_ws/src/nero-demo-dance-RTJ-with-yolo
./run.sh
```

スクリプトは ROS 環境の読み込み、`demo2` の差分ビルド、ワークスペースの
読み込みを行い、RealSense、YOLO 姿勢推定、両腕の RViz 表示を起動します。
デフォルトでは実機に接続せず、動作指令も送信しません。終了は `Ctrl+C`
です。

ビルドを省略する場合、または GUI/RViz を無効にする場合：

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

認識、深度、RViz の動作を十分確認した後に限り、実機を明示的に有効化して
ください。

```bash
./run.sh start_hardware:=true \
  command_output_enabled:=true \
  hardware_execute_motion:=true \
  left_can_interface:=can1 \
  right_can_interface:=can0
```

実機起動前に、非常停止、CAN ポートの左右、周囲の安全を確認してください。
ROS の場所が異なる場合は
`ROS_SETUP=/path/to/setup.bash ./run.sh` を使用します。

## English

### Requirements

- Ubuntu 22.04, ROS 2 Humble, and an Intel RealSense D435i.
- This package must be under a ROS 2 workspace `src` directory, for example
  `~/demo_ws/src/nero-demo-dance-RTJ-with-yolo`.
- Install CPU PyTorch before the first build:

```bash
python3 -m pip install --user torch \
  --index-url https://download.pytorch.org/whl/cpu
```

### One-click run

```bash
cd ~/demo_ws/src/nero-demo-dance-RTJ-with-yolo
./run.sh
```

The script sources ROS, builds `demo2` incrementally, sources the workspace,
and starts RealSense capture, YOLO pose detection, and the dual-arm RViz view.
Robot hardware and motion commands are disabled by default. Press `Ctrl+C` to
stop.

To reuse an existing build or disable GUI/RViz:

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

Enable the physical arms only after recognition, depth, and RViz motion have
been verified:

```bash
./run.sh start_hardware:=true \
  command_output_enabled:=true \
  hardware_execute_motion:=true \
  left_can_interface:=can1 \
  right_can_interface:=can0
```

Before hardware startup, verify the emergency stop, left/right CAN mapping,
and a clear operating area. If ROS is installed elsewhere, run
`ROS_SETUP=/path/to/setup.bash ./run.sh`.
