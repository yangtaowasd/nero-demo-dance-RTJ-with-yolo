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
RealSense、YOLO 姿态识别和双臂 RViz。**默认会连接 `can1`（左臂）和
`can0`（右臂），自动使能真机并下发视觉运动指令。**按 `Ctrl+C` 停止。
编译后脚本会自动执行等价于 `source ~/demo_ws/install/setup.bash` 的命令，
无需在另一个终端手动加载。

运行前必须确认急停可用、CAN 口对应正确、机械臂周围无人，并提前启用
`can0` 和 `can1`。同时停止旧的 `neroarm.launch.py`；脚本检测到旧控制进程时
会拒绝启动真机。启动后有 10 秒运动延迟，可用于最后的安全确认。

跳过已有构建，或关闭 GUI/RViz：

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

仅运行识别和 RViz、完全禁止真机连接与运动：

```bash
./run.sh start_hardware:=false \
  command_output_enabled:=false \
  hardware_execute_motion:=false
```

若 ROS 安装位置不同，可用
`ROS_SETUP=/path/to/setup.bash ./run.sh`。

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
**デフォルトで `can1`（左腕）と `can0`（右腕）に接続し、実機を自動的に
有効化して視覚追従指令を送信します。**終了は `Ctrl+C` です。
ビルド後は `source ~/demo_ws/install/setup.bash` 相当の処理も自動的に
実行されるため、別の端末で手動実行する必要はありません。

実行前に非常停止、CAN ポートの左右、周囲の安全を確認し、`can0` と
`can1` を有効にしてください。古い `neroarm.launch.py` も停止してください。
旧制御プロセスを検出した場合、スクリプトは実機を起動しません。起動後、
実機の動作開始まで 10 秒間の安全待機時間があります。

ビルドを省略する場合、または GUI/RViz を無効にする場合：

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

認識と RViz のみを実行し、実機接続と動作を完全に無効化する場合：

```bash
./run.sh start_hardware:=false \
  command_output_enabled:=false \
  hardware_execute_motion:=false
```

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
**By default it connects to `can1` (left) and `can0` (right), automatically
enables the physical arms, and sends vision motion commands.** Press `Ctrl+C`
to stop.
After building, it automatically performs the equivalent of
`source ~/demo_ws/install/setup.bash`; no separate terminal setup is needed.

Before running, verify the emergency stop, left/right CAN mapping, and a clear
operating area, then bring up `can0` and `can1`. Stop the old
`neroarm.launch.py` as well; the script refuses hardware startup when it
detects an old controller process. A 10-second motion delay after startup
provides time for a final safety check.

To reuse an existing build or disable GUI/RViz:

```bash
./run.sh --no-build show_gui:=false start_rviz:=false
```

To run recognition and RViz while completely disabling hardware and motion:

```bash
./run.sh start_hardware:=false \
  command_output_enabled:=false \
  hardware_execute_motion:=false
```

If ROS is installed elsewhere, run
`ROS_SETUP=/path/to/setup.bash ./run.sh`.
