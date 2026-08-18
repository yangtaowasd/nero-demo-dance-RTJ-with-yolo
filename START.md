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

一键脚本默认不复用旧的人体方向标定；每次启动请面向当前相机自然站立，保持
双肩和双髋可见并静止约 2 秒。这样可避免相机移动后旧标定触发“人物旋转超限”
而禁止运动。相机和站位完全没变时，可用
`./run.sh load_calibration_on_start:=true` 明确复用已保存标定。

运行前必须确认急停可用、CAN 口对应正确、机械臂周围无人，并提前启用
`can0` 和 `can1`。同时停止旧的 `neroarm.launch.py`；脚本检测到旧控制进程时
会拒绝启动真机。启动后有 5 秒运动延迟，可用于最后的安全确认。

真机采用两阶段使能：第一次仅临时使能一次，读取固件、机械臂状态、七轴
使能状态和七轴位置，随后立即失能、断开并等待 0.5 秒；第二次重连后才是
正式运行使能。如果读取到上次软件留下的电子急停，重连后会像其他 Nero 项目
一样自动 reset 一次并确认解除；物理急停未释放或 reset 失败时仍禁止运行。
只有第二阶段确认状态为 `NORMAL`、七轴全部使能且位置反馈完整后，才开始上述
5 秒运动延迟。第一阶段不发送运动指令，失败后也不会自动反复重试，必须排查
后重启脚本。标定和 5 秒延迟期间不会发送 `move_j`；第一条有效视觉指令从最新
实测关节位置开始，只前进一个 20 Hz 限速小步，避免某一关节突然抖动。

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

ワンクリック起動では古い人物方向キャリブレーションを既定で再利用しません。
毎回、現在のカメラに向かって自然に立ち、両肩と両腰を見せたまま約 2 秒静止
してください。カメラ移動後に古い基準で回転制限が作動することを防ぎます。
カメラと立ち位置が全く変わっていない場合は、
`./run.sh load_calibration_on_start:=true` で保存済み基準を再利用できます。

実行前に非常停止、CAN ポートの左右、周囲の安全を確認し、`can0` と
`can1` を有効にしてください。古い `neroarm.launch.py` も停止してください。
旧制御プロセスを検出した場合、スクリプトは実機を起動しません。起動後、
実機の動作開始まで 5 秒間の安全待機時間があります。

実機は 2 段階で有効化されます。第 1 段階では一度だけ一時的に有効化し、
ファームウェア、アーム状態、7 軸の有効状態と現在位置を読み取った後、直ちに
無効化・切断して 0.5 秒待機します。再接続後の第 2 段階が実際の運転用有効化
です。前回のソフトウェア電子停止が残っている場合は、他の Nero プロジェクト
と同様に再接続後一度だけ reset し、解除を確認します。物理非常停止が未解除、
または reset 失敗の場合は動作を禁止します。状態が `NORMAL`、全 7 軸が有効、
位置フィードバックが完全であることを確認してから、上記 5 秒の安全待機を
開始します。第 1 段階では動作指令を送信せず、失敗時に自動で繰り返しません。
キャリブレーション中と 5 秒待機中は `move_j` を送信しません。最初の有効な
視覚指令は最新の実測関節位置から 20 Hz の 1 制限ステップだけ進むため、単一
関節の急な動きを防ぎます。

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

The one-click script does not reuse an old person-orientation calibration by
default. Face the current camera, keep both shoulders and hips visible, and
stand naturally still for about two seconds on every start. This prevents a
moved camera from triggering the person-rotation safety gate against a stale
reference. If the camera and standing position are unchanged, explicitly reuse
the saved calibration with `./run.sh load_calibration_on_start:=true`.

Before running, verify the emergency stop, left/right CAN mapping, and a clear
operating area, then bring up `can0` and `can1`. Stop the old
`neroarm.launch.py` as well; the script refuses hardware startup when it
detects an old controller process. A 5-second motion delay after startup
provides time for a final safety check.

Hardware uses two-stage enable. Stage 1 issues exactly one temporary enable,
reads firmware, arm status, all seven enable flags, and all seven joint
positions, then immediately disables, disconnects, and waits 0.5 seconds.
Stage 2 reconnects and performs the real operational enable. The 5-second
motion delay starts only after `NORMAL` status, seven enabled joints, and
complete position feedback are confirmed. If a software electronic stop was
left by the previous run, startup resets it once after reconnecting and
verifies the reset, matching the other Nero project. A physical or
unresettable stop remains blocked. Stage 1 sends no motion command and does not
repeat automatically after failure; diagnose the cause and restart. No
`move_j` is sent during calibration or the five-second delay. The first valid
vision command advances only one 20 Hz rate-limited step from the latest
measured joints, preventing a single-joint startup twitch.

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
