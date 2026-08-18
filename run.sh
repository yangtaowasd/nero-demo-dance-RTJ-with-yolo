#!/usr/bin/env bash

set -Ee -o pipefail

usage() {
  printf '%s\n' \
    'Usage: ./run.sh [--no-build] [ROS launch arguments...]' \
    '' \
    'Builds demo2 incrementally and starts the complete RealSense pipeline.' \
    'Physical robot hardware and vision motion output are enabled by default.' \
    'Check the emergency stop, CAN mapping, and operating area before running.' \
    '' \
    'Examples:' \
    '  ./run.sh' \
    '  ./run.sh --no-build show_gui:=false start_rviz:=false' \
    '  ./run.sh start_hardware:=false command_output_enabled:=false hardware_execute_motion:=false' \
    "  ./run.sh serial_no:=\"'123456789'\"" \
    '' \
    'Environment:' \
    '  ROS_SETUP=/opt/ros/humble/setup.bash  Override the ROS setup file.'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

build_enabled=true
if [[ "${1:-}" == "--no-build" ]]; then
  build_enabled=false
  shift
fi

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(dirname -- "$package_dir")"
if [[ "$(basename -- "$source_dir")" != "src" ]]; then
  printf 'error: place this package under <workspace>/src before running\n' >&2
  exit 1
fi
workspace_dir="$(dirname -- "$source_dir")"

ros_setup="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
if [[ ! -f "$ros_setup" ]]; then
  printf 'error: ROS setup file not found: %s\n' "$ros_setup" >&2
  printf 'set ROS_SETUP to the correct setup.bash path\n' >&2
  exit 1
fi

# ROS setup scripts are not guaranteed to be compatible with nounset.
source "$ros_setup"

for required_command in python3 colcon ros2; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$required_command" >&2
    exit 1
  fi
done

if ! python3 -c 'import torch' >/dev/null 2>&1; then
  printf '%s\n' \
    'error: Python PyTorch is required for the C++ LibTorch build/runtime.' \
    'install it with the command shown in START.md' >&2
  exit 1
fi

if [[ "$build_enabled" == true ]]; then
  (
    cd -- "$workspace_dir"
    colcon build --packages-select demo2 --symlink-install
  )
elif [[ ! -f "$workspace_dir/install/setup.bash" ]]; then
  printf 'error: install/setup.bash is missing; run without --no-build first\n' >&2
  exit 1
fi

source "$workspace_dir/install/setup.bash"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

launch_arguments=(
  "start_hardware:=true"
  "command_output_enabled:=true"
  "hardware_execute_motion:=true"
)

# Let explicit command-line values override the three real-hardware defaults.
for user_argument in "$@"; do
  case "$user_argument" in
    start_hardware:=*) launch_arguments[0]="$user_argument" ;;
    command_output_enabled:=*) launch_arguments[1]="$user_argument" ;;
    hardware_execute_motion:=*) launch_arguments[2]="$user_argument" ;;
    *) launch_arguments+=("$user_argument") ;;
  esac
done

exec ros2 launch demo2 realsense_depth_arm.launch.py "${launch_arguments[@]}"
