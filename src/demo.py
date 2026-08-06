#!/usr/bin/env python3
import argparse
import math
import time
from collections.abc import Iterable

from arm_and_revo2 import Nero
from pyAgxArm.api.constants import (
    ROBOT_JOINT_LIMIT_PRESET_DEG,
    ROBOT_JOINT_LIMIT_PRESET_RAD,
    ROBOT_JOINT_NAME,
)


ROBOT = "nero"
JOINT_NAMES = ROBOT_JOINT_NAME[ROBOT]
JOINT_COUNT = len(JOINT_NAMES)
LIMITS_RAD = [tuple(ROBOT_JOINT_LIMIT_PRESET_RAD[ROBOT][name]) for name in JOINT_NAMES]
LIMITS_DEG = [tuple(ROBOT_JOINT_LIMIT_PRESET_DEG[ROBOT][name]) for name in JOINT_NAMES]

# Conservative per-joint demo amplitudes. The final value is also clipped by
# --max-delta and the official pyAgxArm joint limits.
DEFAULT_AMPLITUDES_DEG = [12.0, 10.0, 12.0, 10.0, 14.0, 8.0, 20.0]


def unwrap_msg(ret):
    if ret is None:
        return None
    return ret.msg if hasattr(ret, "msg") else ret


def safe_call(func, default=None):
    try:
        return func()
    except Exception:
        return default


def extract_joint_angles(ret):
    msg = unwrap_msg(ret)
    if msg is None:
        return None

    if isinstance(msg, (list, tuple)) or (
        isinstance(msg, Iterable) and not isinstance(msg, (str, bytes, dict))
    ):
        values = [float(x) for x in msg]
        return values[:JOINT_COUNT] if len(values) >= JOINT_COUNT else None

    if isinstance(msg, dict):
        for key in ("joint_angles", "angles", "joint_angle", "angle", "data", "msg"):
            value = msg.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= JOINT_COUNT:
                return [float(x) for x in value[:JOINT_COUNT]]

    for attr in ("joint_angles", "angles", "joint_angle", "angle", "data", "msg"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if isinstance(value, (list, tuple)) and len(value) >= JOINT_COUNT:
                return [float(x) for x in value[:JOINT_COUNT]]

    return None


def extract_pose(ret):
    msg = unwrap_msg(ret)
    if msg is None:
        return None
    if isinstance(msg, (list, tuple)) or (
        isinstance(msg, Iterable) and not isinstance(msg, (str, bytes, dict))
    ):
        values = [float(x) for x in msg]
        return values[:6] if len(values) >= 6 else None
    if isinstance(msg, dict):
        for key in ("pose", "flange_pose", "tcp_pose", "data", "msg"):
            value = msg.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 6:
                return [float(x) for x in value[:6]]
    for attr in ("pose", "flange_pose", "tcp_pose", "data", "msg"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if isinstance(value, (list, tuple)) and len(value) >= 6:
                return [float(x) for x in value[:6]]
    return None


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def clamp_joints(joints):
    return [
        clamp(float(value), LIMITS_RAD[i][0], LIMITS_RAD[i][1])
        for i, value in enumerate(joints)
    ]


def parse_deg_list(text):
    if isinstance(text, (list, tuple)):
        parts = text
    else:
        parts = str(text).replace(",", " ").split()
    return [math.radians(float(v)) for v in parts]


def parse_joint_selection(text):
    if text.strip().lower() in ("all", "*"):
        return list(range(JOINT_COUNT))
    result = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.startswith("joint"):
            part = part[5:]
        if part.startswith("j"):
            part = part[1:]
        index = int(part) - 1
        if index < 0 or index >= JOINT_COUNT:
            raise ValueError(f"joint index out of range: {index + 1}")
        result.append(index)
    return sorted(set(result))


def format_joints(joints):
    deg = [math.degrees(value) for value in joints]
    return " ".join(f"j{i + 1}={deg[i]:7.2f}deg" for i in range(JOINT_COUNT))


def format_pose(pose):
    xyz = " ".join(f"{name}={pose[i]: .4f}m" for i, name in enumerate(("x", "y", "z")))
    rpy = " ".join(f"{name}={math.degrees(pose[i + 3]): .1f}deg" for i, name in enumerate(("r", "p", "y")))
    return f"{xyz}  {rpy}"


class NeroRealDemo:
    def __init__(self, args):
        self.args = args
        self.arm = Nero(channel=args.can)
        self.enabled = False

    def connect(self):
        print(f"connect Nero on {self.args.can}")
        self.arm.connect()
        time.sleep(0.5)

        if self.args.enable:
            print("enable Nero")
            deadline = time.time() + self.args.enable_timeout
            while not self.arm.enable():
                if time.time() > deadline:
                    raise RuntimeError("enable timeout")
                time.sleep(0.05)
            self.enabled = True

        if self.args.speed is not None:
            print(f"set speed percent: {self.args.speed}")
            self.arm.set_speed_percent(self.args.speed)

    def disconnect(self):
        if self.args.disable_on_exit and self.enabled:
            print("disable Nero")
            safe_call(lambda: self.arm.disable())
        safe_call(lambda: self.arm.disconnect())

    def read_joints(self, wait=True):
        deadline = time.time() + self.args.read_timeout
        while True:
            joints = extract_joint_angles(self.arm.get_joint_angles())
            if joints is not None:
                return joints
            if not wait or time.time() > deadline:
                return None
            time.sleep(0.05)

    def read_flange_pose(self, wait=True):
        deadline = time.time() + self.args.read_timeout
        while True:
            pose = extract_pose(self.arm.get_flange_pose())
            if pose is not None:
                return pose
            if not wait or time.time() > deadline:
                return None
            time.sleep(0.05)

    def print_limits(self):
        print("official Nero joint limits:")
        for i, name in enumerate(JOINT_NAMES):
            lo_deg, hi_deg = LIMITS_DEG[i]
            lo_rad, hi_rad = LIMITS_RAD[i]
            print(
                f"  j{i + 1} {name:<7} "
                f"{lo_deg:8.2f}..{hi_deg:8.2f} deg  "
                f"{lo_rad:8.4f}..{hi_rad:8.4f} rad"
            )

    def send(self, joints, label):
        joints = clamp_joints(joints)
        print(f"{label}: {format_joints(joints)}")
        if not self.args.execute:
            print("dry-run: add --execute to move the real arm")
            return
        self.arm.move_j(joints)
        time.sleep(self.args.step_wait)

    def send_pose(self, pose, label, linear=True):
        print(f"{label}: {format_pose(pose)}")
        if not self.args.execute:
            print("dry-run: add --execute to move the real arm")
            return
        if linear:
            self.arm.move_l(pose)
        else:
            self.arm.move_p(pose)
        time.sleep(self.args.cartesian_wait)

    def make_offset_pose(self, center, offsets):
        target = center[:]
        for i, value in enumerate(offsets):
            target[i] = center[i] + value
        return clamp_joints(target)

    def demo_wave(self, center, selected, amplitudes):
        print("demo mode: wave")
        total_steps = max(2, int(self.args.duration / self.args.step_wait))
        phase_offsets = [i * math.pi / 3.0 for i in range(JOINT_COUNT)]
        for step in range(total_steps):
            phase = 2.0 * math.pi * step / total_steps
            offsets = [0.0] * JOINT_COUNT
            for i in selected:
                offsets[i] = amplitudes[i] * math.sin(phase + phase_offsets[i])
            self.send(self.make_offset_pose(center, offsets), f"wave {step + 1}/{total_steps}")
        self.send(center, "return center")

    def demo_each(self, center, selected, amplitudes):
        print("demo mode: each joint")
        for cycle in range(self.args.cycles):
            print(f"cycle {cycle + 1}/{self.args.cycles}")
            for i in selected:
                target_pos = center[:]
                target_neg = center[:]
                target_pos[i] += amplitudes[i]
                target_neg[i] -= amplitudes[i]
                self.send(target_pos, f"j{i + 1} +")
                self.send(target_neg, f"j{i + 1} -")
                self.send(center, f"j{i + 1} center")

    def demo_choreo(self, center, selected, amplitudes):
        print("demo mode: choreography")
        signs = [
            [1, 0, -1, 0, 1, 0, -1],
            [-1, 1, 0, 1, -1, 1, 0],
            [0, -1, 1, -1, 0, -1, 1],
            [1, 1, -1, 1, -1, 0, 1],
            [-1, 0, 1, 0, 1, -1, -1],
        ]
        for cycle in range(self.args.cycles):
            print(f"cycle {cycle + 1}/{self.args.cycles}")
            for pose_index, sign_row in enumerate(signs):
                offsets = [0.0] * JOINT_COUNT
                for i in selected:
                    offsets[i] = amplitudes[i] * sign_row[i]
                self.send(self.make_offset_pose(center, offsets), f"pose {pose_index + 1}")
        self.send(center, "return center")

    def plane_axes(self):
        plane = self.args.plane.lower()
        if plane == "xy":
            return 0, 1
        if plane == "xz":
            return 0, 2
        if plane == "yz":
            return 1, 2
        raise ValueError("--plane must be auto, xy, xz, or yz")

    def xyz_offset(self, u, v, w=0.0):
        plane = self.args.plane.lower()
        if plane == "auto":
            return u, v, w
        offset = [0.0, 0.0, 0.0]
        axis_u, axis_v = self.plane_axes()
        axis_w = ({0, 1, 2} - {axis_u, axis_v}).pop()
        offset[axis_u] = u
        offset[axis_v] = v
        offset[axis_w] = w
        return tuple(offset)

    def pose_from_offset(self, center, dx, dy, dz):
        pose = center[:]
        pose[0] = center[0] + dx
        pose[1] = center[1] + dy
        pose[2] = center[2] + dz
        return pose

    def pose_from_motion_point(self, center, point):
        dx, dy, dz = point[:3]
        pose = self.pose_from_offset(center, dx, dy, dz)
        if len(point) >= 6:
            pose[3] = center[3] + point[3]
            pose[4] = center[4] + point[4]
            pose[5] = center[5] + point[5]
        return pose

    def shape_circle(self):
        radius = self.args.size
        z_amp = self.args.z_size
        steps = max(12, self.args.shape_steps)
        return [
            self.xyz_offset(
                radius * math.cos(2.0 * math.pi * i / steps),
                radius * math.sin(2.0 * math.pi * i / steps),
                z_amp * math.sin(4.0 * math.pi * i / steps),
            )
            for i in range(steps + 1)
        ]

    def shape_square(self):
        half = self.args.size
        z_amp = self.args.z_size
        per_edge = max(2, self.args.shape_steps // 4)
        corners = [(half, half), (-half, half), (-half, -half), (half, -half), (half, half)]
        points = []
        total_edges = len(corners) - 1
        for edge_index, (start, end) in enumerate(zip(corners[:-1], corners[1:])):
            for i in range(per_edge):
                t = i / per_edge
                phase = (edge_index + t) / total_edges
                points.append(self.xyz_offset(
                    start[0] * (1.0 - t) + end[0] * t,
                    start[1] * (1.0 - t) + end[1] * t,
                    z_amp * math.sin(2.0 * math.pi * phase),
                ))
        points.append(self.xyz_offset(corners[-1][0], corners[-1][1], 0.0))
        return points

    def shape_eight(self):
        radius = self.args.size
        z_amp = self.args.z_size
        steps = max(16, self.args.shape_steps)
        return [
            self.xyz_offset(
                radius * math.sin(2.0 * math.pi * i / steps),
                0.5 * radius * math.sin(4.0 * math.pi * i / steps),
                z_amp * math.cos(2.0 * math.pi * i / steps),
            )
            for i in range(steps + 1)
        ]

    def shape_star(self):
        outer = self.args.size
        inner = self.args.size * 0.42
        z_amp = self.args.z_size
        points = []
        for i in range(11):
            radius = outer if i % 2 == 0 else inner
            angle = math.pi / 2.0 + i * math.pi / 5.0
            z = z_amp if i % 2 == 0 else -z_amp
            if i == 10:
                z = z_amp
            points.append(self.xyz_offset(radius * math.cos(angle), radius * math.sin(angle), z))
        return points

    def shape_flower(self):
        radius = self.args.size
        z_amp = self.args.z_size
        steps = max(36, self.args.shape_steps)
        petals = max(3, self.args.petals)
        points = []
        for i in range(steps + 1):
            theta = 2.0 * math.pi * i / steps
            r = radius * (0.45 + 0.55 * math.sin(petals * theta))
            points.append(self.xyz_offset(
                r * math.cos(theta),
                r * math.sin(theta),
                z_amp * math.sin((petals + 1) * theta),
            ))
        return points

    def shape_helix(self):
        radius = self.args.size
        z_amp = self.args.z_size
        steps = max(24, self.args.shape_steps)
        turns = max(1, self.args.turns)
        points = []
        for i in range(steps + 1):
            t = i / steps
            theta = 2.0 * math.pi * turns * t
            z = z_amp * math.sin(2.0 * math.pi * t)
            points.append(self.xyz_offset(radius * math.cos(theta), radius * math.sin(theta), z))
        return points

    def human_reach_points(self):
        reach = self.args.human_reach
        side = self.args.human_side
        lift = self.args.human_lift
        steps = max(24, self.args.shape_steps)
        points = []
        for i in range(steps + 1):
            t = i / steps
            # Smooth reach out and return, like a hand extending and coming back.
            reach_phase = math.sin(math.pi * t)
            # A slower side sweep makes it read less like a machine line.
            side_phase = math.sin(2.0 * math.pi * t)
            # Lift peaks near the middle and softens at both ends.
            lift_phase = math.sin(math.pi * t) ** 1.4
            dx = reach * reach_phase
            dy = side * side_phase
            dz = lift * lift_phase
            wrist_pitch = math.radians(self.args.human_wrist_deg) * 0.35 * math.sin(math.pi * t)
            points.append((dx, dy, dz, 0.0, wrist_pitch, 0.0))
        return points

    def human_wave_points(self):
        reach = self.args.human_reach
        side = self.args.human_side
        lift = self.args.human_lift
        steps = max(32, self.args.shape_steps)
        waves = max(1, self.args.human_waves)
        points = []
        for i in range(steps + 1):
            t = i / steps
            settle = math.sin(math.pi * t)
            dx = reach * (0.45 + 0.25 * math.sin(math.pi * t))
            dy = side * math.sin(2.0 * math.pi * waves * t) * settle
            dz = lift * (0.65 + 0.35 * math.sin(4.0 * math.pi * waves * t)) * settle
            wrist = math.radians(self.args.human_wrist_deg)
            roll = 0.35 * wrist * math.sin(2.0 * math.pi * waves * t) * settle
            pitch = 0.20 * wrist * math.sin(math.pi * t)
            yaw = wrist * math.sin(2.0 * math.pi * waves * t + math.pi / 5.0) * settle
            points.append((dx, dy, dz, roll, pitch, yaw))
        return points

    def make_shape_points(self, mode=None):
        mode = mode or self.args.mode
        if mode == "circle":
            return self.shape_circle()
        if mode == "square":
            return self.shape_square()
        if mode == "eight":
            return self.shape_eight()
        if mode == "star":
            return self.shape_star()
        if mode == "flower":
            return self.shape_flower()
        if mode == "helix":
            return self.shape_helix()
        if mode == "human-reach":
            return self.human_reach_points()
        if mode == "human-wave":
            return self.human_wave_points()
        raise ValueError(f"unknown shape mode: {mode}")

    def demo_shape(self, center_pose, mode=None):
        mode = mode or self.args.mode
        print(f"demo mode: draw {mode}")
        print(f"center flange pose: {format_pose(center_pose)}")
        print(
            f"plane={self.args.plane}, size={self.args.size:.3f}m, "
            f"z_size={self.args.z_size:.3f}m, points={self.args.shape_steps}, "
            f"command={self.args.cartesian_command}"
        )
        points = self.make_shape_points(mode)
        linear = self.args.cartesian_command == "move_l"
        if points:
            first_pose = self.pose_from_motion_point(center_pose, points[0])
            self.send_pose(first_pose, "go shape start", linear=False)
        for i, point in enumerate(points, start=1):
            self.send_pose(
                self.pose_from_motion_point(center_pose, point),
                f"{mode} {i}/{len(points)}",
                linear=linear,
            )
        self.send_pose(center_pose, "return center", linear=False)

    def prepare_joint_demo(self):
        center = self.read_joints(wait=True)
        if center is None:
            raise RuntimeError("cannot read current joint angles")
        selected = parse_joint_selection(self.args.joints)
        amplitudes = parse_deg_list(self.args.amplitudes)
        if len(amplitudes) != JOINT_COUNT:
            raise ValueError(f"--amplitudes needs {JOINT_COUNT} comma-separated values")
        max_delta = math.radians(self.args.max_delta)
        amplitudes = [min(abs(amplitudes[i]), max_delta) for i in range(JOINT_COUNT)]
        print(f"current center: {format_joints(center)}")
        print(f"selected joints: {[i + 1 for i in selected]}")
        print(f"amplitudes deg: {[round(math.degrees(v), 2) for v in amplitudes]}")
        return center, selected, amplitudes

    def run_joint_mode(self, mode):
        center, selected, amplitudes = self.prepare_joint_demo()
        if mode == "wave":
            self.demo_wave(center, selected, amplitudes)
        elif mode == "each":
            self.demo_each(center, selected, amplitudes)
        elif mode == "choreo":
            self.demo_choreo(center, selected, amplitudes)
        else:
            raise ValueError(f"unknown joint mode: {mode}")

    def run_shape_mode(self, mode):
        center_pose = self.read_flange_pose(wait=True)
        if center_pose is None:
            raise RuntimeError("cannot read current flange pose")
        self.demo_shape(center_pose, mode)

    def run_all_modes(self):
        joint_modes = ["choreo", "wave", "each"]
        shape_modes = ["circle", "square", "eight", "star", "flower", "helix", "human-reach", "human-wave"]
        round_index = 0
        while self.args.loop or round_index < self.args.repeat:
            round_index += 1
            print(f"========== all demo round {round_index} ==========")
            for mode in joint_modes:
                print(f"---------- {mode} ----------")
                self.run_joint_mode(mode)
                time.sleep(self.args.pause_between)
            for mode in shape_modes:
                print(f"---------- {mode} ----------")
                self.run_shape_mode(mode)
                time.sleep(self.args.pause_between)

    def run(self):
        self.connect()
        self.print_limits()

        if self.args.mode == "all":
            print(f"execute={self.args.execute}")
            if self.args.countdown > 0:
                print("starting full demo loop soon. Keep emergency stop reachable.")
                for value in range(self.args.countdown, 0, -1):
                    print(value)
                    time.sleep(1.0)
            self.run_all_modes()
            return

        shape_modes = {"circle", "square", "eight", "star", "flower", "helix", "human-reach", "human-wave"}
        if self.args.mode in shape_modes:
            center_pose = self.read_flange_pose(wait=True)
            if center_pose is None:
                raise RuntimeError("cannot read current flange pose")
            print(f"execute={self.args.execute}")
            if self.args.countdown > 0:
                print("starting Cartesian drawing demo soon. Keep emergency stop reachable.")
                for value in range(self.args.countdown, 0, -1):
                    print(value)
                    time.sleep(1.0)
            self.demo_shape(center_pose)
            return

        print(f"execute={self.args.execute}")

        if self.args.countdown > 0:
            print("starting demo soon. Keep emergency stop reachable.")
            for value in range(self.args.countdown, 0, -1):
                print(value)
                time.sleep(1.0)

        self.run_joint_mode(self.args.mode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real Nero demo. Can run joint motions or draw small Cartesian shapes."
    )
    parser.add_argument("--can", default="can0", help="CAN channel")
    parser.add_argument(
        "--mode",
        choices=[
            "wave", "each", "choreo",
            "circle", "square", "eight", "star", "flower", "helix",
            "human-reach", "human-wave", "all",
        ],
        default="choreo",
    )
    parser.add_argument("--joints", default="all", help="all or comma list, e.g. 1,3,5,7")
    parser.add_argument(
        "--amplitudes",
        default=",".join(str(v) for v in DEFAULT_AMPLITUDES_DEG),
        help="7 comma-separated amplitudes in degrees",
    )
    parser.add_argument("--max-delta", type=float, default=18.0, help="max per-joint offset in degrees")
    parser.add_argument("--speed", type=int, default=20, help="speed percent")
    parser.add_argument("--duration", type=float, default=8.0, help="wave mode duration in seconds")
    parser.add_argument("--cycles", type=int, default=1, help="each/choreo cycles")
    parser.add_argument("--repeat", type=int, default=1, help="all mode repeat count")
    parser.add_argument("--loop", action="store_true", help="repeat all mode forever")
    parser.add_argument("--pause-between", type=float, default=0.5, help="seconds between all-mode actions")
    parser.add_argument("--step-wait", type=float, default=1.0, help="seconds after every move_j")
    parser.add_argument("--size", type=float, default=0.04, help="shape radius/half-side in meters")
    parser.add_argument(
        "--plane",
        choices=["auto", "xy", "xz", "yz"],
        default="auto",
        help="auto uses x/y/z together; xy/xz/yz draw mainly in one plane with third-axis lift",
    )
    parser.add_argument("--z-size", type=float, default=0.015, help="third-axis lift amplitude in meters")
    parser.add_argument("--shape-steps", type=int, default=36, help="number of drawing points")
    parser.add_argument("--petals", type=int, default=5, help="flower mode petal count")
    parser.add_argument("--turns", type=int, default=2, help="helix mode turn count")
    parser.add_argument("--human-reach", type=float, default=0.06, help="human modes forward reach in meters")
    parser.add_argument("--human-side", type=float, default=0.035, help="human modes side sweep in meters")
    parser.add_argument("--human-lift", type=float, default=0.045, help="human modes vertical lift in meters")
    parser.add_argument("--human-waves", type=int, default=3, help="human-wave side-to-side repetitions")
    parser.add_argument("--human-wrist-deg", type=float, default=8.0, help="human modes wrist-like orientation amplitude")
    parser.add_argument("--cartesian-command", choices=["move_l", "move_p"], default="move_l")
    parser.add_argument("--cartesian-wait", type=float, default=0.35, help="seconds after every Cartesian point")
    parser.add_argument("--execute", action="store_true", help="actually send move_j commands")
    parser.add_argument("--enable", action="store_true", default=True)
    parser.add_argument("--no-enable", dest="enable", action="store_false")
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=5.0)
    parser.add_argument("--countdown", type=int, default=3)
    return parser.parse_args()


def main():
    demo = NeroRealDemo(parse_args())
    try:
        demo.run()
    finally:
        demo.disconnect()


if __name__ == "__main__":
    main()
