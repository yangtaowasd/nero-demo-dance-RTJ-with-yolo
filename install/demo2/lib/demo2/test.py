#!/usr/bin/env python3
import time
import traceback
from collections.abc import Iterable

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


# =========================
# 基本参数
# =========================

LEFT_CAN = "can0"
RIGHT_CAN = "can1"

JOINT_NUMS = 7

CONNECT_WAIT = 0.5
ENABLE_WAIT = 1.0
READ_TIMEOUT = 5.0
READ_INTERVAL = 0.05

# =========================
# 抖动测试参数
# =========================

SHAKE_DELTA = 0.5       # rad，正负抖动幅度
SHAKE_CYCLES = 1        # 每个关节抖动次数
MOVE_WAIT = 2.0         # 每次运动后等待时间，太短容易 REACH_TARGET_POS_FAILED

# Nero 常见关节保护范围。实际限位如果不同，请按说明书修改。
JOINT_LIMITS = [
    (-2.96, 2.96),
    (-2.96, 2.96),
    (-2.96, 2.96),
    (-2.96, 2.96),
    (-2.96, 2.96),
    (-2.96, 2.96),
    (-2.96, 2.96),
]


# =========================
# 创建机械臂
# =========================

def create_nero_robot(channel: str):
    """
    创建 Nero 机械臂对象。
    注意：左右臂必须分别创建 cfg，不要复用同一个 cfg。
    """
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V111,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    return robot


# =========================
# 安全调用
# =========================

def safe_call(obj, func_name: str, default=None):
    try:
        func = getattr(obj, func_name)
        return func()
    except Exception:
        return default


def unwrap_msg(ret):
    """
    pyAgxArm 很多返回值是 MessageAbstract，真正数据在 .msg 里面。
    如果没有 .msg，就直接返回本体。
    """
    if ret is None:
        return None

    if hasattr(ret, "msg"):
        return ret.msg

    return ret


# =========================
# 解析关节角
# =========================

def extract_joint_angles(msg):
    """
    尽量兼容不同格式的关节角返回值。
    返回 list[float]，单位保持为 rad。
    """

    if msg is None:
        return None

    # 情况 1：直接是 list / tuple / numpy array 等可迭代数值
    if isinstance(msg, (list, tuple)) or (
        isinstance(msg, Iterable) and not isinstance(msg, (str, bytes, dict))
    ):
        return [float(x) for x in msg]

    # 情况 2：dict
    if isinstance(msg, dict):
        for key in [
            "joint_angles",
            "angles",
            "joint_angle",
            "angle",
            "data",
        ]:
            if key in msg:
                value = msg[key]
                if isinstance(value, (list, tuple)):
                    return [float(x) for x in value]

        # j1 ~ j7
        keys = [f"j{i}" for i in range(1, JOINT_NUMS + 1)]
        if all(k in msg for k in keys):
            return [float(msg[k]) for k in keys]

        # joint_1 ~ joint_7
        keys = [f"joint_{i}" for i in range(1, JOINT_NUMS + 1)]
        if all(k in msg for k in keys):
            return [float(msg[k]) for k in keys]

    # 情况 3：对象属性
    for attr in [
        "joint_angles",
        "angles",
        "joint_angle",
        "angle",
        "data",
    ]:
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]

    # 情况 4：对象属性 j1 ~ j7
    attrs = [f"j{i}" for i in range(1, JOINT_NUMS + 1)]
    if all(hasattr(msg, a) for a in attrs):
        return [float(getattr(msg, a)) for a in attrs]

    # 情况 5：对象属性 joint_1 ~ joint_7
    attrs = [f"joint_{i}" for i in range(1, JOINT_NUMS + 1)]
    if all(hasattr(msg, a) for a in attrs):
        return [float(getattr(msg, a)) for a in attrs]

    # 情况 6：打印未知结构，方便之后修正
    raise RuntimeError(f"unknown joint angle msg format: {msg}")


def get_joint_list(robot, name: str, timeout=READ_TIMEOUT, interval=READ_INTERVAL):
    """
    读取关节角。
    如果刚 enable 后反馈还没准备好，会等待一段时间。
    """

    start_time = time.time()
    last_error = None

    while time.time() - start_time < timeout:
        try:
            ret = robot.get_joint_angles()

            if ret is not None:
                msg = unwrap_msg(ret)
                joints = extract_joint_angles(msg)

                if joints is not None:
                    if len(joints) < JOINT_NUMS:
                        raise RuntimeError(
                            f"{name}: joint angle length error, "
                            f"expected >= {JOINT_NUMS}, got {len(joints)}: {joints}"
                        )

                    joints = joints[:JOINT_NUMS]
                    return joints

        except Exception as e:
            last_error = e

        time.sleep(interval)

    connected = safe_call(robot, "is_connected", "unknown")
    has_comm_error = safe_call(robot, "has_comm_error", "unknown")
    comm_error = safe_call(robot, "get_comm_error", "unknown")

    arm_status = safe_call(robot, "get_arm_status", None)
    arm_status_msg = unwrap_msg(arm_status)

    raise RuntimeError(
        f"{name}: get_joint_angles returned None for {timeout}s\n"
        f"{name}: connected={connected}\n"
        f"{name}: has_comm_error={has_comm_error}\n"
        f"{name}: comm_error={comm_error}\n"
        f"{name}: arm_status={arm_status_msg}\n"
        f"{name}: last_error={last_error}"
    )


# =========================
# 连接 / 使能 / 断开
# =========================

def connect_and_enable(robot, name: str, channel: str):
    print(f"{name}: connecting on {channel}...")

    ret = robot.connect()
    time.sleep(CONNECT_WAIT)

    connected = safe_call(robot, "is_connected", None)

    print(f"{name}: connect() ret = {ret}")
    print(f"{name}: connected = {connected}")

    if connected is False:
        raise RuntimeError(f"{name}: connect failed on {channel}")

    ret = robot.enable()
    time.sleep(ENABLE_WAIT)

    print(f"{name}: enable() ret = {ret}")

    # enable 后先尝试读一次状态
    arm_status = safe_call(robot, "get_arm_status", None)
    arm_status_msg = unwrap_msg(arm_status)
    print(f"{name}: arm_status = {arm_status_msg}")

    print(f"{name}: enabled")


def disable_and_disconnect(robot, name: str):
    if robot is None:
        return

    try:
        print(f"{name}: disabling...")
        robot.disable()
        time.sleep(0.2)
    except Exception as e:
        print(f"{name}: disable error: {e}")

    try:
        print(f"{name}: disconnecting...")
        robot.disconnect()
        time.sleep(0.2)
    except Exception as e:
        print(f"{name}: disconnect error: {e}")


# =========================
# 打印关节
# =========================

def print_joint_list(name: str, joints):
    print(f"{name}: joint angles rad = [")
    for i, value in enumerate(joints, start=1):
        print(f"  j{i}: {value:.6f}")
    print("]")


# =========================
# 运动相关
# =========================

def get_move_j_func(robot):
    """
    兼容 SDK 里可能的 move_j / movej / move_joint 命名。
    你的原代码注释里写的是 move_j，所以优先使用 move_j。
    """
    for func_name in ["move_j", "movej", "move_joint"]:
        func = getattr(robot, func_name, None)
        if callable(func):
            return func, func_name

    raise RuntimeError("robot has no move_j / movej / move_joint method")


def check_joint_limits(name: str, target):
    """
    运动前检查目标角度是否超限。
    单位：rad。
    """
    if len(target) < JOINT_NUMS:
        raise RuntimeError(
            f"{name}: target length error, expected {JOINT_NUMS}, got {len(target)}"
        )

    for i, value in enumerate(target[:JOINT_NUMS], start=1):
        low, high = JOINT_LIMITS[i - 1]

        if value < low or value > high:
            raise RuntimeError(
                f"{name}: target joint limit error: "
                f"j{i}={value:.6f} rad, limit=[{low}, {high}]"
            )


def move_joint_target(robot, name: str, target, wait_time=MOVE_WAIT):
    """
    发送一次关节角目标。
    target 单位：rad。
    """
    target = [float(x) for x in target[:JOINT_NUMS]]

    check_joint_limits(name, target)

    move_func, move_name = get_move_j_func(robot)

    print(f"{name}: {move_name} target rad = [")
    for i, value in enumerate(target, start=1):
        print(f"  j{i}: {value:.6f}")
    print("]")

    ret = move_func(target)
    print(f"{name}: {move_name} ret = {ret}")

    time.sleep(wait_time)

    # 运动后读一次当前角度，确认是否有反馈
    try:
        now = get_joint_list(robot, name, timeout=2.0)
        print_joint_list(f"{name}_now", now)
    except Exception as e:
        print(f"{name}: read after move warning: {e}")

    return ret


def shake_one_arm(robot, name: str, start_joints):
    """
    每次只动一个关节：

    j1: +0.5 -> -0.5 -> 回初始
    j2: +0.5 -> -0.5 -> 回初始
    ...
    j7: +0.5 -> -0.5 -> 回初始

    单位：rad。
    """
    print(f"{name}: start shaking, delta = {SHAKE_DELTA} rad")

    home = [float(x) for x in start_joints[:JOINT_NUMS]]

    for joint_index in range(JOINT_NUMS):
        joint_no = joint_index + 1

        for cycle in range(SHAKE_CYCLES):
            print("")
            print(f"{name}: shaking j{joint_no}, cycle {cycle + 1}/{SHAKE_CYCLES}")

            target_plus = home.copy()
            target_plus[joint_index] += SHAKE_DELTA
            print(f"{name}: j{joint_no} +{SHAKE_DELTA} rad")
            move_joint_target(robot, name, target_plus)

            target_minus = home.copy()
            target_minus[joint_index] -= SHAKE_DELTA
            print(f"{name}: j{joint_no} -{SHAKE_DELTA} rad")
            move_joint_target(robot, name, target_minus)

            print(f"{name}: j{joint_no} back to start")
            move_joint_target(robot, name, home)

    print(f"{name}: shake finished")


# =========================
# 主程序
# =========================

def main():
    left_robot = None
    right_robot = None

    try:
        left_robot = create_nero_robot(LEFT_CAN)
        right_robot = create_nero_robot(RIGHT_CAN)

        connect_and_enable(left_robot, "left", LEFT_CAN)
        connect_and_enable(right_robot, "right", RIGHT_CAN)

        print("")
        print("reading initial joint angles...")

        left_start = get_joint_list(left_robot, "left", timeout=READ_TIMEOUT)
        right_start = get_joint_list(right_robot, "right", timeout=READ_TIMEOUT)

        print_joint_list("left_start", left_start)
        print_joint_list("right_start", right_start)

        print("")
        print("both arms are ready")

        # =========================
        # 抖动测试
        # =========================
        # 为了安全，先 left 再 right。
        # 每个臂每次只动一个关节，不让 7 个关节同时大幅动作。

        print("")
        print("start left arm shake test")
        shake_one_arm(left_robot, "left", left_start)

        print("")
        print("start right arm shake test")
        shake_one_arm(right_robot, "right", right_start)

        print("")
        print("both arms shake test finished")

    except KeyboardInterrupt:
        print("")
        print("KeyboardInterrupt")

    except Exception:
        print("")
        print("ERROR:")
        traceback.print_exc()

    finally:
        print("")
        disable_and_disconnect(left_robot, "left")
        disable_and_disconnect(right_robot, "right")
        print("disabled and disconnected")


if __name__ == "__main__":
    main()