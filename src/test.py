#!/usr/bin/env python3

import argparse
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


def get_joint_list(robot):
    result = robot.get_joint_angles()
    joints = getattr(result, "msg", result)
    if joints is None:
        raise RuntimeError("get_joint_angles returned None")
    joints = list(joints)
    if len(joints) != 7:
        raise RuntimeError(f"expected 7 joints, got {len(joints)}: {joints}")
    return joints


def move_and_wait(robot, joints, wait_sec, label):
    print(label)
    print("move_j:", [round(v, 4) for v in joints])
    robot.move_j(joints)
    time.sleep(wait_sec)


def main():
    parser = argparse.ArgumentParser(description="Small Nero arm movement test.")
    parser.add_argument("--channel", default="can0", help="CAN interface, for example can0")
    parser.add_argument("--delta", type=float, default=0.1, help="Move delta in radians")
    parser.add_argument("--wait", type=float, default=1.0, help="Seconds to wait after each move")
    args = parser.parse_args()

    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V111,
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)

    print(f"connecting on {args.channel}...")
    robot.connect()
    time.sleep(0.5)

    connected = robot.is_connected()
    print("connected:", connected)
    if not connected:
        raise RuntimeError("arm is not connected")

    robot.enable()
    time.sleep(0.5)
    print("enabled")

    start = get_joint_list(robot)
    print("start:", [round(v, 4) for v in start])

    try:
        for idx in range(7):
            joint_id = idx + 1

            target_plus = start.copy()
            target_plus[idx] += args.delta
            move_and_wait(
                robot,
                target_plus,
                args.wait,
                f"J{joint_id} +{args.delta:.4f} rad"
            )

            target_minus = start.copy()
            target_minus[idx] -= args.delta
            move_and_wait(
                robot,
                target_minus,
                args.wait,
                f"J{joint_id} -{args.delta:.4f} rad"
            )

            move_and_wait(
                robot,
                start,
                args.wait,
                f"J{joint_id} back to start"
            )
    finally:
        robot.disable()
        time.sleep(0.2)
        robot.disconnect()
        print("disabled and disconnected")


if __name__ == "__main__":
    main()
