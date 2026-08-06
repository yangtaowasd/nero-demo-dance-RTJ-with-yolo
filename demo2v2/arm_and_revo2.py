#!/usr/bin/env python3

import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


class Nero:
    def __init__(self, channel="can0"):
        self.cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.V111,
            channel=channel,
        )
        self.robot = AgxArmFactory.create_arm(self.cfg)

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def connect(self):
        self.robot.connect()
        time.sleep(0.5)
        print(self.robot.is_connected())

    def disconnect(self):
        self.robot.disconnect()
        print(self.robot.is_connected())

    def get_joint_nums(self):
        return self.robot.joint_nums

    def set_speed_percent(self, percent=100):
        return self.robot.set_speed_percent(percent)

    def set_motion_mode(self, motion_mode="p"):
        return self.robot.set_motion_mode(motion_mode)

    def get_joint_enable_status(self, joint_index=64):
        return self.robot.get_joint_enable_status(joint_index)

    def enable(self, joint_index=255):
        return self.robot.enable(joint_index)

    def disable(self, joint_index=255):
        return self.robot.disable(joint_index)

    def calibrate_joint(self, joint_index=255):
        return self.robot.calibrate_joint(joint_index)

    def clear_joint_error(self, joint_index=255):
        return self.robot.clear_joint_error(joint_index)

    def set_joint_angle_vel_limits(
        self,
        joint_index=255,
        min_angle_limit=None,
        max_angle_limit=None,
        max_joint_spd=None,
    ):
        return self.robot.set_joint_angle_vel_limits(
            joint_index,
            min_angle_limit,
            max_angle_limit,
            max_joint_spd,
        )

    def set_joint_acc_limits(self, joint_index=255, max_joint_acc=None):
        return self.robot.set_joint_acc_limits(joint_index, max_joint_acc)

    def set_flange_vel_acc_limits(
        self,
        max_linear_vel=None,
        max_angular_vel=None,
        max_linear_acc=None,
        max_angular_acc=None,
    ):
        return self.robot.set_flange_vel_acc_limits(
            max_linear_vel,
            max_angular_vel,
            max_linear_acc,
            max_angular_acc,
        )

    def set_crash_protection_rating(self, joint_index=255, rating=0):
        return self.robot.set_crash_protection_rating(joint_index, rating)

    def read_all(self):
        data = {
            "is_connected": self.robot.is_connected(),
            "has_comm_error": self.robot.has_comm_error(),
            "comm_error": self.robot.get_comm_error(),
            "firmware": self.robot.get_firmware(),
            "joint_nums": self.get_joint_nums(),
            "arm_status": self.robot.get_arm_status(),
            "joint_angles": self.robot.get_joint_angles(),
            "flange_pose": self.robot.get_flange_pose(),
            "tcp_pose": self.robot.get_tcp_pose(),
            "joints_enable": self.robot.get_joints_enable_status_list(),
            "auto_motion_mode": self.robot.get_auto_set_motion_mode_enabled(),
            "joint_limits_enabled": self.robot.get_joint_limits_enabled(),
            "leader_joint_angles": self.robot.get_leader_joint_angles(),
            "flange_vel_acc_limits": self.robot.get_flange_vel_acc_limits(),
            "crash_protection_rating": self.robot.get_crash_protection_rating(),
        }

        for joint_index in range(1, 8):
            data[f"joint_{joint_index}_motor"] = self.robot.get_motor_states(joint_index)
            data[f"joint_{joint_index}_driver"] = self.robot.get_driver_states(joint_index)

        return data
