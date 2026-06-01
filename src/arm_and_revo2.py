#!/usr/bin/env python3

import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW, PiperFW


class Nero:
    def __init__(self,channel="can0"):
        # 1. 创建 Nero 配置
        self.cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.V111,
            channel=channel
            )
        
        self.robot = AgxArmFactory.create_arm(self.cfg)
        
         # ========== 基本 ==========

    def connect(self):
        self.robot.connect()
        time.sleep(0.5)
        print(self.robot.is_connected())

    def disconnect(self):
        self.robot.disconnect()
        print(self.robot.is_connected())

    def has_comm_error(self):
        return self.robot.has_comm_error()

    def get_comm_error(self):
        return self.robot.get_comm_error()

    def init_effector(self, effector):
        return self.robot.init_effector(effector)

    def get_joint_nums(self):
        return self.robot.joint_nums

    # ========== 读取 ==========

    def get_arm_status(self):
        return self.robot.get_arm_status()

    def get_joint_angles(self):
        return self.robot.get_joint_angles()

    def get_flange_pose(self):
        return self.robot.get_flange_pose()

    def get_motor_states(self, joint_index):
        return self.robot.get_motor_states(joint_index)

    def get_driver_states(self, joint_index):
        return self.robot.get_driver_states(joint_index)

    def get_joint_enable_status(self, joint_index=64):
        return self.robot.get_joint_enable_status(joint_index)

    def get_joints_enable_status_list(self):
        return self.robot.get_joints_enable_status_list()

    def get_firmware(self):
        return self.robot.get_firmware()

    def get_tcp_pose(self):
        return self.robot.get_tcp_pose()

    def get_leader_joint_angles(self):
        return self.robot.get_leader_joint_angles()

    # ========== 设置 ==========

    def set_speed_percent(self, percent=100):
        return self.robot.set_speed_percent(percent)

    def set_motion_mode(self, motion_mode="p"):
        return self.robot.set_motion_mode(motion_mode)

    def set_tcp_offset(self, pose):
        return self.robot.set_tcp_offset(pose)

    def set_auto_set_motion_mode_enabled(self, enabled):
        return self.robot.set_auto_set_motion_mode_enabled(enabled)

    def get_auto_set_motion_mode_enabled(self):
        return self.robot.get_auto_set_motion_mode_enabled()

    def set_joint_limits_enabled(self, enabled):
        return self.robot.set_joint_limits_enabled(enabled)

    def get_joint_limits_enabled(self):
        return self.robot.get_joint_limits_enabled()

    # ========== 坐标 / 运动学 ==========

    def get_flange2tcp_pose(self, flange_pose):
        return self.robot.get_flange2tcp_pose(flange_pose)

    def get_tcp2flange_pose(self, tcp_pose):
        return self.robot.get_tcp2flange_pose(tcp_pose)

    def fk(self, joint_angles):
        return self.robot.fk(joint_angles)

    # ========== 模式 ==========

    def set_normal_mode(self):
        return self.robot.set_normal_mode()

    def set_leader_mode(self):
        return self.robot.set_leader_mode()

    def set_follower_mode(self):
        return self.robot.set_follower_mode()

    # ========== 使能 / 急停 ==========

    def enable(self, joint_index=255):
        return self.robot.enable(joint_index)

    def disable(self, joint_index=255):
        return self.robot.disable(joint_index)

    def electronic_emergency_stop(self):
        return self.robot.electronic_emergency_stop()

    def reset(self):
        return self.robot.reset()

    # ========== 运动控制 ==========

    def move_j(self, joints):
        return self.robot.move_j(joints)

    def move_js(self, joints):
        return self.robot.move_js(joints)

    def move_p(self, pose):
        return self.robot.move_p(pose)

    def move_l(self, pose):
        return self.robot.move_l(pose)

    def move_c(self, start_pose, mid_pose, end_pose):
        return self.robot.move_c(start_pose, mid_pose, end_pose)

    def move_mit(self, joint_index, p_des=0.0, v_des=0.0, kp=10.0, kd=0.8, t_ff=0.0):
        return self.robot.move_mit(joint_index, p_des, v_des, kp, kd, t_ff)

    # ========== CPV ==========

    def move_cpv_pos(self, joint_index, pos):
        return self.robot.move_cpv_pos(joint_index, pos)

    def move_cpv_vel(self, joint_index, vel):
        return self.robot.move_cpv_vel(joint_index, vel)

    def get_cpv_pos(self, joint_index):
        return self.robot.get_cpv_pos(joint_index)

    def get_cpv_vel(self, joint_index):
        return self.robot.get_cpv_vel(joint_index)

    def get_cpv_acc(self, joint_index):
        return self.robot.get_cpv_acc(joint_index)

    def get_cpv_dcc(self, joint_index):
        return self.robot.get_cpv_dcc(joint_index)

    def get_cpv_cv(self, joint_index):
        return self.robot.get_cpv_cv(joint_index)

    def get_cpv_pp(self, joint_index):
        return self.robot.get_cpv_pp(joint_index)

    def get_cpv_kp(self, joint_index):
        return self.robot.get_cpv_kp(joint_index)

    def get_cpv_ki(self, joint_index):
        return self.robot.get_cpv_ki(joint_index)

    def set_cpv_acc(self, joint_index, acc):
        return self.robot.set_cpv_acc(joint_index, acc)

    def set_cpv_dcc(self, joint_index, dcc):
        return self.robot.set_cpv_dcc(joint_index, dcc)

    def set_cpv_cv(self, joint_index, cv):
        return self.robot.set_cpv_cv(joint_index, cv)

    def set_cpv_pp(self, joint_index, pp):
        return self.robot.set_cpv_pp(joint_index, pp)

    def set_cpv_kp(self, joint_index, kp):
        return self.robot.set_cpv_kp(joint_index, kp)

    def set_cpv_ki(self, joint_index, ki):
        return self.robot.set_cpv_ki(joint_index, ki)

    # ========== 高级参数 ==========

    def get_joint_angle_vel_limits(self, joint_index):
        return self.robot.get_joint_angle_vel_limits(joint_index)

    def get_joint_acc_limits(self, joint_index):
        return self.robot.get_joint_acc_limits(joint_index)

    def get_flange_vel_acc_limits(self):
        return self.robot.get_flange_vel_acc_limits()

    def get_crash_protection_rating(self):
        return self.robot.get_crash_protection_rating()

    def calibrate_joint(self, joint_index=255):
        return self.robot.calibrate_joint(joint_index)

    def clear_joint_error(self, joint_index=255):
        return self.robot.clear_joint_error(joint_index)

    def set_joint_angle_vel_limits(
        self,
        joint_index=255,
        min_angle_limit=None,
        max_angle_limit=None,
        max_joint_spd=None
    ):
        return self.robot.set_joint_angle_vel_limits(
            joint_index,
            min_angle_limit,
            max_angle_limit,
            max_joint_spd
        )

    def set_joint_acc_limits(self, joint_index=255, max_joint_acc=None):
        return self.robot.set_joint_acc_limits(joint_index, max_joint_acc)

    def set_flange_vel_acc_limits(
        self,
        max_linear_vel=None,
        max_angular_vel=None,
        max_linear_acc=None,
        max_angular_acc=None
    ):
        return self.robot.set_flange_vel_acc_limits(
            max_linear_vel,
            max_angular_vel,
            max_linear_acc,
            max_angular_acc
        )

    def set_crash_protection_rating(self, joint_index=255, rating=0):
        return self.robot.set_crash_protection_rating(joint_index, rating)

    # ========== 全部读取 ==========

    def read_all(self):
        return {
            "is_connected": self.robot.is_connected(),
            "has_comm_error": self.has_comm_error(),
            "comm_error": self.get_comm_error(),
            "firmware": self.get_firmware(),
            "joint_nums": self.get_joint_nums(),

            "arm_status": self.get_arm_status(),
            "joint_angles": self.get_joint_angles(),
            "flange_pose": self.get_flange_pose(),
            "tcp_pose": self.get_tcp_pose(),
            "joints_enable": self.get_joints_enable_status_list(),

            "auto_motion_mode": self.get_auto_set_motion_mode_enabled(),
            "joint_limits_enabled": self.get_joint_limits_enabled(),
            "leader_joint_angles": self.get_leader_joint_angles(),

            "flange_vel_acc_limits": self.get_flange_vel_acc_limits(),
            "crash_protection_rating": self.get_crash_protection_rating(),

            "joint_1_motor": self.get_motor_states(1),
            "joint_2_motor": self.get_motor_states(2),
            "joint_3_motor": self.get_motor_states(3),
            "joint_4_motor": self.get_motor_states(4),
            "joint_5_motor": self.get_motor_states(5),
            "joint_6_motor": self.get_motor_states(6),
            "joint_7_motor": self.get_motor_states(7),

            "joint_1_driver": self.get_driver_states(1),
            "joint_2_driver": self.get_driver_states(2),
            "joint_3_driver": self.get_driver_states(3),
            "joint_4_driver": self.get_driver_states(4),
            "joint_5_driver": self.get_driver_states(5),
            "joint_6_driver": self.get_driver_states(6),
            "joint_7_driver": self.get_driver_states(7),
        }