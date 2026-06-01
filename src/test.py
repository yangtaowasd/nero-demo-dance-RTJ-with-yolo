import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V111, channel="can0")
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
robot.connect()

time.sleep(0.5)
print("effector is_ok =", end_effector.is_ok())