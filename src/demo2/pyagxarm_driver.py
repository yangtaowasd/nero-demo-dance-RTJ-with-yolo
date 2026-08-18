#!/usr/bin/env python3
"""Safely bridge one Nero command topic to one pyAgxArm CAN device."""

import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger

from demo2.instance_guard import parent_process_changed

from demo2.arm_sides import validate_side
from demo2.nero_direction_ik import NeroKinematics


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
DEFAULT_SHUTDOWN_HOME_JOINTS = (
    0.0, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 0.0
)
FIRMWARE_NAMES = ("auto", "default", "v111", "v112", "v120")
ARM_STATUS_NAMES = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_solution",
    0x03: "singularity_point",
    0x04: "target_position_exceeds_limit",
    0x05: "joint_communication_error",
    0x06: "joint_brake_not_released",
    0x07: "collision_occurred",
    0x08: "overspeed_during_teaching_drag",
    0x09: "joint_status_error",
    0x0A: "other_error",
    0x0B: "teaching_record",
    0x0C: "teaching_execution",
    0x0D: "teaching_pause",
    0x0E: "main_controller_over_temperature",
    0x0F: "release_resistor_over_temperature",
}


def default_urdf_path():
    """Return the installed Nero URDF, with a source-tree fallback."""
    try:
        share = Path(get_package_share_directory("demo2"))
    except Exception:
        share = Path(__file__).resolve().parents[2]
    return str(share / "urdf/nero_description.urdf")


def joint_values(value, count=7):
    """Return a finite joint vector from a pyAgxArm feedback wrapper."""
    if value is None:
        return None
    raw = getattr(value, "msg", value)
    try:
        joints = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if joints.shape != (int(count),) or not np.all(np.isfinite(joints)):
        return None
    return joints.copy()


def firmware_name_from_version(version):
    """Map a Nero software version to the driver named by the official API."""
    match = re.search(r"(\d+)\.(\d+)", str(version))
    if match is None:
        raise ValueError(f"invalid Nero firmware version: {version!r}")
    numeric = (int(match.group(1)), int(match.group(2)))
    if numeric >= (1, 20):
        return "v120"
    if numeric == (1, 12):
        return "v112"
    if numeric == (1, 11):
        return "v111"
    if numeric <= (1, 10):
        return "default"
    raise ValueError(
        f"Nero firmware {version!r} is not covered by the official "
        "driver compatibility table"
    )


def firmware_name_from_info(info):
    """Extract and map the SDK get_firmware() result."""
    if not isinstance(info, dict) or not info.get("software_version"):
        raise ValueError("get_firmware() returned no software_version")
    return firmware_name_from_version(info["software_version"])


def arm_status_code(value):
    """Extract the official arm_status field from a feedback wrapper."""
    if value is None:
        return None
    message = getattr(value, "msg", value)
    raw = getattr(message, "arm_status", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def joint_enable_values(value, count=7):
    """Normalize the official seven-joint enable-status response."""
    try:
        enabled = list(value)
    except (TypeError, ValueError):
        return None
    if len(enabled) != int(count):
        return None
    return [bool(item) for item in enabled]


def checked_joint_target(values, limits):
    """Validate one seven-axis target against finite URDF limits."""
    target = np.asarray(values, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if target.shape != (7,):
        raise ValueError("joint command must contain seven values")
    if limits.shape != (7, 2):
        raise ValueError("joint limits must have shape 7x2")
    if not np.all(np.isfinite(target)):
        raise ValueError("joint command contains NaN or infinity")
    if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
        raise ValueError("joint command exceeds Nero URDF limits")
    return target.copy()


def bounded_joint_step(current, target, dt, max_speed_rad_sec):
    """Rate-limit a target independently on all seven joints."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError("current and target joint shapes differ")
    maximum_step = max(float(dt), 0.0) * max(
        float(max_speed_rad_sec), 0.0
    )
    return current + np.clip(
        target - current, -maximum_step, maximum_step
    )


def command_is_fresh(command_time, now, timeout):
    """Return whether a command timestamp is still inside its watchdog."""
    return bool(
        command_time is not None
        and float(now) - float(command_time) <= max(float(timeout), 0.0)
    )


def feedback_requires_enable(firmware):
    """Return whether this Nero firmware starts CAN push after enable."""
    return str(firmware).strip().lower() in ("default", "v111")


def temporary_enable_probe(
    arm,
    timeout,
    *,
    poll_period=0.05,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Temporarily enable once, read startup state, then always disconnect."""
    timeout = max(float(timeout), 0.01)
    poll_period = max(float(poll_period), 0.0)
    connect_attempted = False
    enable_attempted = False
    operation_error = None
    snapshot = None
    try:
        connect_attempted = True
        arm.connect()
        enable_attempted = True
        enable_result = bool(arm.enable())
        deadline = monotonic() + timeout
        firmware_info = None
        status = None
        enabled = None
        measured = None
        while True:
            remaining = max(deadline - monotonic(), 0.0)
            if firmware_info is None and remaining > 0.0:
                firmware_info = arm.get_firmware(
                    timeout=min(1.0, remaining), min_interval=0.0
                )
            status = arm_status_code(arm.get_arm_status())
            enabled = joint_enable_values(
                arm.get_joints_enable_status_list()
            )
            measured = joint_values(arm.get_joint_angles())
            if (
                firmware_info is not None
                and status is not None
                and enabled is not None
                and measured is not None
            ):
                snapshot = {
                    "enable_request_accepted": enable_result,
                    "firmware_info": dict(firmware_info),
                    "arm_status": status,
                    "joints_enabled": enabled,
                    "joint_positions": measured.tolist(),
                }
                break
            if monotonic() >= deadline:
                missing = []
                if firmware_info is None:
                    missing.append("firmware")
                if status is None:
                    missing.append("arm status")
                if enabled is None:
                    missing.append("seven-joint enable state")
                if measured is None:
                    missing.append("seven-joint position")
                raise RuntimeError(
                    "temporary enable probe timed out waiting for "
                    + ", ".join(missing)
                )
            sleep(min(poll_period, max(deadline - monotonic(), 0.0)))
    except Exception as exc:
        operation_error = exc

    cleanup_errors = []
    if enable_attempted:
        try:
            disable_confirmed = bool(arm.disable())
            disable_deadline = monotonic() + timeout
            while not disable_confirmed and monotonic() < disable_deadline:
                enabled = joint_enable_values(
                    arm.get_joints_enable_status_list()
                )
                disable_confirmed = (
                    enabled is not None and not any(enabled)
                )
                if not disable_confirmed:
                    sleep(min(
                        poll_period,
                        max(disable_deadline - monotonic(), 0.0),
                    ))
            if not disable_confirmed:
                try:
                    arm.electronic_emergency_stop()
                except Exception:
                    pass
                cleanup_errors.append(
                    "temporary enable could not confirm motor disable"
                )
            elif snapshot is not None:
                snapshot["disable_confirmed"] = True
        except Exception as exc:
            try:
                arm.electronic_emergency_stop()
            except Exception:
                pass
            cleanup_errors.append(f"disable failed: {exc}")
    if connect_attempted:
        try:
            arm.disconnect()
        except Exception as exc:
            cleanup_errors.append(f"disconnect failed: {exc}")
    if operation_error is not None:
        raise operation_error
    if cleanup_errors:
        raise RuntimeError("; ".join(cleanup_errors))
    return snapshot


def wait_for_enabled_hardware(
    arm,
    timeout,
    *,
    poll_period=0.05,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Wait for positions, seven enabled joints, and NORMAL arm status."""
    deadline = monotonic() + max(float(timeout), 0.01)
    measured = None
    enabled = None
    status = None
    while True:
        measured = joint_values(arm.get_joint_angles())
        enabled = joint_enable_values(arm.get_joints_enable_status_list())
        status = arm_status_code(arm.get_arm_status())
        if (
            measured is not None
            and enabled is not None
            and all(enabled)
            and status == 0x00
        ):
            return measured, enabled, status
        if monotonic() >= deadline:
            status_name = ARM_STATUS_NAMES.get(status, "missing")
            enabled_count = sum(enabled) if enabled is not None else 0
            raise RuntimeError(
                "hardware readiness timed out: "
                f"arm_status={status_name}, joints_enabled="
                f"{enabled_count}/7, joint_positions="
                f"{'ready' if measured is not None else 'missing'}"
            )
        sleep(min(max(float(poll_period), 0.0), max(
            deadline - monotonic(), 0.0
        )))


def reset_emergency_stop_and_wait(
    arm,
    timeout,
    *,
    poll_period=0.05,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Reset one electronic stop and confirm status leaves EMERGENCY_STOP."""
    arm.reset()
    deadline = monotonic() + max(float(timeout), 0.01)
    while True:
        status = arm_status_code(arm.get_arm_status())
        if status is not None and status != 0x01:
            return status
        if monotonic() >= deadline:
            raise RuntimeError("electronic emergency-stop reset timed out")
        sleep(min(max(float(poll_period), 0.0), max(
            deadline - monotonic(), 0.0
        )))


def automatic_enable_ready(
    auto_enable, execute_motion, connected, enabled, estopped, consumed
):
    """Return whether the one-shot startup enable may run."""
    return bool(
        auto_enable
        and execute_motion
        and connected
        and not enabled
        and not estopped
        and not consumed
    )


def joint_target_reached(measured, target, tolerance_rad):
    """Return whether all joints are within the shutdown tolerance."""
    measured = np.asarray(measured, dtype=float)
    target = np.asarray(target, dtype=float)
    if measured.shape != (7,) or target.shape != (7,):
        return False
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
        return False
    return bool(
        np.max(np.abs(measured - target))
        <= max(float(tolerance_rad), 0.0)
    )


def motion_delay_remaining(enabled_time, now, delay_sec):
    """Return seconds before physical vision commands may be accepted."""
    if enabled_time is None:
        return 0.0
    return max(
        float(enabled_time) + max(float(delay_sec), 0.0) - float(now),
        0.0,
    )


def acquire_can_lock(can_interface):
    """Give one process exclusive command ownership of one CAN interface."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(can_interface))
    path = f"/tmp/demo2-pyagxarm-{safe_name}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(
            f"another process already owns CAN interface {can_interface}"
        ) from None
    return descriptor


class PyAgxArmDriver(Node):
    """Own one Nero CAN interface and enforce feedback/command watchdogs."""

    def __init__(self):
        """Configure one independently gated left or right hardware arm."""
        super().__init__("nero_pyagxarm_driver")
        defaults = {
            "side": "left",
            "can_interface": "can0",
            "firmware": "auto",
            "urdf_file": default_urdf_path(),
            "command_topic": "/left/neroarm/command_joints",
            "feedback_topic": "/left/neroarm/measured_joint_states",
            "status_topic": "/left/neroarm/hardware_status",
            "execute_motion": False,
            "auto_enable": True,
            "reset_emergency_stop_on_start": True,
            "emergency_reset_timeout_sec": 5.0,
            "require_command_before_enable": False,
            "motion_start_delay_sec": 5.0,
            "return_to_home_on_shutdown": True,
            "shutdown_home_positions": list(DEFAULT_SHUTDOWN_HOME_JOINTS),
            "shutdown_return_timeout_sec": 8.0,
            "shutdown_position_tolerance_deg": 1.5,
            "disable_on_shutdown": False,
            "feedback_rate_hz": 20.0,
            "command_timeout_sec": 0.35,
            "feedback_timeout_sec": 0.50,
            "connect_timeout_sec": 2.0,
            "probe_reconnect_delay_sec": 0.5,
            "reconnect_interval_sec": 2.0,
            "enable_timeout_sec": 5.0,
            "max_command_speed_deg_sec": 30.0,
            "speed_percent": 20,
            "exit_if_parent_changes": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def value(name):
            return self.get_parameter(name).value

        self.side = validate_side(value("side"))
        self.can_interface = str(value("can_interface"))
        self.firmware = str(value("firmware")).strip().lower()
        if self.firmware not in FIRMWARE_NAMES:
            raise ValueError(
                f"firmware must be one of {', '.join(FIRMWARE_NAMES)}"
            )
        self.execute_motion = bool(value("execute_motion"))
        self.auto_enable = bool(value("auto_enable"))
        self.reset_emergency_stop_on_start = bool(
            value("reset_emergency_stop_on_start")
        )
        self.emergency_reset_timeout = max(
            float(value("emergency_reset_timeout_sec")), 0.1
        )
        self.require_command_before_enable = bool(
            value("require_command_before_enable")
        )
        self.motion_start_delay = max(
            float(value("motion_start_delay_sec")), 0.0
        )
        self.return_to_home_on_shutdown = bool(
            value("return_to_home_on_shutdown")
        )
        self.shutdown_home_positions = np.asarray(
            value("shutdown_home_positions"), dtype=float
        )
        self.shutdown_return_timeout = max(
            float(value("shutdown_return_timeout_sec")), 0.1
        )
        self.shutdown_position_tolerance = np.deg2rad(
            max(float(value("shutdown_position_tolerance_deg")), 0.1)
        )
        self.disable_on_shutdown = bool(value("disable_on_shutdown"))
        self.command_timeout = max(
            float(value("command_timeout_sec")), 0.05
        )
        self.feedback_timeout = max(
            float(value("feedback_timeout_sec")), 0.05
        )
        self.connect_timeout = max(
            float(value("connect_timeout_sec")), 0.1
        )
        self.probe_reconnect_delay = max(
            float(value("probe_reconnect_delay_sec")), 0.0
        )
        self.reconnect_interval = max(
            float(value("reconnect_interval_sec")), 0.25
        )
        self.enable_timeout = max(float(value("enable_timeout_sec")), 0.1)
        self.max_command_speed = np.deg2rad(
            max(float(value("max_command_speed_deg_sec")), 0.1)
        )
        self.speed_percent = int(np.clip(int(value("speed_percent")), 1, 100))
        self.exit_if_parent_changes = bool(value("exit_if_parent_changes"))
        self.launch_parent_pid = os.getppid()
        self.orphan_shutdown_requested = False
        rate = max(float(value("feedback_rate_hz")), 1.0)

        self.kinematics = NeroKinematics(str(value("urdf_file")))
        self.joint_limits = self.kinematics.limits
        self.shutdown_home_positions = checked_joint_target(
            self.shutdown_home_positions, self.joint_limits
        )
        self.can_lock = acquire_can_lock(self.can_interface)
        self.arm = None
        self.resolved_firmware = None
        self.firmware_info = None
        self.arm_status = None
        self.joints_enabled = None
        self.connected = False
        self.enabled = False
        self.estopped = False
        self.latest_command = None
        self.latest_command_time = None
        self.measured_joints = None
        self.measured_velocity = np.zeros(7)
        self.last_feedback_time = None
        self.last_feedback_sample_time = None
        self.last_sent_joints = None
        self.last_send_time = None
        self.motion_enable_time = None
        self.motion_delay_hold_commanded = False
        self.watchdog_holding = False
        self.motion_command_started = False
        self.auto_enable_consumed = False
        self.startup_probe_consumed = False
        self.startup_probe_complete = False
        self.startup_probe_failed = False
        self.startup_probe_snapshot = None
        self.startup_reset_consumed = False
        self.startup_reset_complete = False
        self.startup_reset_failed = False
        self.next_connect_time = 0.0
        self.last_status_time = 0.0
        self.last_status_state = None
        self.last_health_time = 0.0

        self.feedback_publisher = self.create_publisher(
            JointState, str(value("feedback_topic")), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(value("status_topic")), 10
        )
        self.create_subscription(
            Float32MultiArray,
            str(value("command_topic")),
            self.command_callback,
            10,
        )
        self.create_service(SetBool, "~/enable", self.enable_callback)
        self.create_service(Trigger, "~/estop", self.estop_callback)
        self.create_service(Trigger, "~/reset", self.reset_callback)
        self.create_timer(1.0 / rate, self.tick)
        self.publish_status("waiting_for_can")
        if not self.execute_motion:
            motion_state = "READ ONLY"
        elif self.auto_enable:
            motion_state = "TWO-STAGE AUTO ENABLE ON CONNECT"
        else:
            motion_state = "ARMED BY SERVICE"
        self.get_logger().info(
            f"{self.side} pyAgxArm driver owns {self.can_interface}; "
            f"firmware={self.firmware}; mode={motion_state}"
        )

    def create_arm(self, firmware=None):
        """Construct one pyAgxArm Nero driver for this CAN interface."""
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW
        from pyAgxArm import create_agx_arm_config

        firmwares = {
            "default": NeroFW.DEFAULT,
            "v111": NeroFW.V111,
            "v112": NeroFW.V112,
            "v120": NeroFW.V120,
        }
        selected = self.firmware if firmware is None else str(firmware)
        if selected == "auto":
            selected = "default"
        config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=firmwares[selected],
            interface="socketcan",
            channel=self.can_interface,
            bitrate=1_000_000,
            enable_check_can=True,
            auto_connect=True,
            timeout=1.0,
            receive_own_messages=False,
            local_loopback=False,
        )
        return AgxArmFactory.create_arm(config)

    def detect_firmware(self):
        """Query firmware with NeroFW.DEFAULT exactly as documented."""
        probe = self.create_arm("default")
        try:
            probe.connect()
            info = probe.get_firmware(timeout=self.connect_timeout)
            resolved = firmware_name_from_info(info)
            return resolved, dict(info)
        finally:
            try:
                probe.disconnect()
            except Exception:
                pass

    def run_startup_probe(self):
        """Perform stage 1/2 without opening the physical motion gate."""
        self.get_logger().info(
            f"{self.side} enable stage 1/2: temporary state probe on "
            f"{self.can_interface}"
        )
        probe = self.create_arm("default")
        snapshot = temporary_enable_probe(probe, self.connect_timeout)
        detected = firmware_name_from_info(snapshot["firmware_info"])
        if self.firmware not in ("auto", detected):
            raise RuntimeError(
                f"configured firmware {self.firmware} does not match "
                f"detected firmware {detected}"
            )
        status = snapshot["arm_status"]
        self.startup_probe_snapshot = snapshot
        self.startup_probe_complete = True
        self.resolved_firmware = detected
        self.firmware_info = snapshot["firmware_info"]
        self.arm_status = status
        self.joints_enabled = snapshot["joints_enabled"]
        self.get_logger().info(
            f"{self.side} enable stage 1/2 complete and disabled: "
            f"firmware={detected}; arm_status="
            f"{ARM_STATUS_NAMES.get(status, status)}; joints_enabled="
            f"{sum(snapshot['joints_enabled'])}/7; joints="
            f"{np.round(snapshot['joint_positions'], 4).tolist()}"
        )
        self.publish_status("enable_stage_1_complete")
        if self.probe_reconnect_delay > 0.0:
            time.sleep(self.probe_reconnect_delay)
        return detected, snapshot["firmware_info"]

    @staticmethod
    def communication_error(arm):
        """Return the current official SDK communication error, if any."""
        if not arm.has_comm_error():
            return None
        error = arm.get_comm_error()
        return str(error) if error is not None else "unknown CAN error"

    def try_connect(self, now):
        """Run the one-shot probe, then establish the formal connection."""
        if self.connected or now < self.next_connect_time:
            return
        if self.startup_probe_failed:
            return
        self.next_connect_time = now + self.reconnect_interval
        arm = None
        try:
            if self.execute_motion and not self.startup_probe_complete:
                if self.startup_probe_consumed:
                    return
                self.startup_probe_consumed = True
                try:
                    resolved, info = self.run_startup_probe()
                except Exception as exc:
                    self.startup_probe_failed = True
                    self.get_logger().error(
                        f"{self.side} enable stage 1/2 failed; motion "
                        f"remains disabled until restart: {exc}"
                    )
                    self.publish_status("enable_stage_1_failed", str(exc))
                    return
                self.startup_probe_complete = True
            elif self.startup_probe_complete:
                resolved, info = (
                    self.resolved_firmware,
                    self.firmware_info,
                )
            elif self.firmware == "auto":
                resolved, info = self.detect_firmware()
            else:
                resolved, info = self.firmware, None
            arm = self.create_arm(resolved)
            arm.connect()
            deadline = time.monotonic() + self.connect_timeout
            feedback = None
            status = None
            while time.monotonic() < deadline:
                feedback = joint_values(arm.get_joint_angles())
                status = arm.get_arm_status()
                if feedback is not None:
                    break
                time.sleep(0.05)
            waits_for_enable = feedback_requires_enable(resolved)
            if feedback is None and not waits_for_enable:
                detail = self.communication_error(arm)
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"complete seven-joint feedback timed out{suffix}"
                )
            detail = self.communication_error(arm)
            if detail is not None:
                raise RuntimeError(f"pyAgxArm communication error: {detail}")
            formal_status = arm_status_code(status)
            probe_status = (
                self.startup_probe_snapshot or {}
            ).get("arm_status")
            needs_reset = 0x01 in (formal_status, probe_status)
            reset_failure = None
            if needs_reset:
                if not self.reset_emergency_stop_on_start:
                    reset_failure = (
                        "electronic emergency stop is active; automatic "
                        "startup reset is disabled"
                    )
                elif self.startup_reset_consumed:
                    reset_failure = (
                        "electronic emergency-stop reset was already "
                        "attempted"
                    )
                else:
                    self.startup_reset_consumed = True
                    self.get_logger().warning(
                        f"{self.side} startup preparation: resetting the "
                        "persisted electronic emergency stop"
                    )
                    try:
                        formal_status = reset_emergency_stop_and_wait(
                            arm, self.emergency_reset_timeout
                        )
                        self.startup_reset_complete = True
                        self.get_logger().info(
                            f"{self.side} startup electronic emergency "
                            "stop reset confirmed"
                        )
                    except Exception as exc:
                        self.startup_reset_failed = True
                        reset_failure = str(exc)
            self.arm = arm
            self.resolved_firmware = resolved
            self.firmware_info = info
            self.arm_status = formal_status
            self.joints_enabled = joint_enable_values(
                arm.get_joints_enable_status_list()
            )
            self.connected = True
            self.measured_joints = feedback
            self.measured_velocity[:] = 0.0
            connected_time = time.monotonic()
            self.last_feedback_time = (
                connected_time if feedback is not None else None
            )
            self.last_feedback_sample_time = self.last_feedback_time
            self.last_health_time = connected_time
            if self.arm_status == 0x01:
                self.estopped = True
            elif self.startup_reset_complete:
                self.estopped = False
            if reset_failure is not None:
                self.estopped = True
                self.get_logger().error(
                    f"{self.side} startup reset failed; formal enable "
                    f"remains blocked: {reset_failure}"
                )
                self.publish_status("startup_reset_failed", reset_failure)
                return
            if feedback is None:
                self.get_logger().info(
                    f"{self.side} enable stage 2/2 ready: Nero v1.11 "
                    f"transport connected on "
                    f"{self.can_interface}; joint feedback starts only "
                    "after the formal enable"
                )
                self.publish_status("enable_stage_2_ready")
            else:
                arm.set_joint_limits_enabled(True)
                self.get_logger().info(
                    f"{self.side} Nero connected read-only on "
                    f"{self.can_interface}; firmware={resolved}; "
                    f"joints={np.round(feedback, 4).tolist()}"
                )
                self.publish_status("connected_read_only")
        except Exception as exc:
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception:
                    pass
            self.arm = None
            self.connected = False
            self.get_logger().warning(
                f"{self.side} Nero connection failed on "
                f"{self.can_interface}: {exc}"
            )
            self.publish_status("connection_failed", str(exc))

    def return_to_home(self):
        """Return to [0, 90, 0, 0, 0, 0, 0] and stop on timeout."""
        if self.arm is None or not self.enabled:
            return False
        target = self.shutdown_home_positions.copy()
        self.get_logger().info(
            f"{self.side} Ctrl+C: returning to shutdown home pose "
            "[0, 90, 0, 0, 0, 0, 0] deg"
        )
        try:
            self.arm.set_speed_percent(self.speed_percent)
            self.arm.move_j(target.tolist())
            deadline = time.monotonic() + self.shutdown_return_timeout
            while time.monotonic() < deadline:
                measured = joint_values(self.arm.get_joint_angles())
                if measured is not None:
                    self.measured_joints = measured
                    if joint_target_reached(
                        measured,
                        target,
                        self.shutdown_position_tolerance,
                    ):
                        self.get_logger().info(
                            f"{self.side} shutdown home pose reached"
                        )
                        return True
                time.sleep(0.02)
        except Exception as exc:
            self.get_logger().error(
                f"{self.side} startup return failed: {exc}"
            )
        if self.measured_joints is not None:
            try:
                self.arm.move_j(self.measured_joints.tolist())
            except Exception:
                pass
        self.get_logger().error(
            f"{self.side} shutdown home return timed out; holding latest pose"
        )
        return False

    def disconnect(self):
        """Return/hold, optionally disable, and release the CAN transport."""
        arm = self.arm
        if arm is None:
            self.connected = False
            return
        if self.enabled and self.measured_joints is not None:
            returned = False
            if self.return_to_home_on_shutdown:
                returned = self.return_to_home()
            if not returned:
                try:
                    arm.move_j(self.measured_joints.tolist())
                except Exception:
                    pass
        if self.enabled and self.disable_on_shutdown:
            try:
                arm.disable()
            except Exception:
                pass
        try:
            arm.disconnect()
        except Exception:
            pass
        self.arm = None
        self.connected = False
        self.enabled = False
        self.motion_command_started = False

    def refresh_arm_health(self):
        """Read communication, controller status, and motor enable feedback."""
        detail = self.communication_error(self.arm)
        if detail is not None:
            raise RuntimeError(f"pyAgxArm communication error: {detail}")
        status = self.arm.get_arm_status()
        code = arm_status_code(status)
        if code is not None:
            self.arm_status = code
            if code == 0x01:
                self.estopped = True
                self.enabled = False
        enabled = joint_enable_values(
            self.arm.get_joints_enable_status_list()
        )
        if enabled is not None:
            self.joints_enabled = enabled
        self.last_health_time = time.monotonic()

    def require_normal_arm_status(self):
        """Reject motion unless current official status feedback is NORMAL."""
        self.refresh_arm_health()
        if self.arm_status is None:
            raise RuntimeError("arm status feedback is missing")
        if self.arm_status != 0x00:
            name = ARM_STATUS_NAMES.get(
                self.arm_status, f"unknown_0x{self.arm_status:02x}"
            )
            raise RuntimeError(f"arm status is not NORMAL: {name}")

    def command_callback(self, message):
        """Store only a finite, limit-safe, latest joint target."""
        try:
            target = checked_joint_target(message.data, self.joint_limits)
        except ValueError as exc:
            self.latest_command = None
            self.latest_command_time = None
            self.get_logger().error(
                f"{self.side} hardware command rejected: {exc}"
            )
            self.publish_status("command_rejected", str(exc))
            return
        self.latest_command = target
        self.latest_command_time = time.monotonic()

    def read_feedback(self, now):
        """Update and publish measured hardware joint state."""
        try:
            measured = joint_values(self.arm.get_joint_angles())
        except Exception as exc:
            measured = None
            self.get_logger().warning(
                f"{self.side} feedback read failed: {exc}",
                throttle_duration_sec=1.0,
            )
        if measured is None:
            if (
                self.last_feedback_time is not None
                and now - self.last_feedback_time > self.feedback_timeout
            ):
                self.feedback_lost()
            return False

        if self.measured_joints is not None:
            sample_time = self.last_feedback_sample_time or now
            dt = max(now - sample_time, 1e-3)
            raw_velocity = (measured - self.measured_joints) / dt
            self.measured_velocity += 0.25 * (
                raw_velocity - self.measured_velocity
            )
        self.measured_joints = measured
        self.last_feedback_time = now
        self.last_feedback_sample_time = now
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = measured.tolist()
        message.velocity = self.measured_velocity.tolist()
        self.feedback_publisher.publish(message)
        return True

    def feedback_lost(self):
        """Emergency-stop a moving arm when closed-loop feedback disappears."""
        if self.enabled and not self.estopped:
            try:
                self.arm.electronic_emergency_stop()
            except Exception:
                pass
            self.estopped = True
            self.enabled = False
        self.get_logger().error(
            f"{self.side} Nero feedback lost; command gate closed"
        )
        self.publish_status("feedback_lost")
        self.disconnect()
        self.next_connect_time = time.monotonic() + self.reconnect_interval

    def send_latest_command(self, now):
        """Send a rate-limited latest command or stop on watchdog expiry."""
        if not self.enabled or self.estopped or self.measured_joints is None:
            return
        delay_remaining = motion_delay_remaining(
            self.motion_enable_time, now, self.motion_start_delay
        )
        if delay_remaining > 0.0:
            self.last_sent_joints = self.measured_joints.copy()
            self.last_send_time = now
            if not self.motion_delay_hold_commanded:
                self.motion_delay_hold_commanded = True
                self.get_logger().info(
                    f"{self.side} gating all move_j commands for the "
                    f"{self.motion_start_delay:.1f}s startup motion delay"
                )
                self.publish_status("startup_motion_delay")
            return
        if self.motion_delay_hold_commanded:
            self.motion_delay_hold_commanded = False
            self.last_sent_joints = self.measured_joints.copy()
            self.last_send_time = now
            self.get_logger().info(
                f"{self.side} startup motion delay complete; "
                "physical vision commands are now accepted"
            )
            self.publish_status("motion_enabled")
        fresh = command_is_fresh(
            self.latest_command_time, now, self.command_timeout
        )
        if not fresh:
            if not self.motion_command_started:
                self.last_sent_joints = self.measured_joints.copy()
                self.last_send_time = now
            if not self.watchdog_holding:
                if self.motion_command_started:
                    self.arm.move_j(self.measured_joints.tolist())
                    self.last_sent_joints = self.measured_joints.copy()
                    self.last_send_time = now
                self.watchdog_holding = True
                self.get_logger().warning(
                    f"{self.side} command watchdog expired; "
                    + (
                        "holding measured pose"
                        if self.motion_command_started
                        else "no move_j trajectory has been sent"
                    )
                )
                self.publish_status("command_timeout_holding")
            return

        if self.watchdog_holding:
            self.last_sent_joints = self.measured_joints.copy()
            self.last_send_time = now
            self.watchdog_holding = False
        first_motion_command = not self.motion_command_started
        start = (
            self.last_sent_joints
            if self.last_sent_joints is not None
            else self.measured_joints
        )
        previous_time = self.last_send_time or now
        target = bounded_joint_step(
            start,
            self.latest_command,
            max(now - previous_time, 1e-3),
            self.max_command_speed,
        )
        target = np.clip(
            target, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        self.arm.move_j(target.tolist())
        self.last_sent_joints = target
        self.last_send_time = now
        self.motion_command_started = True
        if first_motion_command:
            requested_delta = float(np.max(np.abs(
                self.latest_command - self.measured_joints
            )))
            actual_step = float(np.max(np.abs(
                target - self.measured_joints
            )))
            self.get_logger().info(
                f"{self.side} first vision command ramped from measured "
                f"pose: requested_delta={np.rad2deg(requested_delta):.2f}deg, "
                f"first_step={np.rad2deg(actual_step):.3f}deg"
            )

    def enable_callback(self, request, response):
        """Explicitly enable or disable this arm after safety checks."""
        if not request.data:
            if self.arm is not None and self.enabled:
                try:
                    response.success = bool(self.arm.disable())
                except Exception as exc:
                    response.success = False
                    response.message = f"disable failed: {exc}"
                    return response
            else:
                response.success = True
            self.enabled = False
            self.motion_enable_time = None
            self.motion_delay_hold_commanded = False
            self.motion_command_started = False
            response.message = f"{self.side} Nero disabled"
            state = "emergency_stopped" if self.estopped else "disabled"
            self.publish_status(state)
            return response

        if not self.execute_motion:
            response.success = False
            response.message = "execute_motion is false; driver is read-only"
            return response
        if not self.connected or self.arm is None:
            response.success = False
            response.message = f"{self.side} Nero is not connected"
            return response
        if self.estopped:
            response.success = False
            response.message = "electronic stop is active; call reset first"
            return response
        now = time.monotonic()
        waits_for_enable = feedback_requires_enable(self.resolved_firmware)
        if (
            not waits_for_enable
            and not command_is_fresh(
                self.last_feedback_time, now, self.feedback_timeout
            )
        ):
            response.success = False
            response.message = "fresh seven-joint hardware feedback is missing"
            return response
        if (
            self.require_command_before_enable
            and not command_is_fresh(
                self.latest_command_time, now, self.command_timeout
            )
        ):
            response.success = False
            response.message = "no fresh validated vision command"
            return response
        enable_attempted = False
        try:
            if not waits_for_enable:
                self.require_normal_arm_status()
            deadline = time.monotonic() + self.enable_timeout
            enable_attempted = True
            while time.monotonic() < deadline:
                if waits_for_enable:
                    self.arm.set_normal_mode()
                if self.arm.enable():
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("motor enable timed out")
            measured, enabled, status = wait_for_enabled_hardware(
                self.arm, self.enable_timeout
            )
            detail = self.communication_error(self.arm)
            if detail is not None:
                raise RuntimeError(f"pyAgxArm communication error: {detail}")
            self.joints_enabled = enabled
            self.arm_status = status
            self.arm.set_joint_limits_enabled(True)
            self.arm.set_speed_percent(self.speed_percent)
            feedback_time = time.monotonic()
            self.measured_joints = measured
            self.measured_velocity[:] = 0.0
            self.last_feedback_time = feedback_time
            self.last_feedback_sample_time = feedback_time
        except Exception as exc:
            if enable_attempted:
                try:
                    self.arm.electronic_emergency_stop()
                    self.estopped = True
                except Exception:
                    pass
                self.enabled = False
            response.success = False
            response.message = f"enable failed: {exc}"
            self.publish_status("enable_failed", str(exc))
            return response
        self.enabled = True
        self.estopped = False
        self.last_sent_joints = self.measured_joints.copy()
        self.motion_enable_time = time.monotonic()
        self.last_send_time = self.motion_enable_time
        self.motion_delay_hold_commanded = False
        self.watchdog_holding = False
        self.motion_command_started = False
        response.success = True
        response.message = (
            f"{self.side} Nero enable stage 2/2 complete; physical "
            "commands start after "
            f"{self.motion_start_delay:.1f}s"
        )
        initial_state = (
            "startup_motion_delay"
            if self.motion_start_delay > 0.0
            else "motion_enabled"
        )
        self.publish_status(initial_state)
        return response

    def try_automatic_enable(self):
        """Enable once after startup, then hold until commands are valid."""
        if not automatic_enable_ready(
            self.auto_enable,
            self.execute_motion,
            self.connected,
            self.enabled,
            self.estopped,
            self.auto_enable_consumed,
        ):
            return False
        self.auto_enable_consumed = True
        request = SetBool.Request()
        request.data = True
        response = self.enable_callback(request, SetBool.Response())
        if response.success:
            self.get_logger().info(
                f"{self.side} Nero enable stage 2/2 complete"
            )
            return True
        self.get_logger().error(
            f"{self.side} Nero startup auto-enable failed: "
            f"{response.message}"
        )
        return False

    def estop_callback(self, _request, response):
        """Trigger the pyAgxArm damped electronic emergency stop."""
        if self.arm is None or not self.connected:
            response.success = False
            response.message = f"{self.side} Nero is not connected"
            return response
        try:
            self.arm.electronic_emergency_stop()
        except Exception as exc:
            response.success = False
            response.message = f"emergency stop failed: {exc}"
            return response
        self.estopped = True
        self.enabled = False
        self.motion_enable_time = None
        self.motion_delay_hold_commanded = False
        self.motion_command_started = False
        response.success = True
        response.message = f"{self.side} Nero emergency-stopped"
        self.publish_status("emergency_stopped")
        return response

    def reset_callback(self, _request, response):
        """Reset an electronic stop without enabling motion."""
        if self.arm is None or not self.connected:
            response.success = False
            response.message = f"{self.side} Nero is not connected"
            return response
        if not self.estopped:
            response.success = False
            response.message = "reset is valid only after electronic stop"
            return response
        try:
            status = reset_emergency_stop_and_wait(
                self.arm, self.emergency_reset_timeout
            )
        except Exception as exc:
            response.success = False
            response.message = f"reset failed: {exc}"
            return response
        self.arm_status = status
        self.estopped = False
        self.startup_reset_complete = True
        self.startup_reset_failed = False
        self.enabled = False
        self.motion_enable_time = None
        self.motion_delay_hold_commanded = False
        self.motion_command_started = False
        response.success = True
        response.message = f"{self.side} Nero reset; motion remains disabled"
        self.publish_status("reset_motion_disabled")
        return response

    def status_payload(self, state, detail=""):
        """Return a machine-readable snapshot of the hardware gate."""
        now = time.monotonic()
        delay_remaining = motion_delay_remaining(
            self.motion_enable_time, now, self.motion_start_delay
        )
        return {
            "side": self.side,
            "can_interface": self.can_interface,
            "firmware_requested": self.firmware,
            "firmware": self.resolved_firmware,
            "firmware_info": self.firmware_info,
            "arm_status": self.arm_status,
            "arm_status_name": ARM_STATUS_NAMES.get(
                self.arm_status, "missing"
            ),
            "joints_enabled": self.joints_enabled,
            "state": str(state),
            "detail": str(detail),
            "connected": self.connected,
            "execute_motion": self.execute_motion,
            "auto_enable": self.auto_enable,
            "auto_enable_consumed": self.auto_enable_consumed,
            "startup_probe_consumed": self.startup_probe_consumed,
            "startup_probe_complete": self.startup_probe_complete,
            "startup_probe_failed": self.startup_probe_failed,
            "startup_probe_snapshot": self.startup_probe_snapshot,
            "startup_reset_consumed": self.startup_reset_consumed,
            "startup_reset_complete": self.startup_reset_complete,
            "startup_reset_failed": self.startup_reset_failed,
            "enabled": self.enabled,
            "motion_delay_remaining_sec": delay_remaining,
            "estopped": self.estopped,
            "command_fresh": command_is_fresh(
                self.latest_command_time, now, self.command_timeout
            ),
            "feedback_fresh": command_is_fresh(
                self.last_feedback_time, now, self.feedback_timeout
            ),
        }

    def publish_status(self, state, detail=""):
        """Publish one status immediately and remember its state."""
        payload = self.status_payload(state, detail)
        self.status_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )
        self.last_status_state = str(state)
        self.last_status_time = time.monotonic()

    def tick(self):
        """Reconnect, read feedback, apply watchdogs, and drive one arm."""
        if (
            self.exit_if_parent_changes
            and not self.orphan_shutdown_requested
            and parent_process_changed(self.launch_parent_pid)
        ):
            self.orphan_shutdown_requested = True
            self.get_logger().error(
                f"{self.side} parent launch exited; returning safely and "
                f"releasing {self.can_interface}"
            )
            if rclpy.ok():
                rclpy.shutdown()
            return
        now = time.monotonic()
        if not self.connected:
            self.try_connect(now)
            return
        try:
            self.try_automatic_enable()
            now = time.monotonic()
            feedback_valid = self.read_feedback(now)
            if feedback_valid:
                self.send_latest_command(now)
            if now - self.last_health_time >= 1.0:
                self.refresh_arm_health()
            if now - self.last_status_time >= 1.0:
                if self.estopped:
                    state = "emergency_stopped"
                elif (
                    self.enabled
                    and motion_delay_remaining(
                        self.motion_enable_time,
                        now,
                        self.motion_start_delay,
                    ) > 0.0
                ):
                    state = "startup_motion_delay"
                elif self.enabled and self.watchdog_holding:
                    state = "command_timeout_holding"
                elif self.enabled:
                    state = "motion_enabled"
                elif self.measured_joints is None:
                    state = "connected_waiting_for_enable"
                else:
                    state = "connected_read_only"
                self.publish_status(state)
        except Exception as exc:
            self.get_logger().error(
                f"{self.side} pyAgxArm runtime failure: {exc}"
            )
            self.publish_status("runtime_failure", str(exc))
            self.disconnect()
            self.next_connect_time = now + self.reconnect_interval

    def close(self):
        """Release hardware and process-local command ownership."""
        self.disconnect()
        if self.can_lock is not None:
            os.close(self.can_lock)
            self.can_lock = None


def main(args=None):
    """Run one independently namespaced pyAgxArm hardware driver."""
    node = None
    try:
        rclpy.init(args=args)
        node = PyAgxArmDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        print(f"pyAgxArm driver stopped: {exc}", file=sys.stderr)
    finally:
        if node is not None:
            try:
                node.close()
            except (KeyboardInterrupt, Exception):
                pass
            try:
                node.destroy_node()
            except (KeyboardInterrupt, Exception):
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, Exception):
                pass


if __name__ == "__main__":
    main()
