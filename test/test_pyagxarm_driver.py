"""Tests for the safety boundary around the pyAgxArm hardware bridge."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from demo2.pyagxarm_driver import (
    acquire_can_lock,
    arm_status_code,
    bounded_joint_step,
    checked_joint_target,
    command_is_fresh,
    firmware_name_from_info,
    firmware_name_from_version,
    joint_enable_values,
    joint_values,
)


LIMITS = np.asarray([[-1.0, 1.0]] * 7)


def test_feedback_requires_seven_finite_joint_values():
    """Incomplete or non-finite CAN feedback never reaches control."""
    wrapped = SimpleNamespace(msg=np.arange(7, dtype=float))

    np.testing.assert_allclose(joint_values(wrapped), np.arange(7))
    assert joint_values(SimpleNamespace(msg=[0.0] * 6)) is None
    assert joint_values(SimpleNamespace(msg=[0.0] * 6 + [np.nan])) is None


def test_joint_target_enforces_shape_finiteness_and_urdf_limits():
    """Only complete, finite, in-limit commands pass the hardware gate."""
    np.testing.assert_allclose(checked_joint_target([0.0] * 7, LIMITS), 0.0)

    with pytest.raises(ValueError, match="seven"):
        checked_joint_target([0.0] * 6, LIMITS)
    with pytest.raises(ValueError, match="NaN"):
        checked_joint_target([0.0] * 6 + [np.nan], LIMITS)
    with pytest.raises(ValueError, match="limits"):
        checked_joint_target([0.0] * 6 + [1.1], LIMITS)


def test_joint_step_cannot_exceed_configured_hardware_speed():
    """A large vision target becomes a bounded incremental command."""
    stepped = bounded_joint_step(
        np.zeros(7), np.ones(7), 0.05, np.deg2rad(30.0)
    )

    np.testing.assert_allclose(stepped, np.deg2rad(1.5))


def test_command_watchdog_rejects_missing_and_stale_targets():
    """Motion requires a recently received controller command."""
    assert not command_is_fresh(None, 10.0, 0.35)
    assert command_is_fresh(9.8, 10.0, 0.35)
    assert not command_is_fresh(9.0, 10.0, 0.35)


def test_one_process_exclusively_owns_each_can_interface():
    """Two ROS drivers cannot publish competing commands on one CAN bus."""
    interface = f"test-can-{os.getpid()}"
    first = acquire_can_lock(interface)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            acquire_can_lock(interface)
    finally:
        os.close(first)


@pytest.mark.parametrize(
    ("version", "driver"),
    [
        ("1.09", "default"),
        ("1.10", "default"),
        ("1.11", "v111"),
        ("1.12", "v112"),
        ("1.20", "v120"),
        ("Nero 2.01", "v120"),
    ],
)
def test_firmware_mapping_matches_official_nero_table(version, driver):
    """Auto discovery selects the exact driver family in the Nero API."""
    assert firmware_name_from_version(version) == driver
    assert firmware_name_from_info({"software_version": version}) == driver


def test_firmware_mapping_rejects_missing_and_undocumented_versions():
    """An unknown firmware is never guessed before commanding hardware."""
    with pytest.raises(ValueError, match="software_version"):
        firmware_name_from_info(None)
    with pytest.raises(ValueError, match="compatibility table"):
        firmware_name_from_version("1.15")


def test_official_arm_status_and_joint_enable_feedback_are_normalized():
    """Hardware enable gates consume the documented feedback shapes."""
    status = SimpleNamespace(msg=SimpleNamespace(arm_status=0x05))
    assert arm_status_code(status) == 0x05
    assert arm_status_code(None) is None
    assert joint_enable_values([1] * 7) == [True] * 7
    assert joint_enable_values([True] * 6) is None
