"""Process ownership guards shared by launch files and ROS nodes."""

import fcntl
import os


INSTANCE_LOCK = "/tmp/demo2-dual-arm-visualization.lock"


def acquire_instance_lock(lock_path=INSTANCE_LOCK):
    """Acquire the dual-arm controller lock and record its owner PID."""
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(
            "another dual-arm display/control pipeline is already running"
        ) from None
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def instance_owner_pid(lock_path=INSTANCE_LOCK):
    """Return the recorded lock owner PID when one is available."""
    try:
        with open(lock_path, encoding="ascii") as stream:
            value = stream.readline().strip()
    except (OSError, UnicodeError):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def require_instance_available(_context=None, lock_path=INSTANCE_LOCK):
    """Fail a launch before any camera or CAN process starts on conflict."""
    try:
        descriptor = acquire_instance_lock(lock_path)
    except RuntimeError as exc:
        owner = instance_owner_pid(lock_path)
        owner_text = f" (PID {owner})" if owner is not None else ""
        raise RuntimeError(
            f"{exc}{owner_text}; stop the old launch with Ctrl+C first"
        ) from None
    os.close(descriptor)
    return []


def parent_process_changed(initial_parent_pid, current_parent_pid=None):
    """Return whether a node was orphaned from the launch that started it."""
    current = os.getppid() if current_parent_pid is None else current_parent_pid
    return int(current) != int(initial_parent_pid)
