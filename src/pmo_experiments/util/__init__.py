"""Utility functions for pmo-experiments."""

import os
import socket


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    Returns:
        int: Number of CPU cores.

    """
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def get_free_port():
    """Get a free port on the localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
