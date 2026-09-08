"""Console debug logging and memory helpers for training scripts."""

from datetime import datetime
import os
import resource

import torch


def get_current_cpu_memory_mb():
    """Return current process RSS memory in MB.

    Returns:
        float: Current resident memory (MB) for this process.
    """
    rss_pages = 0
    with open('/proc/self/statm', 'r') as statm_file:
        parts = statm_file.read().strip().split()
        if len(parts) >= 2:
            rss_pages = int(parts[1])
    page_size = os.sysconf('SC_PAGE_SIZE')
    return float(rss_pages * page_size) / 1024.0 / 1024.0


def get_peak_cpu_memory_mb():
    """Return peak process RSS memory in MB.

    Returns:
        float: Peak resident memory (MB) observed by the process.
    """
    # On Linux, ru_maxrss is reported in KB.
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0


def get_current_gpu_memory_mb(device_):
    """Return currently allocated GPU memory in MB.

    Args:
        device_ (torch.device): Target CUDA device.

    Returns:
        float: Current allocated CUDA memory (MB), or 0.0 on CPU.
    """
    if device_.type != 'cuda':
        return 0.0
    return float(torch.cuda.memory_allocated(device_)) / 1024.0 / 1024.0


def get_current_gpu_allocated_mb(device_):
    """Return currently allocated GPU memory in MB (explicit alias).

    Args:
        device_ (torch.device): Target CUDA device.

    Returns:
        float: Current allocated CUDA memory (MB), or 0.0 on CPU.
    """
    return get_current_gpu_memory_mb(device_)


def get_peak_gpu_memory_mb(device_):
    """Return peak allocated GPU memory in MB.

    Args:
        device_ (torch.device): Target CUDA device.

    Returns:
        float: Peak allocated CUDA memory (MB), or 0.0 on CPU.
    """
    if device_.type != 'cuda':
        return 0.0
    return float(torch.cuda.max_memory_allocated(device_)) / 1024.0 / 1024.0


def console_log(enabled, message, include_memory=False, device_=None):
    """Print timestamped debug message with optional memory snapshot.

    Args:
        enabled (bool): If False, no output is printed.
        message (str): Human-readable log message.
        include_memory (bool): If True, append CPU/GPU memory snapshot.
        device_ (torch.device, optional): Device used for GPU memory metrics.

    Returns:
        None.
    """
    if not enabled:
        return

    timestamp = datetime.now().strftime('%H:%M:%S')
    output = '[{}] {}'.format(timestamp, message)
    if include_memory and device_ is not None:
        current_cpu = get_current_cpu_memory_mb()
        peak_cpu = get_peak_cpu_memory_mb()
        current_gpu = get_current_gpu_memory_mb(device_)
        peak_gpu = get_peak_gpu_memory_mb(device_)
        output += ' | CPU current {:.1f} MB, CPU peak {:.1f} MB, GPU current {:.1f} MB, GPU peak {:.1f} MB'.format(
            current_cpu, peak_cpu, current_gpu, peak_gpu
        )
    print(output)
