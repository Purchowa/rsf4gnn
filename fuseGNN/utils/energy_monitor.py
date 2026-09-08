"""GPU energy/power monitoring for training scripts.

This module provides :class:`EnergyMonitor`, a background sampler that measures
GPU power draw and accumulated energy during training.  It is intended for the
energy-efficiency analysis of the GNN aggregation backends (ref/gas/gar) and
graph reordering, where the hypothesis is that lower memory traffic translates
into lower energy consumption.

Two backends are supported, selected automatically at construction time:

* ``nvml``       -- via ``pynvml`` (package ``nvidia-ml-py``).  Preferred,
  because on Volta+ GPUs (incl. the A100 used on Colab) it exposes a hardware
  energy counter (``nvmlDeviceGetTotalEnergyConsumption``) that integrates
  energy inside the driver and avoids sampling error.
* ``nvidia_smi`` -- fallback that parses ``nvidia-smi --query-gpu=power.draw``.
  Always available on Colab even without ``pynvml`` installed; energy is then
  obtained by trapezoidal integration of the sampled power.

If neither backend is available (e.g. the analysis laptop without an NVIDIA
GPU), the monitor degrades gracefully: ``available`` stays ``False`` and
:meth:`summary` reports the reason instead of raising.
"""

import subprocess
import threading
import time


try:
    import pynvml
    _PYNVML_IMPORTED = True
except Exception:  # pragma: no cover - import guard
    _PYNVML_IMPORTED = False


def _nvidia_smi_available():
    """Return True if ``nvidia-smi`` can be invoked and reports power draw."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False
        float(result.stdout.strip().splitlines()[0])
        return True
    except Exception:
        return False


class EnergyMonitor:
    """Background GPU power sampler with cumulative energy accounting.

    Typical usage::

        monitor = EnergyMonitor(device_index=0, sample_interval_sec=0.2,
                                enabled=args.energy_profiling)
        monitor.start()
        for epoch in range(...):
            start = monitor.snapshot()
            train_one_epoch()
            end = monitor.snapshot()
            epoch_energy_j = end['cumulative_energy_j'] - start['cumulative_energy_j']
        monitor.stop()
        logger_summary = monitor.summary()
        monitor.close()

    Args:
        device_index (int): Physical GPU index for NVML/nvidia-smi.  Note this
            is the *physical* index and is independent of CUDA_VISIBLE_DEVICES;
            pass the first GPU listed in --gpus.
        sample_interval_sec (float): Delay between power samples in seconds.
        enabled (bool): If False, the monitor is a no-op (``available`` False).
    """

    def __init__(self, device_index=0, sample_interval_sec=0.2, enabled=True):
        self.device_index = int(device_index)
        self.sample_interval_sec = max(float(sample_interval_sec), 0.01)
        self.enabled = bool(enabled)

        self.available = False
        self.backend = None              # 'nvml' | 'nvidia_smi' | None
        self.energy_source = None        # 'nvml_energy_counter' | 'power_integration'
        self.unavailable_reason = None
        self.gpu_name = None
        self.power_limit_w = None

        self._handle = None
        self._has_energy_counter = False

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Sampling state (guarded by _lock).
        self._power_samples = []         # power in W, for avg/max/min
        self._energy_integ_j = 0.0       # trapezoidal integral (fallback energy)
        self._last_sample_t = None
        self._last_sample_p = None
        self._energy_counter_start_mj = None
        self._energy_counter_latest_mj = None
        self._t_start = None
        self._t_end = None

        if not self.enabled:
            self.unavailable_reason = 'disabled'
            return

        self._init_backend()

    # ------------------------------------------------------------------ setup
    def _init_backend(self):
        """Select and initialise the best available sampling backend."""
        if _PYNVML_IMPORTED:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                name = pynvml.nvmlDeviceGetName(self._handle)
                self.gpu_name = name.decode() if isinstance(name, bytes) else name
                try:
                    self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
                except Exception:
                    self.power_limit_w = None
                # Probe the hardware energy counter (Volta+).
                try:
                    self._energy_counter_latest_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                    self._has_energy_counter = True
                except Exception:
                    self._has_energy_counter = False
                self.backend = 'nvml'
                self.energy_source = ('nvml_energy_counter' if self._has_energy_counter
                                      else 'power_integration')
                self.available = True
                return
            except Exception as exc:
                self.unavailable_reason = 'nvml_init_failed: {}'.format(exc)

        if _nvidia_smi_available():
            self.backend = 'nvidia_smi'
            self.energy_source = 'power_integration'
            self.gpu_name = self._nvidia_smi_query('name')
            limit = self._nvidia_smi_query('power.limit')
            try:
                self.power_limit_w = float(limit) if limit is not None else None
            except Exception:
                self.power_limit_w = None
            self.available = True
            return

        if self.unavailable_reason is None:
            self.unavailable_reason = 'no_nvml_and_no_nvidia_smi'

    def _nvidia_smi_query(self, field):
        """Query a single ``nvidia-smi`` field for this device, or None."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu={}'.format(field),
                 '--format=csv,noheader,nounits', '--id={}'.format(self.device_index)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip().splitlines()[0].strip()
        except Exception:
            return None

    # --------------------------------------------------------------- sampling
    def _read_power_w(self):
        """Return instantaneous GPU power draw in Watts, or None on failure."""
        if self.backend == 'nvml':
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except Exception:
                return None
        if self.backend == 'nvidia_smi':
            value = self._nvidia_smi_query('power.draw')
            try:
                return float(value)
            except Exception:
                return None
        return None

    def _read_energy_counter_mj(self):
        """Return the NVML hardware energy counter in mJ, or None."""
        if self.backend == 'nvml' and self._has_energy_counter:
            try:
                return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            except Exception:
                return None
        return None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            power_w = self._read_power_w()
            now = time.perf_counter()
            counter_mj = self._read_energy_counter_mj()
            with self._lock:
                if power_w is not None:
                    self._power_samples.append(power_w)
                    if self._last_sample_t is not None and self._last_sample_p is not None:
                        dt = now - self._last_sample_t
                        # Trapezoidal energy integration (fallback / cross-check).
                        self._energy_integ_j += 0.5 * (power_w + self._last_sample_p) * dt
                    self._last_sample_t = now
                    self._last_sample_p = power_w
                if counter_mj is not None:
                    self._energy_counter_latest_mj = counter_mj
            self._stop_event.wait(self.sample_interval_sec)

    def _cumulative_energy_j_locked(self):
        """Cumulative energy since start in Joules. Caller must hold _lock."""
        if (self._has_energy_counter
                and self._energy_counter_start_mj is not None
                and self._energy_counter_latest_mj is not None):
            return (self._energy_counter_latest_mj - self._energy_counter_start_mj) / 1000.0
        return self._energy_integ_j

    # ------------------------------------------------------------ public API
    def measure_idle_power_w(self, seconds):
        """Sample idle GPU power for ``seconds`` and return the average in W.

        Call before :meth:`start` (e.g. while the GPU is idle) to obtain a
        baseline that can be subtracted in offline analysis.  Returns None if
        the monitor is unavailable or no samples were collected.
        """
        if not self.available or seconds <= 0:
            return None
        samples = []
        deadline = time.perf_counter() + float(seconds)
        while time.perf_counter() < deadline:
            power_w = self._read_power_w()
            if power_w is not None:
                samples.append(power_w)
            time.sleep(min(self.sample_interval_sec, float(seconds)))
        if not samples:
            return None
        return sum(samples) / len(samples)

    def start(self):
        """Start the background sampling thread (no-op if unavailable)."""
        if not self.available:
            return
        with self._lock:
            self._power_samples = []
            self._energy_integ_j = 0.0
            self._last_sample_t = None
            self._last_sample_p = None
            self._t_start = time.perf_counter()
            self._t_end = None
            self._energy_counter_start_mj = self._read_energy_counter_mj()
            self._energy_counter_latest_mj = self._energy_counter_start_mj
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def snapshot(self):
        """Return current cumulative energy/power, or None if unavailable.

        Reads the hardware energy counter fresh (when present) so per-epoch
        deltas are accurate even for short epochs.
        """
        if not self.available:
            return None
        counter_mj = self._read_energy_counter_mj()
        with self._lock:
            if counter_mj is not None:
                self._energy_counter_latest_mj = counter_mj
            return {
                'cumulative_energy_j': self._cumulative_energy_j_locked(),
                'latest_power_w': self._last_sample_p,
            }

    def stop(self):
        """Stop the background sampler and finalise the measurement window."""
        if not self.available:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        counter_mj = self._read_energy_counter_mj()
        with self._lock:
            if counter_mj is not None:
                self._energy_counter_latest_mj = counter_mj
            self._t_end = time.perf_counter()

    def summary(self):
        """Return a flat dict of energy metrics suitable for the JSON log."""
        if not self.available:
            return {
                'energy_available': False,
                'energy_unavailable_reason': self.unavailable_reason,
            }
        with self._lock:
            end_t = self._t_end if self._t_end is not None else time.perf_counter()
            duration = (end_t - self._t_start) if self._t_start is not None else 0.0
            total_energy_j = self._cumulative_energy_j_locked()
            samples = list(self._power_samples)
        avg_power_w = (sum(samples) / len(samples)) if samples else None
        mean_power_from_energy_w = (total_energy_j / duration) if duration > 0 else None
        return {
            'energy_available': True,
            'energy_backend': self.backend,
            'energy_source': self.energy_source,
            'total_energy_j': total_energy_j,
            'total_energy_wh': total_energy_j / 3600.0,
            'measured_duration_sec': duration,
            'avg_power_w': avg_power_w,
            'mean_power_from_energy_w': mean_power_from_energy_w,
            'max_power_w': max(samples) if samples else None,
            'min_power_w': min(samples) if samples else None,
            'num_power_samples': len(samples),
            'sample_interval_sec': self.sample_interval_sec,
            'gpu_name': self.gpu_name,
            'power_limit_w': self.power_limit_w,
            'device_index': self.device_index,
        }

    def close(self):
        """Release NVML resources (safe to call once at the very end)."""
        if self.backend == 'nvml' and _PYNVML_IMPORTED:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
