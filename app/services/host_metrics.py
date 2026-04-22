from __future__ import annotations

import subprocess
import time

from app.models import HostCPUInfo, HostGPUInfo, HostMemoryInfo, RuntimeMetricsResponse

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal runtime environments
    psutil = None


class HostMetricsCollector:
    def __init__(self) -> None:
        self._primed = False
        self._previous_cpu_sample: tuple[int, int] | None = None

    def collect(self) -> RuntimeMetricsResponse:
        return RuntimeMetricsResponse(
            timestamp=int(time.time()),
            cpu=HostCPUInfo(usage=round(self._cpu_usage(), 1)),
            memory=HostMemoryInfo(usage=round(self._memory_usage(), 1)),
            gpu=self._gpu_info(),
        )

    def _cpu_usage(self) -> float:
        if psutil is None:
            return self._cpu_usage_fallback()
        if not self._primed:
            self._primed = True
            return psutil.cpu_percent(interval=0.1)
        return psutil.cpu_percent(interval=None)

    def _cpu_usage_fallback(self) -> float:
        sample = self._read_cpu_sample()
        previous = self._previous_cpu_sample
        self._previous_cpu_sample = sample
        if previous is None:
            return 0.0
        idle_delta = sample[0] - previous[0]
        total_delta = sample[1] - previous[1]
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, round((1.0 - (idle_delta / total_delta)) * 100.0, 1)))

    def _read_cpu_sample(self) -> tuple[int, int]:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            parts = handle.readline().split()
        values = [int(value) for value in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return idle, total

    def _memory_usage(self) -> float:
        if psutil is not None:
            return psutil.virtual_memory().percent
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                value = raw_value.strip().split()[0]
                meminfo[key] = int(value)
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        if total <= 0:
            return 0.0
        used = total - available
        return (used / total) * 100.0

    def _gpu_info(self) -> HostGPUInfo:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return HostGPUInfo(available=False)

        if completed.returncode != 0:
            return HostGPUInfo(available=False)

        candidates: list[HostGPUInfo] = []
        for line in (completed.stdout or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                name = parts[0]
                gpu_utilization = float(parts[1])
                used_memory = int(float(parts[2]))
                total_memory = int(float(parts[3]))
            except ValueError:
                continue
            memory_utilization = round((used_memory / total_memory) * 100.0, 1) if total_memory > 0 else 0.0
            candidates.append(
                HostGPUInfo(
                    available=True,
                    name=name,
                    gpu_utilization=gpu_utilization,
                    memory_utilization=memory_utilization,
                    used_memory=used_memory,
                    total_memory=total_memory,
                )
            )

        if not candidates:
            return HostGPUInfo(available=False)
        return max(candidates, key=lambda info: (info.gpu_utilization or 0.0, info.used_memory or 0))


_collector = HostMetricsCollector()


def get_host_metrics_collector() -> HostMetricsCollector:
    return _collector
