"""CUDA execution checks and observations from the actual inference process.

Copied into the three torch workers. Runtime probes use a short-lived process
with the worker's Python environment; model observations come from inference
itself. Never infer model placement from nvidia-smi or configuration alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

_DIRECTORY = Path('/tmp/mcc-gpu-observations')
_lock = threading.Lock()
_check = {'state': 'checking', 'checkedAt': 0, 'detail': 'not_tested'}
_running = False
_worker_checks = {}


def cuda_check(torch=None, device='cuda:0'):
    try:
        if torch is None:
            import torch
        if not torch.cuda.is_available():
            return {'state': 'error', 'checkedAt': int(time.time()), 'detail': 'cuda_unavailable'}
        # Allocation alone is insufficient: execute a kernel, synchronize, and
        # check its result. Context/library errors must reach the status response.
        a = torch.ones((32, 32), device=device)
        result = a @ a
        torch.cuda.synchronize(device)
        if float(result[0, 0].item()) != 32.0:
            raise RuntimeError('Unexpected CUDA result')
        return {'state': 'ready', 'checkedAt': int(time.time()), 'detail': 'kernel_passed'}
    except Exception as exc:
        return {'state': 'error', 'checkedAt': int(time.time()), 'detail': 'cuda_check_failed:' + type(exc).__name__}


def _probe():
    global _check, _running
    activity = None
    try:
        directory = os.environ.get('GATEWAY_MAINTENANCE_DIR')
        if directory:
            import fcntl
            activity = (Path(directory) / 'activity.lock').open('rb')
            fcntl.flock(activity, fcntl.LOCK_SH | fcntl.LOCK_NB)
        completed = subprocess.run([sys.executable, '-m', 'app.services.gpu_diagnostics'],
                                   capture_output=True, text=True, timeout=20)
        value = json.loads(completed.stdout) if completed.returncode == 0 else None
        if not isinstance(value, dict) or value.get('state') not in {'ready', 'error'}:
            raise ValueError('Invalid CUDA probe response')
    except BlockingIOError:
        value = {'state': 'checking', 'checkedAt': 0, 'detail': 'maintenance_check'}
    except Exception as exc:
        value = {'state': 'error', 'checkedAt': int(time.time()), 'detail': 'probe_failed:' + type(exc).__name__}
    finally:
        if activity is not None:
            activity.close()
    with _lock:
        _check, _running = value, False


def _process_identity(pid):
    try:
        # starttime prevents PID reuse from turning an old observation into a
        # claim that a model is still loaded. Linux is the GPU deployment target.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


def model_devices(model):
    try:
        devices = set()
        # Pipeline.parameters() describes tunable hyperparameters, not tensors.
        # Inspect its contained torch models before considering that method.
        if isinstance(getattr(model, '_models', None), dict):
            for attribute in ('_models', '_pipelines', '_inferences'):
                for child in getattr(model, attribute, {}).values():
                    devices.update(model_devices(getattr(child, 'model', child)))
        elif hasattr(model, 'parameters'):
            devices.update(str(p.device) for p in model.parameters())
            if hasattr(model, 'buffers'):
                devices.update(str(p.device) for p in model.buffers())
        if not devices and hasattr(model, 'device'):
            devices.add(str(model.device))
        return sorted(devices)
    except Exception:
        return []


def observe_model(model, name):
    """Called before inference, including warm models; does not change placement."""
    devices = model_devices(model)
    gpu = any(d.startswith('cuda') for d in devices)
    device = 'unknown' if not devices else ('cuda' if all(d.startswith('cuda') for d in devices) else ('mixed' if gpu else 'cpu'))
    target = next((d for d in devices if d.startswith('cuda')), 'cuda:0')
    check = _worker_checks.get(target)
    if check is None or time.time() - check['checkedAt'] > 60:
        check = cuda_check(device=target)
        _worker_checks[target] = check
    value = {'name': Path(str(name)).name, 'device': device, 'devices': devices,
             'inference': 'not_tested', 'cuda': check['state'], 'detail': check['detail'],
             'checkedAt': int(time.time()), 'pid': os.getpid(),
             'identity': _process_identity(os.getpid()), 'released': False}
    _write_observation(value)
    return value


def inference_succeeded(value):
    try:
        if any(d.startswith('cuda') for d in value['devices']):
            import torch
            for device in value['devices']:
                if device.startswith('cuda'):
                    torch.cuda.synchronize(device)
        value = {**value, 'inference': 'passed', 'checkedAt': int(time.time())}
    except Exception as exc:
        _write_observation({**value, 'inference': 'failed', 'cuda': 'error',
                            'detail': 'inference_failed:' + type(exc).__name__})
        raise
    _write_observation(value)


def model_released():
    path = _DIRECTORY / f'{os.getpid()}.json'
    try:
        value = json.loads(path.read_text())
        _write_observation({**value, 'released': True})
    except (OSError, ValueError):
        pass


def _write_observation(value):
    # Diagnostic persistence must not make an otherwise valid job fail.
    try:
        _DIRECTORY.mkdir(mode=0o700, exist_ok=True)
        path = _DIRECTORY / f'{os.getpid()}.json'
        temporary = path.with_suffix('.new')
        temporary.write_text(json.dumps(value))
        os.replace(temporary, path)
        # Retain recent finished processes as explicitly historical evidence.
        paths = sorted(_DIRECTORY.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in paths[32:]:
            data = json.loads(old.read_text())
            if _process_identity(data['pid']) != data.get('identity'):
                old.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError):
        pass


def snapshot():
    global _running
    with _lock:
        if not _running and time.time() - _check['checkedAt'] > 60:
            _running = True
            threading.Thread(target=_probe, daemon=True).start()
        check = dict(_check)
    if check['state'] == 'ready' and time.time() - check['checkedAt'] > 120:
        check = {**check, 'state': 'unknown', 'detail': 'stale_check'}
    models = []
    for path in _DIRECTORY.glob('*.json'):
        try:
            value = json.loads(path.read_text())
            identity = _process_identity(value['pid'])
            value['state'] = 'loaded' if identity is not None and identity == value.get('identity') and not value.get('released') else 'previous'
            models.append({k: v for k, v in value.items() if k not in {'pid', 'identity', 'released'}})
        except (OSError, ValueError, KeyError):
            continue
    models.sort(key=lambda v: v['checkedAt'], reverse=True)
    # Show all loaded models and the latest completed process, without suggesting
    # that a batch worker's already terminated model remains in GPU memory.
    loaded = [m for m in models if m['state'] == 'loaded']
    previous = [m for m in models if m['state'] == 'previous'][:1]
    return {'backend': 'cuda', 'check': check, 'models': loaded + previous}


if __name__ == '__main__':
    print(json.dumps(cuda_check()))
