"""Select official Torch wheels from driver/GPU capabilities and verify execution."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TORCH_VERSION = '2.14.1'
BACKENDS = ('auto', 'cpu', 'cu126', 'cu130')


def version_tuple(value: str) -> tuple[int, ...]:
    """Parse numeric NVIDIA driver/compute versions without lexicographic ordering."""
    parts = tuple(int(part) for part in value.strip().split('.'))
    return parts + (0,) * max(0, 3 - len(parts))


def detect_gpus() -> list[dict]:
    """Inspect accessible NVIDIA GPUs without importing Torch or relying on nvcc."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() in ('', '-1'):
        return []
    binary = shutil.which('nvidia-smi')
    if binary is None:
        return []
    command = [binary, '--query-gpu=index,uuid,driver_version,compute_cap,name', '--format=csv,noheader']
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError('NVIDIA driver query failed. Repair the driver or select --backend cpu.\n' + result.stderr)
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            raise ValueError('Unexpected nvidia-smi GPU record')
        index, identifier, driver, capability, name = row
        devices.append(dict(index=index.strip(), uuid=identifier.strip(), driver=version_tuple(driver),
                            capability=version_tuple(capability), name=name.strip()))
    if visible:
        selectors = [part.strip() for part in visible.split(',')]
        selected = [gpu for gpu in devices if any(gpu['index'] == key or gpu['uuid'].startswith(key) for key in selectors)]
        # MIG identifiers name partitions rather than physical UUIDs; verify visibility in Torch below.
        if selected or not any(key.startswith('MIG-') for key in selectors):
            devices = selected
    return devices


def select_backend(requested: str, devices: list[dict]) -> str:
    """Favor validated cu126, requiring cu130 and driver580 for Blackwell or newer."""
    if requested not in BACKENDS:
        raise ValueError('Unsupported Torch backend')
    if requested == 'cpu' or (requested == 'auto' and not devices):
        return 'cpu'
    if not devices:
        raise ValueError('No accessible NVIDIA GPU; select --backend cpu')
    backend = ('cu130' if any(gpu['capability'] >= (10, 0, 0) for gpu in devices) else 'cu126') if requested == 'auto' else requested
    architecture_minimum = (7, 5, 0) if backend == 'cu130' else (5, 0, 0)
    if any(gpu['capability'] < architecture_minimum for gpu in devices):
        raise ValueError(f'{backend} does not support every visible GPU; select compatible devices or CPU')
    minimum = (580, 0, 0) if backend == 'cu130' else (525, 60, 13)
    if any(gpu['driver'] < minimum for gpu in devices):
        raise ValueError(f'{backend} requires a newer NVIDIA driver (minimum {minimum}); CUDA toolkit version alone is insufficient')
    if backend == 'cu126' and any(gpu['capability'] >= (10, 0, 0) for gpu in devices):
        raise ValueError('Blackwell or newer GPUs require the cu130 wheel for this Torch release')
    return backend


def installation_plan(backend: str) -> list[str]:
    """Construct a shell-free pip command using the exact official local wheel version."""
    return [sys.executable, '-m', 'pip', 'install', '--upgrade',
            f'torch=={TORCH_VERSION}+{backend}', '--index-url', f'https://download.pytorch.org/whl/{backend}']


def inspect_runtime() -> dict | None:
    """Read installed Torch metadata in a fresh process; missing/broken runtimes are replaceable."""
    code = "import json,torch; print(json.dumps(dict(version=torch.__version__, cuda=torch.version.cuda)))"
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=60)
    if result.returncode:
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def verify_runtime(backend: str) -> dict:
    """Execute FP32 matrix and convolution kernels in a fresh interpreter after setup."""
    code = '''import json,torch
backend = BACKEND_VALUE
use_gpu = backend != 'cpu'
if use_gpu and not torch.cuda.is_available():
    raise RuntimeError('Selected CUDA wheel cannot access the GPU; check driver and CUDA_VISIBLE_DEVICES')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
devices = ['cuda:' + str(i) for i in range(torch.cuda.device_count())] if use_gpu else ['cpu']
x = torch.arange(256,dtype=torch.float32).reshape(16,16)
checks = []
for device in devices:
    y = x.to(device) @ x.to(device)
    torch.testing.assert_close(y.cpu(),x @ x,atol=0,rtol=0)
    layer = torch.nn.Conv1d(3,8,1,bias=False).to(device)
    with torch.no_grad():
        layer.weight.fill_(1)
        result = layer(torch.ones(1,3,4,device=device))
    assert torch.isfinite(result).all() and torch.equal(result,torch.full_like(result,3))
    checks.append(dict(device=device,gpu=torch.cuda.get_device_name(device) if use_gpu else None))
print(json.dumps(dict(version=torch.__version__,cuda=torch.version.cuda,devices=checks,kernel_checks_passed=True)))
'''.replace('BACKEND_VALUE', repr(backend))
    environment = dict(os.environ)
    environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    result = subprocess.run([sys.executable, '-c', code], env=environment, text=True,
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Runtime validation failed. A newer driver or --backend cpu may be required.\n' + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv=None) -> dict:
    """Install the selected runtime explicitly and optionally save its validated report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Show selection/command without importing or installing Torch')
    parser.add_argument('--report', type=Path)
    options = parser.parse_args(argv)
    devices = [] if options.backend == 'cpu' else detect_gpus()
    backend = select_backend(options.backend, devices)
    command = installation_plan(backend)
    report = dict(backend=backend, gpu_detection=devices, command=command, dry_run=options.dry_run,
                  compatibility='Wheel selection uses driver and GPU architecture; validation checks actual kernels.')
    if not options.dry_run:
        current = inspect_runtime()
        expected_cuda = {'cpu': None, 'cu126': '12.6', 'cu130': '13.0'}[backend]
        matched = bool(current and current['version'].split('+')[0] == TORCH_VERSION and current['cuda'] == expected_cuda)
        if not matched:
            environment = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_EXTRA_INDEX_URL='',
                               PIP_INDEX_URL=f'https://download.pytorch.org/whl/{backend}')
            subprocess.run(command, env=environment, check=True)
        report.update(installed=not matched, validation=verify_runtime(backend))
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report
