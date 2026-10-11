"""Audit a regular wheel's source/resources and its installed CLI outside checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def run(command: list[str], cwd: Path, environment: dict) -> str:
    """Run a checked subprocess and retain stdout or a useful failure diagnostic."""
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'{command} failed:\n{result.stdout}\n{result.stderr}')
    if '-m' in command and 'unittest' in command:
        if 'skipped=' in result.stderr:
            raise AssertionError('Installed numerical regression unexpectedly skipped tests')
        return result.stdout + result.stderr
    return result.stdout


def validate(wheel: Path, root: Path) -> dict:
    """Require every Python/YAML source to match and test the installed application."""
    expected = {}
    for path in (root / 'src').rglob('*'):
        if path.suffix not in ('.py', '.yaml', '.yml'):
            continue
        relative = path.relative_to(root / 'src')
        member = ('masif/' + str(relative.relative_to('api')) if relative.parts[0] == 'api'
                  else 'masif_torch/' + str(relative))
        expected[member] = path.read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        for name, content in expected.items():
            if archive.read(name) != content:
                raise AssertionError(f'Stale/missing wheel resource: {name}')
        metadata = archive.read('masif-0.2.0.dist-info/METADATA').decode()
        if 'Name: masif\n' not in metadata:
            raise AssertionError('Distribution must be named masif')
    with tempfile.TemporaryDirectory(prefix='masif_wheel_audit_') as folder:
        destination = Path(folder) / 'installed'
        environment = dict(os.environ)
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
             '--target', str(destination), str(wheel)], Path(folder), environment)
        environment['PYTHONPATH'] = str(destination)
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        code = '''import importlib,json,pathlib,sys
import masif,masif_torch
assert pathlib.Path(masif.__file__).is_relative_to(sys.argv[1])
assert pathlib.Path(masif_torch.__file__).is_relative_to(sys.argv[1])
assert 'torch' not in sys.modules
from masif_torch.configuration import compose_config,configuration_dict
names=['site','search','ligand','score','alignment','seed','neosurface','prepare_protein','prepare_ligand','rosetta','benchmark','train']
for name in names:
    compose_config(name)
for task in ['site','search','ligand','score','alignment']:
    c=configuration_dict(compose_config('train',['task='+task,'dataset.input=/tmp/x','trainer.output=/tmp/y']))
    a=masif.build_model(c['model']);b=masif_torch.build_model(c['model'])
    assert type(a) is type(b)
    b.load_state_dict(a.state_dict(),strict=True)
    assert all(__import__('torch').equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
for task in ['site','search','ligand','score','alignment','seed','neosurface','preparation','rosetta','benchmark']:
    importlib.import_module('masif_torch.tasks.'+task)
assert masif.build_model is masif_torch.build_model
print(json.dumps(dict(presets=len(names),task_folders=10,public_api_identity=True,model_state_exact=True,lazy_import=True)))
'''
        imports = json.loads(run([sys.executable, '-c', code, str(destination)], Path(folder), environment))
        for module in ['masif', 'masif_torch']:
            run([sys.executable, '-m', module, '--help'], Path(folder), environment)
        run([str(destination / 'bin/masif'), '--help'], Path(folder), environment)
        run([str(destination / 'bin/masif-torch'), '--help'], Path(folder), environment)
        run([str(destination / 'bin/masif'), 'train', 'task=search', '--cfg', 'job'], Path(folder), environment)
        plan = json.loads(run([str(destination / 'bin/masif'), 'install-runtime', '--backend', 'cpu', '--dry-run'], Path(folder), environment))
        bootstrap = json.loads(run([sys.executable, '-S', '-m', 'masif', 'install-runtime',
                                    '--backend', 'cpu', '--dry-run'], Path(folder), environment))
        if bootstrap['backend'] != 'cpu':
            raise AssertionError('Runtime setup must work before third-party dependencies load')
        if plan['backend'] != 'cpu' or not plan['dry_run']:
            raise AssertionError('Installed lightweight runtime launcher failed')
        suite = run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-v'], root, environment)
    return dict(passed=True, wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                python_files=sum(name.endswith('.py') for name in expected),
                yaml_resources=sum(name.endswith(('.yaml','.yml')) for name in expected),
                all_resources_byte_equal=True, cli_outside_checkout=True, stdlib_only_bootstrap=True, installed_imports=imports,
                installed_regressions_passed=True, installed_regressions_skipped=0,
                installed_regression_count=int(re.search(r'Ran (\d+) tests', suite).group(1)),
                dependency_scope='Regular --no-deps wheel in isolated prefix using the validated Python/Torch environment; not a clean dependency resolver or PyPI publication.')


def main() -> None:
    """Validate an explicitly built wheel and save auditable installation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, default=Path('artifacts/distribution/masif-0.2.0-py3-none-any.whl'))
    parser.add_argument('--report', type=Path, default=Path('docs/torch_migration/numerical_distribution.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(args.wheel.resolve(), root)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
