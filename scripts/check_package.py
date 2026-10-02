"""Offline packaging checks; never import or execute archived research runners."""
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(root=ROOT):
    sources = list((root / 'src').glob('*.py'))
    modules = {p.stem for p in sources}
    local_prefixes = ('phase16_', 'phase17_', 'phase18', 'run_phase2_', 'run_external_', 'prepare_external_', 'build_scvi_')
    for path in sources:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                name = name.split('.')[0]
                if name.startswith(local_prefixes):
                    assert name in modules, f'{path.name}: missing local module {name}'
    configs = list((root / 'configs').glob('*.json'))
    for path in configs:
        json.loads(path.read_text(encoding='utf-8'))
    for name in ('pilot_manifest.csv', 'root_label_mapping_draft.csv'):
        with (root / 'configs' / name).open(encoding='utf-8', newline='') as handle:
            assert list(csv.DictReader(handle)), f'Empty {name}'
    return {'python_sources_parsed': len(sources), 'json_configs_parsed': len(configs),
            'scope': 'syntax, named local imports, configuration parsing; not full-data reproduction'}

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))

