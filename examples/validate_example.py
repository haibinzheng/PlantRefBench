"""Validate the synthetic input contract without training or opening real data."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def validate(features_path=ROOT / 'synthetic_orthogroup_cells.csv', truth_path=ROOT / 'synthetic_target_truth.csv'):
    with features_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        assert {'cell_id', 'species', 'split', 'cell_family'} <= set(fields)
        features = [field for field in fields if field.startswith('OG_')]
        rows = list(reader)
    assert rows and features
    ids = [row['cell_id'] for row in rows]
    assert all(ids) and len(ids) == len(set(ids)), 'Duplicate or empty cell IDs'
    assert {row['split'] for row in rows} == {'reference', 'target'}
    reference = [row for row in rows if row['split'] == 'reference']
    target = [row for row in rows if row['split'] == 'target']
    assert all(row['cell_family'] for row in reference)
    assert all(not row['cell_family'] for row in target), 'Target labels leaked into feature table'
    for row in rows:
        assert all(math.isfinite(float(row[f])) and float(row[f]) >= 0 for f in features)
    with truth_path.open(encoding='utf-8', newline='') as handle:
        truth = list(csv.DictReader(handle))
    truth_ids = [row['cell_id'] for row in truth]
    assert len(truth_ids) == len(set(truth_ids))
    assert set(truth_ids) == {row['cell_id'] for row in target}
    assert all(row['cell_family'] for row in truth)
    return {'synthetic_only': True, 'reference_cells': len(reference), 'target_cells': len(target),
            'orthogroup_features': len(features), 'feature_sha256': hashlib.sha256(features_path.read_bytes()).hexdigest(),
            'scope': 'schema validation only; no model trained and no research metrics reproduced'}

if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))

