#!/usr/bin/env python3
"""Low-cost within-species, independent-study rank-feature baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
STATUS = PROJECT / "STATUS.json"
MANIFEST = PROJECT / "configs" / "pilot_manifest.csv"
MAPPING = PROJECT / "configs" / "root_label_mapping_draft.csv"
REPORTS = Path("/data/reports/phylo_plant_fm_pilot")
SEED = 20260910
TOP_K = 256
PAIRS = [
    ("Arabidopsis thaliana_PRJCA016521_Root", "Arabidopsis thaliana_PRJCA021408_Root"),
    ("Oryza sativa_PRJNA706435", "Oryza sativa_CRA004082_Root"),
    ("Zea mays_PRJNA454730", "Zea mays_PRJNA759548"),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_status(**changes: object) -> None:
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    payload.update(changes)
    payload["updated_at"] = now()
    atomic_json(STATUS, payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_label_column(adata: ad.AnnData) -> str:
    for candidate in ("cell_type_original", "celltype_after"):
        if candidate in adata.obs:
            return candidate
    raise KeyError("No preserved source label column")


def stratified_indices(labels: pd.Series, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {label: np.flatnonzero(labels.to_numpy() == label) for label in sorted(labels.unique())}
    quota = max(1, cap // max(1, len(groups)))
    selected: list[np.ndarray] = []
    remaining: list[int] = []
    for positions in groups.values():
        take = min(quota, len(positions))
        chosen = rng.choice(positions, size=take, replace=False)
        selected.append(chosen)
        remaining.extend(np.setdiff1d(positions, chosen, assume_unique=False).tolist())
    current = sum(len(part) for part in selected)
    if current < cap and remaining:
        take = min(cap - current, len(remaining))
        selected.append(rng.choice(np.asarray(remaining), size=take, replace=False))
    return np.sort(np.concatenate(selected).astype(int))


def topk_binary(matrix: object, k: int) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        source = sparse.csr_matrix(matrix)
        rows: list[int] = []
        cols: list[int] = []
        for row in range(source.shape[0]):
            start, end = source.indptr[row], source.indptr[row + 1]
            indices = source.indices[start:end]
            values = source.data[start:end]
            if len(indices) > k:
                keep = np.argpartition(values, -k)[-k:]
                indices = indices[keep]
            rows.extend([row] * len(indices))
            cols.extend(indices.tolist())
    else:
        source = np.asarray(matrix)
        width = min(k, source.shape[1])
        chosen = np.argpartition(source, -width, axis=1)[:, -width:]
        rows = np.repeat(np.arange(source.shape[0]), width).tolist()
        cols = chosen.reshape(-1).tolist()
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=source.shape)


def load_dataset(row: dict, label_map: dict[str, str], common_genes: pd.Index, seed: int) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        column = source_label_column(adata)
        canonical = adata.obs[column].astype(str).map(label_map)
        eligible = canonical.notna() & canonical.ne("")
        positions = np.flatnonzero(eligible.to_numpy())
        eligible_labels = canonical.iloc[positions].reset_index(drop=True)
        sampled_local = stratified_indices(eligible_labels, int(row["max_cells"]), seed)
        sampled_obs = positions[sampled_local]
        sampled_labels = eligible_labels.iloc[sampled_local].to_numpy(dtype=str)
        var_positions = adata.var_names.get_indexer(common_genes)
        if np.any(var_positions < 0):
            raise RuntimeError("Common-gene indexing failed")
        matrix = adata[sampled_obs, var_positions].to_memory().X
        features = topk_binary(matrix, min(TOP_K, len(common_genes)))
        meta = {
            "dataset_id": row["dataset_id"],
            "expression_track": row["expression_track"],
            "eligible_cells": int(eligible.sum()),
            "sampled_cells": int(len(sampled_obs)),
            "sampled_classes": int(len(np.unique(sampled_labels))),
        }
        return features, sampled_labels, meta
    finally:
        adata.file.close()


def metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> int:
    run_id = datetime.now().astimezone().strftime("phase2_study_baseline_%Y%m%d_%H%M%S")
    report_dir = REPORTS / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    update_status(
        current_stage="P2_STUDY_HOLDOUT_BASELINE",
        state="running",
        pid=os.getpid(),
        last_report=str(report_dir),
        next_action="Run rank-feature independent-study baselines",
        blocking_issue=None,
    )

    manifest = {row["dataset_id"]: row for row in csv.DictReader(MANIFEST.open(encoding="utf-8", newline=""))}
    mapping = pd.read_csv(MAPPING).fillna("")
    label_map = dict(
        mapping.loc[mapping["mapping_status"].eq("mapped"), ["source_label", "canonical_family"]].itertuples(index=False, name=None)
    )
    result_rows: list[dict] = []
    pair_rows: list[dict] = []

    for pair_index, (train_id, test_id) in enumerate(PAIRS):
        train_row, test_row = manifest[train_id], manifest[test_id]
        train_adata = ad.read_h5ad(train_row["source_h5ad"], backed="r")
        test_adata = ad.read_h5ad(test_row["source_h5ad"], backed="r")
        try:
            common = train_adata.var_names.intersection(test_adata.var_names, sort=False)
        finally:
            train_adata.file.close()
            test_adata.file.close()
        if len(common) < 500:
            raise RuntimeError(f"Too few within-species common genes for {train_id} -> {test_id}: {len(common)}")

        x_train, y_train, train_meta = load_dataset(train_row, label_map, common, SEED + pair_index)
        x_test, y_test, test_meta = load_dataset(test_row, label_map, common, SEED + 100 + pair_index)
        train_classes = set(y_train)
        evaluable = np.asarray([label in train_classes for label in y_test])
        unseen_test = sorted(set(y_test[~evaluable]))
        x_eval, y_eval = x_test[evaluable], y_test[evaluable]
        if len(y_eval) == 0:
            raise RuntimeError(f"No shared canonical labels for {train_id} -> {test_id}")

        dummy = DummyClassifier(strategy="prior")
        dummy.fit(x_train, y_train)
        linear = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=SEED,
            n_jobs=4,
        )
        linear.fit(x_train, y_train)
        for name, model in (("dummy_prior", dummy), ("rank_top256_sgd", linear)):
            result_rows.append(
                {
                    "species": train_row["species"],
                    "train_dataset": train_id,
                    "test_dataset": test_id,
                    "common_raw_genes": len(common),
                    "train_cells": len(y_train),
                    "test_cells_total": len(y_test),
                    "test_cells_evaluable": len(y_eval),
                    "unseen_test_families": json.dumps(unseen_test),
                    **metrics(name, y_eval, model.predict(x_eval)),
                }
            )
        pair_rows.append({**train_meta, "pair_role": "train", "common_raw_genes": len(common)})
        pair_rows.append({**test_meta, "pair_role": "study_holdout", "common_raw_genes": len(common)})

    results = pd.DataFrame(result_rows)
    results.to_csv(report_dir / "study_holdout_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(report_dir / "dataset_feature_summary.csv", index=False)
    linear_rows = results[results["model"].eq("rank_top256_sgd")]
    summary = {
        "run_id": run_id,
        "completed_at": now(),
        "status": "completed",
        "scope": "within-species independent-study baseline; not cross-species evidence",
        "pair_count": len(PAIRS),
        "top_k": TOP_K,
        "seed": SEED,
        "mean_linear_macro_f1": float(linear_rows["macro_f1"].mean()),
        "mean_linear_balanced_accuracy": float(linear_rows["balanced_accuracy"].mean()),
        "manifest_sha256": sha256(MANIFEST),
        "mapping_sha256": sha256(MAPPING),
        "code_sha256": sha256(Path(__file__)),
        "python": sys.version,
    }
    atomic_json(report_dir / "summary.json", summary)
    update_status(
        state="completed",
        pid=None,
        last_report=str(report_dir),
        next_action="Review study-holdout baseline and acquire versioned official orthogroup references",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        update_status(state="failed", pid=None, blocking_issue=str(exc), next_action="Inspect Phase 2 baseline traceback")
        traceback.print_exc()
        raise

