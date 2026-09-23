#!/usr/bin/env python3
"""Leakage-safe orthogroup rank-feature baselines on frozen pilot splits."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
STATUS = PROJECT / "STATUS.json"
MANIFEST = PROJECT / "configs" / "pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs" / "root_label_mapping_draft.csv"
CONFIG = PROJECT / "configs" / "orthogroup_baseline_v1.json"
GENE_MAPPING = Path("/data/derived/phylo_plant_fm_pilot/mappings/gene_mapping_v1/h5ad_to_reference_gene.tsv.gz")
ORTHOGROUPS = Path("/data/derived/phylo_plant_fm_pilot/orthology_runs/orthofinder_2_5_5_six_species_v1/Results_Sep10/Orthogroups/Orthogroups.tsv")
REPORTS = Path("/data/reports/phylo_plant_fm_pilot")
CHECKPOINTS = Path("/data/checkpoints/phylo_plant_fm_pilot/orthogroup_baseline_v1")
LOG_PATH = Path("/data/logs/phylo_plant_fm_pilot/orthogroup_baseline_v1.log")
PAIRS = [
    ("Arabidopsis thaliana_PRJCA016521_Root", "Arabidopsis thaliana_PRJCA021408_Root"),
    ("Oryza sativa_PRJNA706435", "Oryza sativa_CRA004082_Root"),
    ("Zea mays_PRJNA454730", "Zea mays_PRJNA759548"),
]
SPECIES_KEY = {
    "Arabidopsis thaliana": "arabidopsis_thaliana",
    "Glycine max": "glycine_max",
    "Medicago truncatula": "medicago_truncatula",
    "Oryza sativa": "oryza_sativa",
    "Sorghum bicolor": "sorghum_bicolor",
    "Zea mays": "zea_mays",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_status(**changes: object) -> None:
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    payload.update(changes)
    payload["updated_at"] = now()
    atomic_json(STATUS, payload)


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
        selected.append(rng.choice(np.asarray(remaining), size=min(cap - current, len(remaining)), replace=False))
    return np.sort(np.concatenate(selected).astype(int))


def parse_orthogroups(training_species: set[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
    gene_to_og: dict[str, dict[str, str]] = defaultdict(dict)
    eligible: list[str] = []
    with ORTHOGROUPS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        species_columns = {column: column.removesuffix(".longest_protein") for column in reader.fieldnames[1:]}
        for row in reader:
            present_training = sum(bool(row[column].strip()) for column, species in species_columns.items() if species in training_species)
            if present_training < 2:
                continue
            og = row["Orthogroup"]
            eligible.append(og)
            for column, species in species_columns.items():
                for item in row[column].split(","):
                    item = item.strip()
                    if not item:
                        continue
                    prefix = species + "|"
                    gene = item[len(prefix):] if item.startswith(prefix) else item.split("|", 1)[-1]
                    if gene in gene_to_og[species] and gene_to_og[species][gene] != og:
                        raise RuntimeError(f"Reference gene assigned to multiple orthogroups: {species} {gene}")
                    gene_to_og[species][gene] = og
    return eligible, gene_to_og


def read_safe_gene_mappings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    with gzip.open(GENE_MAPPING, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["mapping_method"] in {"exact", "unique_canonical"} and row["reference_gene"]:
                result[row["dataset_id"]][row["h5ad_gene"]] = row["reference_gene"]
    return result


def rank_to_orthogroup(matrix: object, var_to_feature: np.ndarray, n_features: int, top_k: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    if sparse.issparse(matrix):
        source = sparse.csr_matrix(matrix)
        for row in range(source.shape[0]):
            start, end = source.indptr[row], source.indptr[row + 1]
            indices = source.indices[start:end]
            values = source.data[start:end]
            if len(indices) > top_k:
                indices = indices[np.argpartition(values, -top_k)[-top_k:]]
            mapped = np.unique(var_to_feature[indices])
            mapped = mapped[mapped >= 0]
            rows.extend([row] * len(mapped))
            cols.extend(mapped.tolist())
    else:
        source = np.asarray(matrix)
        width = min(top_k, source.shape[1])
        chosen = np.argpartition(source, -width, axis=1)[:, -width:]
        for row, indices in enumerate(chosen):
            mapped = np.unique(var_to_feature[indices])
            mapped = mapped[mapped >= 0]
            rows.extend([row] * len(mapped))
            cols.extend(mapped.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(matrix.shape[0], n_features))


def load_dataset(row: dict, label_map: dict[str, str], safe_maps: dict[str, dict[str, str]],
                 gene_to_og: dict[str, dict[str, str]], og_index: dict[str, int], top_k: int, seed: int):
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        column = source_label_column(adata)
        canonical = adata.obs[column].astype(str).map(label_map)
        eligible = canonical.notna() & canonical.ne("")
        positions = np.flatnonzero(eligible.to_numpy())
        labels = canonical.iloc[positions].reset_index(drop=True)
        sampled_local = stratified_indices(labels, int(row["max_cells"]), seed)
        sampled_obs = positions[sampled_local]
        y = labels.iloc[sampled_local].to_numpy(dtype=str)
        species = SPECIES_KEY[row["species"]]
        dataset_map = safe_maps[row["dataset_id"]]
        species_ogs = gene_to_og[species]
        var_to_feature = np.full(adata.n_vars, -1, dtype=np.int32)
        mapped_vars = 0
        for index, h5_gene in enumerate(map(str, adata.var_names)):
            reference_gene = dataset_map.get(h5_gene)
            og = species_ogs.get(reference_gene, "")
            if og in og_index:
                var_to_feature[index] = og_index[og]
                mapped_vars += 1
        matrix = adata[sampled_obs, :].to_memory().X
        x = rank_to_orthogroup(matrix, var_to_feature, len(og_index), top_k)
    finally:
        adata.file.close()
    meta = {
        "dataset_id": row["dataset_id"], "species": row["species"], "role": row["role"],
        "expression_track": row["expression_track"], "eligible_cells": int(eligible.sum()),
        "sampled_cells": len(y), "classes": len(np.unique(y)), "mapped_feature_genes": mapped_vars,
        "mean_active_orthogroups": float(x.getnnz(axis=1).mean()),
    }
    return x, y, meta


def score(scope: str, model_name: str, train_ids: list[str], test_id: str,
          y_true: np.ndarray, y_pred: np.ndarray, unseen: list[str]) -> dict:
    return {
        "scope": scope, "model": model_name, "train_datasets": json.dumps(train_ids),
        "test_dataset": test_id, "test_cells_evaluable": len(y_true),
        "unseen_test_families": json.dumps(unseen),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def new_linear(seed: int) -> SGDClassifier:
    return SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced", max_iter=1000,
                         tol=1e-3, random_state=seed, n_jobs=4)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_dir = REPORTS / f"phase2_orthogroup_baseline_{run_stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    if CHECKPOINTS.exists():
        raise RuntimeError(f"Checkpoint directory already exists; refusing overwrite: {CHECKPOINTS}")
    CHECKPOINTS.mkdir(parents=True)
    update_status(current_stage="P2_ORTHOGROUP_BASELINES", state="running", attempt=1, pid=os.getpid(),
                  last_log=str(LOG_PATH), last_report=str(report_dir),
                  next_action="Construct reference-defined orthogroup rank features and fit identical-split baselines",
                  blocking_issue=None)

    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    manifest = {row["dataset_id"]: row for row in manifest_rows}
    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    training_species = {SPECIES_KEY[row["species"]] for row in manifest_rows if row["role"] == "train"}
    vocabulary, gene_to_og = parse_orthogroups(training_species)
    og_index = {og: index for index, og in enumerate(vocabulary)}
    safe_maps = read_safe_gene_mappings()

    data = {}
    metadata = []
    included = [row for row in manifest_rows if row["species"] != "Catharanthus roseus"]
    for index, row in enumerate(included):
        x, y, meta = load_dataset(row, label_map, safe_maps, gene_to_og, og_index,
                                  int(cfg["top_k_genes_per_cell"]), int(cfg["seed"]) + index)
        data[row["dataset_id"]] = (x, y)
        metadata.append(meta)
        print(json.dumps({"loaded": row["dataset_id"], **meta}, ensure_ascii=False), flush=True)

    metrics = []
    for pair_index, (train_id, test_id) in enumerate(PAIRS):
        x_train, y_train = data[train_id]
        x_test, y_test = data[test_id]
        classes = set(y_train)
        evaluable = np.asarray([label in classes for label in y_test])
        x_eval, y_eval = x_test[evaluable], y_test[evaluable]
        unseen = sorted(set(y_test[~evaluable]))
        for name, model in (("dummy_prior", DummyClassifier(strategy="prior")),
                            ("orthogroup_rank_top256_sgd", new_linear(int(cfg["seed"]) + pair_index))):
            model.fit(x_train, y_train)
            metrics.append(score("paired_study_holdout", name, [train_id], test_id,
                                 y_eval, model.predict(x_eval), unseen))
            if name != "dummy_prior":
                joblib.dump(model, CHECKPOINTS / f"paired_{pair_index}_{name}.joblib")

    train_ids = [row["dataset_id"] for row in included if row["role"] == "train"]
    test_ids = [row["dataset_id"] for row in included if row["role"] in {"study_holdout", "species_holdout"}]
    x_train = sparse.vstack([data[dataset_id][0] for dataset_id in train_ids], format="csr")
    y_train = np.concatenate([data[dataset_id][1] for dataset_id in train_ids])
    models = [
        ("dummy_prior", DummyClassifier(strategy="prior"), None),
        ("orthogroup_rank_top256_sgd", new_linear(int(cfg["seed"])), None),
        ("orthogroup_rank_top256_mlp64", MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4,
            batch_size=1024, learning_rate_init=1e-3, max_iter=20, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=4, random_state=int(cfg["seed"]), verbose=True,
        ), compute_sample_weight(class_weight="balanced", y=y_train)),
    ]
    fitted = []
    for name, model, weights in models:
        if weights is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=weights)
        fitted.append((name, model))
        if name != "dummy_prior":
            joblib.dump(model, CHECKPOINTS / f"combined_{name}.joblib")
    train_classes = set(y_train)
    for test_id in test_ids:
        x_test, y_test = data[test_id]
        evaluable = np.asarray([label in train_classes for label in y_test])
        x_eval, y_eval = x_test[evaluable], y_test[evaluable]
        unseen = sorted(set(y_test[~evaluable]))
        for name, model in fitted:
            metrics.append(score("combined_train_to_holdout", name, train_ids, test_id,
                                 y_eval, model.predict(x_eval), unseen))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    pd.DataFrame(metadata).to_csv(report_dir / "dataset_feature_summary.csv", index=False)
    (report_dir / "orthogroup_vocabulary.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    checkpoint_manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(CHECKPOINTS.glob("*.joblib"))
    }
    paired_linear = metrics_df[(metrics_df.scope == "paired_study_holdout") &
                               (metrics_df.model == "orthogroup_rank_top256_sgd")]
    species_linear = metrics_df[(metrics_df.scope == "combined_train_to_holdout") &
                                (metrics_df.model == "orthogroup_rank_top256_sgd") &
                                metrics_df.test_dataset.str.startswith("Sorghum bicolor")]
    species_mlp = metrics_df[(metrics_df.scope == "combined_train_to_holdout") &
                             (metrics_df.model == "orthogroup_rank_top256_mlp64") &
                             metrics_df.test_dataset.str.startswith("Sorghum bicolor")]
    summary = {
        "run_id": cfg["run_id"], "completed_at": now(), "status": "completed",
        "seed": cfg["seed"], "feature_count": len(vocabulary), "training_cells": len(y_train),
        "paired_study_mean_sgd_macro_f1": float(paired_linear.macro_f1.mean()),
        "sorghum_sgd_macro_f1": float(species_linear.macro_f1.iloc[0]),
        "sorghum_mlp_macro_f1": float(species_mlp.macro_f1.iloc[0]),
        "catharanthus_evaluated": False, "catharanthus_reason": "exact study reference unresolved",
        "leakage_control": cfg["leakage_control"], "checkpoint_manifest": checkpoint_manifest,
        "input_sha256": {"manifest": sha256(MANIFEST), "label_mapping": sha256(LABEL_MAPPING),
                         "gene_mapping": sha256(GENE_MAPPING), "orthogroups": sha256(ORTHOGROUPS),
                         "config": sha256(CONFIG), "code": sha256(Path(__file__))},
        "python": sys.version,
    }
    atomic_json(report_dir / "summary.json", summary)
    update_status(state="completed", pid=None, last_report=str(report_dir),
                  next_action="Compare orthogroup gains with raw-gene baselines across seeds before any phylogeny-aware model",
                  blocking_issue=None)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        update_status(state="failed", pid=None, blocking_issue=str(exc),
                      next_action="Inspect the preserved orthogroup baseline log; retry once only if safely recoverable")
        traceback.print_exc()
        raise
