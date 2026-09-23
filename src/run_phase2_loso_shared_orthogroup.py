#!/usr/bin/env python3
"""Five-seed leave-one-development-species-out shared-orthogroup benchmark."""

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

import run_phase2_orthogroup_baseline as ogbase
import run_phase2_study_holdout_baseline as rawbase


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
STATUS = PROJECT / "STATUS.json"
PROTOCOL = PROJECT / "BENCHMARK_PROTOCOL.md"
CONFIG = PROJECT / "configs" / "loso_shared_orthogroup_v1.json"
MANIFEST = PROJECT / "configs" / "pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs" / "root_label_mapping_draft.csv"
GENE_COUNTS = Path("/data/derived/phylo_plant_fm_pilot/orthology_runs/orthofinder_2_5_5_six_species_v1/Results_Sep10/Orthogroups/Orthogroups.GeneCount.tsv")
REPORTS = Path("/data/reports/phylo_plant_fm_pilot")
LOG_PATH = Path("/data/logs/phylo_plant_fm_pilot/loso_shared_orthogroup_v1.log")


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


def read_single_copy_orthogroups() -> list[str]:
    selected = []
    with GENE_COUNTS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        species_columns = [column for column in reader.fieldnames if column not in {"Orthogroup", "Total"}]
        for row in reader:
            if species_columns and all(int(row[column]) == 1 for column in species_columns):
                selected.append(row["Orthogroup"])
    if len(selected) < 100:
        raise RuntimeError(f"Implausibly few complete single-copy orthogroups: {len(selected)}")
    return selected


def sampled_rows(adata: ad.AnnData, row: dict, label_map: dict[str, str], seed: int):
    column = ogbase.source_label_column(adata)
    canonical = adata.obs[column].astype(str).map(label_map)
    positions = np.flatnonzero((canonical.notna() & canonical.ne("")).to_numpy())
    labels = canonical.iloc[positions].reset_index(drop=True)
    local = ogbase.stratified_indices(labels, int(row["max_cells"]), seed)
    obs_positions = positions[local]
    y = labels.iloc[local].to_numpy(dtype=str)
    obs_names = np.asarray(adata.obs_names.astype(str))[obs_positions]
    obs_hash = hashlib.sha256("\n".join(obs_names).encode("utf-8")).hexdigest()
    return obs_positions, y, obs_hash


def load_features(row: dict, label_map: dict[str, str], safe_maps: dict[str, dict[str, str]],
                  gene_to_og: dict[str, dict[str, str]], full_index: dict[str, int],
                  single_index: dict[str, int], full_k: int, single_k: int, seed: int):
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        obs_positions, y, obs_hash = sampled_rows(adata, row, label_map, seed)
        species = ogbase.SPECIES_KEY[row["species"]]
        dataset_map = safe_maps[row["dataset_id"]]
        species_ogs = gene_to_og[species]
        full_map = np.full(adata.n_vars, -1, dtype=np.int32)
        single_map = np.full(adata.n_vars, -1, dtype=np.int32)
        for position, h5_gene in enumerate(map(str, adata.var_names)):
            reference_gene = dataset_map.get(h5_gene)
            og = species_ogs.get(reference_gene, "")
            if og in full_index:
                full_map[position] = full_index[og]
            if og in single_index:
                single_map[position] = single_index[og]
        matrix = adata[obs_positions, :].to_memory().X
    finally:
        adata.file.close()

    x_full = ogbase.rank_to_orthogroup(matrix, full_map, len(full_index), full_k)
    variable_positions = np.flatnonzero(single_map >= 0)
    feature_positions = single_map[variable_positions]
    mapping = sparse.csr_matrix(
        (np.ones(len(variable_positions), dtype=np.float32),
         (np.arange(len(variable_positions)), feature_positions)),
        shape=(len(variable_positions), len(single_index)),
    )
    subset = matrix[:, variable_positions]
    if sparse.issparse(subset):
        aggregated = sparse.csr_matrix(subset) @ mapping
    else:
        aggregated = sparse.csr_matrix(np.asarray(subset)) @ mapping
    x_single = rawbase.topk_binary(aggregated, min(single_k, len(single_index)))
    metadata = {
        "dataset_id": row["dataset_id"], "species": row["species"], "seed": seed,
        "sampled_cells": len(y), "sampled_obs_names_sha256": obs_hash,
        "full_mapped_variables": int((full_map >= 0).sum()),
        "single_copy_mapped_variables": len(variable_positions),
        "mean_active_full_orthogroups": float(x_full.getnnz(axis=1).mean()),
        "mean_active_single_copy_orthogroups": float(x_single.getnnz(axis=1).mean()),
    }
    return x_full, x_single, y, metadata


def new_linear(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", alpha=1e-4, class_weight="balanced", max_iter=1000,
        tol=1e-3, random_state=seed, n_jobs=4,
    )


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_dir = REPORTS / f"phase2_loso_shared_orthogroup_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    update_status(
        current_stage="P2_LOSO_SHARED_ORTHOGROUP", state="running", attempt=1,
        pid=os.getpid(), last_log=str(LOG_PATH), last_report=str(report_dir),
        next_action="Compare complete single-copy and multi-copy orthogroup representations in five-species LOSO",
        blocking_issue=None,
    )

    all_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    rows = [row for row in all_rows if row["role"] in set(cfg["development_roles"])]
    if len(rows) != 5 or len({row["species"] for row in rows}) != 5:
        raise RuntimeError("Expected exactly one frozen training dataset for each of five development species")
    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    training_species = {ogbase.SPECIES_KEY[row["species"]] for row in rows}
    full_vocabulary, gene_to_og = ogbase.parse_orthogroups(training_species)
    single_vocabulary = read_single_copy_orthogroups()
    full_index = {og: index for index, og in enumerate(full_vocabulary)}
    single_index = {og: index for index, og in enumerate(single_vocabulary)}
    missing = sorted(set(single_vocabulary) - set(full_vocabulary))
    if missing:
        raise RuntimeError(f"Complete single-copy orthogroups missing from full vocabulary: {len(missing)}")
    safe_maps = ogbase.read_safe_gene_mappings()
    dataset_order = {row["dataset_id"]: index for index, row in enumerate(rows)}

    metrics = []
    metadata = []
    for seed in cfg["seeds"]:
        data = {}
        for row in rows:
            sample_seed = int(seed) + 1009 * dataset_order[row["dataset_id"]]
            full_x, single_x, y, meta = load_features(
                row, label_map, safe_maps, gene_to_og, full_index, single_index,
                int(cfg["top_k_full_orthogroups"]), int(cfg["top_k_single_copy_orthogroups"]),
                sample_seed,
            )
            data[row["dataset_id"]] = {"full": full_x, "single": single_x, "y": y}
            metadata.append({**meta, "sampling_seed": sample_seed})
            print(json.dumps({"loaded_seed_dataset": [seed, row["dataset_id"]]}, ensure_ascii=False), flush=True)

        for fold, held_row in enumerate(rows):
            held_id = held_row["dataset_id"]
            train_ids = [row["dataset_id"] for row in rows if row["dataset_id"] != held_id]
            y_train = np.concatenate([data[dataset_id]["y"] for dataset_id in train_ids])
            y_test = data[held_id]["y"]
            train_classes = set(y_train)
            evaluable = np.asarray([label in train_classes for label in y_test])
            unseen = sorted(set(y_test[~evaluable]))
            y_eval = y_test[evaluable]
            representations = {
                "dummy_prior": "single",
                "complete_single_copy_rank_top64_sgd": "single",
                "multicopy_orthogroup_rank_top256_sgd": "full",
            }
            for model_index, (model_name, representation) in enumerate(representations.items()):
                x_train = sparse.vstack([data[dataset_id][representation] for dataset_id in train_ids], format="csr")
                x_test = data[held_id][representation][evaluable]
                if model_name == "dummy_prior":
                    model = DummyClassifier(strategy="prior")
                else:
                    model = new_linear(int(seed) + 101 * fold + model_index)
                model.fit(x_train, y_train)
                metrics.append({
                    "seed": seed, "heldout_species": held_row["species"],
                    "heldout_dataset": held_id, "train_datasets": json.dumps(train_ids),
                    "model": model_name, "train_cells": len(y_train),
                    "test_cells_total": len(y_test), "test_cells_evaluable": len(y_eval),
                    "unseen_test_families": json.dumps(unseen),
                    **metric_row(y_eval, model.predict(x_test)),
                })
            print(json.dumps({"completed_seed_fold": [seed, held_row["species"]]}, ensure_ascii=False), flush=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    pd.DataFrame(metadata).to_csv(report_dir / "dataset_feature_summary.csv", index=False)
    pivot = metrics_df.pivot_table(
        index=["seed", "heldout_species", "heldout_dataset"], columns="model", values="macro_f1",
    ).reset_index()
    pivot["multicopy_minus_single_copy_macro_f1"] = (
        pivot["multicopy_orthogroup_rank_top256_sgd"] - pivot["complete_single_copy_rank_top64_sgd"]
    )
    pivot.to_csv(report_dir / "paired_differences.csv", index=False)
    delta = pivot["multicopy_minus_single_copy_macro_f1"]
    summary = {
        "run_id": cfg["run_id"], "completed_at": now(), "status": "completed",
        "seeds": cfg["seeds"], "development_species": [row["species"] for row in rows],
        "complete_single_copy_features": len(single_vocabulary),
        "multicopy_orthogroup_features": len(full_vocabulary),
        "single_copy_mean_macro_f1": float(pivot["complete_single_copy_rank_top64_sgd"].mean()),
        "multicopy_mean_macro_f1": float(pivot["multicopy_orthogroup_rank_top256_sgd"].mean()),
        "mean_paired_delta_macro_f1": float(delta.mean()),
        "delta_standard_deviation": float(delta.std(ddof=1)),
        "multicopy_win_count": int((delta > 0).sum()),
        "single_copy_win_count": int((delta < 0).sum()),
        "input_sha256": {
            "protocol": sha256(PROTOCOL), "config": sha256(CONFIG), "manifest": sha256(MANIFEST),
            "label_mapping": sha256(LABEL_MAPPING), "gene_mapping": sha256(ogbase.GENE_MAPPING),
            "orthogroups": sha256(ogbase.ORTHOGROUPS), "gene_counts": sha256(GENE_COUNTS),
            "code": sha256(Path(__file__)), "orthogroup_module": sha256(Path(ogbase.__file__)),
        },
        "query_genome_policy": cfg["query_genome_policy"], "python": sys.version,
    }
    atomic_json(report_dir / "summary.json", summary)
    update_status(
        state="completed", pid=None, last_report=str(report_dir),
        next_action="Review five-species LOSO; install no new dependencies until an isolated established-baseline environment is approved",
        blocking_issue=None,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        update_status(
            state="failed", pid=None, blocking_issue=str(exc),
            next_action="Inspect preserved LOSO log and retry once only if safely recoverable",
        )
        traceback.print_exc()
        raise
