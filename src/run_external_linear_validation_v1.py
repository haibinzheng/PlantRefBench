#!/usr/bin/env python3
"""Frozen matched-budget linear external validation; predictions precede label opening."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import run_phase2_loso_shared_orthogroup as losobase
import run_phase2_orthogroup_baseline as ogbase


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
MANIFEST = PROJECT / "configs/pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs/root_label_mapping_draft.csv"
PROTOCOL = PROJECT / "configs/external_validation_v1/external_validation_protocol_v1.json"
REPORT_ROOT = Path("/data/reports/phylo_plant_fm_pilot/external_validation_v1")
CATH_MAP = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/catharanthus_mapping_v1/catharanthus_h5ad_to_frozen_orthogroup.tsv.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_linear(seed):
    return SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced",
                         max_iter=1000, tol=1e-3, random_state=seed, n_jobs=4)


def score(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def load_target_blind(row, safe_maps, gene_to_og, og_index):
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        var_to_feature = np.full(adata.n_vars, -1, dtype=np.int32)
        if row["species"] == "Catharanthus roseus":
            if not CATH_MAP.exists():
                raise FileNotFoundError(f"Frozen Catharanthus mapping missing: {CATH_MAP}")
            with gzip.open(CATH_MAP, "rt", encoding="utf-8", newline="") as handle:
                external_map = {item["h5ad_gene"]: item["orthogroup"] for item in csv.DictReader(handle, delimiter="\t")}
            for position, h5_gene in enumerate(map(str, adata.var_names)):
                og = external_map.get(h5_gene, "")
                if og in og_index:
                    var_to_feature[position] = og_index[og]
        else:
            species = ogbase.SPECIES_KEY[row["species"]]
            dataset_map = safe_maps[row["dataset_id"]]
            species_ogs = gene_to_og[species]
            for position, h5_gene in enumerate(map(str, adata.var_names)):
                reference_gene = dataset_map.get(h5_gene)
                og = species_ogs.get(reference_gene, "")
                if og in og_index:
                    var_to_feature[position] = og_index[og]
        matrix = adata[:, :].to_memory().X
        obs_names = np.asarray(adata.obs_names.astype(str))
    finally:
        adata.file.close()
    x = ogbase.rank_to_orthogroup(matrix, var_to_feature, len(og_index), 256)
    return x, obs_names, int((var_to_feature >= 0).sum()), float(x.getnnz(axis=1).mean())


def open_target_labels(row, label_map):
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        labels = adata.obs[ogbase.source_label_column(adata)].astype(str).map(label_map)
        obs_names = np.asarray(adata.obs_names.astype(str))
    finally:
        adata.file.close()
    return obs_names, labels.to_numpy(object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["Sorghum bicolor", "Catharanthus roseus"])
    args = parser.parse_args()
    cfg = json.loads(PROTOCOL.read_text())
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    dev_rows = [row for row in rows if row["role"] == "train"]
    target_row = next(row for row in rows if row["species"] == args.target and row["role"] == "species_holdout")
    out = REPORT_ROOT / f"{args.target.lower().replace(' ', '_')}_linear_v1"
    out.mkdir(parents=True, exist_ok=False)

    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    training_species = {ogbase.SPECIES_KEY[row["species"]] for row in dev_rows}
    vocabulary, gene_to_og = ogbase.parse_orthogroups(training_species)
    og_index = {og: idx for idx, og in enumerate(vocabulary)}
    single_vocabulary = losobase.read_single_copy_orthogroups()
    single_index = {og: idx for idx, og in enumerate(single_vocabulary)}
    safe_maps = ogbase.read_safe_gene_mappings()
    x_target, target_obs, mapped_target_genes, mean_active_target_features = load_target_blind(
        target_row, safe_maps, gene_to_og, og_index
    )
    if args.target == "Catharanthus roseus" and (
        mapped_target_genes / int(target_row.get("n_genes", 17449) or 17449) < 0.50
        or mean_active_target_features < 64
    ):
        raise RuntimeError(
            f"Catharanthus mapping gate failed: mapped={mapped_target_genes}, "
            f"mean_active={mean_active_target_features:.2f}"
        )

    reference = cfg["reference_conditions"][args.target]
    conditions = {"all_sources": reference["all_sources"], "nearest_clade": reference["nearest_clade"]}
    order = {row["dataset_id"]: idx for idx, row in enumerate(dev_rows)}
    blinded_frames = []
    audit_rows = []
    for seed in cfg["primary_seeds"]:
        data = {}
        for row in dev_rows:
            sample_seed = int(seed) + 1009 * order[row["dataset_id"]]
            x, _, y, meta = losobase.load_features(
                row, label_map, safe_maps, gene_to_og, og_index, single_index,
                full_k=256, single_k=64, seed=sample_seed,
            )
            data[row["species"]] = {"x": x, "y": y}
            audit_rows.append({**meta, "seed": seed, "sampling_seed": sample_seed})

        pools = {}
        label_sets = {}
        for condition, species_set in conditions.items():
            pools[condition] = {
                "x": sparse.vstack([data[species]["x"] for species in species_set], format="csr"),
                "y": np.concatenate([data[species]["y"] for species in species_set]),
            }
            label_sets[condition] = set(pools[condition]["y"])
        common_classes = sorted(set.intersection(*label_sets.values()))
        common_set = set(common_classes)
        for pool in pools.values():
            keep = np.asarray([label in common_set for label in pool["y"]])
            pool["x"], pool["y"] = pool["x"][keep], pool["y"][keep]
        budget = min(len(pool["y"]) for pool in pools.values())

        for condition_index, condition in enumerate(("all_sources", "nearest_clade")):
            pool = pools[condition]
            rng = np.random.default_rng(int(seed) + 997 * condition_index)
            chosen = np.sort(rng.choice(len(pool["y"]), budget, replace=False))
            model = new_linear(int(seed) + 17 * condition_index)
            model.fit(pool["x"][chosen], pool["y"][chosen])
            prediction = model.predict(x_target)
            blinded_frames.append(pd.DataFrame({
                "seed": seed,
                "condition": condition,
                "obs_name": target_obs,
                "prediction": prediction,
                "common_classes": json.dumps(common_classes),
                "matched_train_cells": budget,
            }))

    blinded = pd.concat(blinded_frames, ignore_index=True)
    blinded_path = out / "predictions_blinded.csv.gz"
    blinded.to_csv(blinded_path, index=False)
    blinded_sha = sha256(blinded_path)

    # Labels are opened only after every prediction is durably saved and hashed.
    label_obs, canonical_labels = open_target_labels(target_row, label_map)
    if not np.array_equal(target_obs, label_obs):
        raise RuntimeError("Target observation order changed between prediction and scoring")
    label_lookup = dict(zip(label_obs, canonical_labels))
    metric_rows, family_rows, scored_frames = [], [], []
    for (seed, condition), frame in blinded.groupby(["seed", "condition"], sort=True):
        truth = np.asarray([label_lookup[item] for item in frame["obs_name"]], dtype=object)
        common_classes = json.loads(frame["common_classes"].iloc[0])
        evaluable = pd.notna(truth) & np.isin(truth, common_classes)
        y_true = truth[evaluable].astype(str)
        y_pred = frame.loc[evaluable, "prediction"].astype(str).to_numpy()
        metric_rows.append({
            "seed": seed, "target_species": args.target, "condition": condition,
            "matched_train_cells": int(frame["matched_train_cells"].iloc[0]),
            "common_train_classes": len(common_classes), "target_cells_total": len(frame),
            "target_cells_evaluable": int(evaluable.sum()), **score(y_true, y_pred),
        })
        scored = frame.loc[evaluable, ["seed", "condition", "obs_name", "prediction"]].copy()
        scored["true_label"] = y_true
        scored_frames.append(scored)
        for family in common_classes:
            mask = y_true == family
            family_rows.append({
                "seed": seed, "target_species": args.target, "condition": condition,
                "canonical_family": family, "support": int(mask.sum()),
                "f1": float(f1_score(mask, y_pred == family, zero_division=0)),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out / "family_metrics.csv", index=False)
    pd.concat(scored_frames, ignore_index=True).to_csv(out / "predictions_scored.csv.gz", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "development_feature_audit.csv", index=False)
    pivot = metrics.pivot(index="seed", columns="condition", values="macro_f1").reset_index()
    pivot["nearest_minus_all"] = pivot["nearest_clade"] - pivot["all_sources"]
    pivot.to_csv(out / "paired_differences.csv", index=False)
    summary = {
        "run_id": "external_linear_validation_v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "target_species": args.target,
        "mean_all_sources_macro_f1": float(pivot["all_sources"].mean()),
        "mean_nearest_clade_macro_f1": float(pivot["nearest_clade"].mean()),
        "mean_nearest_minus_all_macro_f1": float(pivot["nearest_minus_all"].mean()),
        "nearest_wins": int((pivot["nearest_minus_all"] > 0).sum()),
        "all_wins": int((pivot["nearest_minus_all"] < 0).sum()),
        "mapped_target_genes": mapped_target_genes,
        "mean_active_target_features": mean_active_target_features,
        "feature_count": len(vocabulary),
        "predictions_saved_before_label_open": True,
        "blinded_predictions_sha256": blinded_sha,
        "protocol_sha256": sha256(PROTOCOL),
        "source_h5ad_sha256": sha256(Path(target_row["source_h5ad"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
