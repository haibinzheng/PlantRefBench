#!/usr/bin/env python3
"""Score Phase 16 only after every blind prediction has been verified and saved."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

CONFIG = PROJECT / "configs/phase16_reference_mechanism_v1.json"
MANIFEST = PROJECT / "configs/pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs/root_label_mapping_draft.csv"
ROOT = Path("/data/reports/phylo_plant_fm_pilot/phase16_reference_mechanism_v1")
OUT = ROOT / "final_mechanism"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def target_labels(species: str, role: str) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    row = next(value for value in rows if value["species"] == species and value["role"] == role)
    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"), ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        obs = np.asarray(adata.obs_names.astype(str))
        labels = adata.obs[ogbase.source_label_column(adata)].astype(str).map(label_map).to_numpy(object)
    finally:
        adata.file.close()
    return obs, labels, row


def target_evaluation_indices(target: str, seed: int, labels: np.ndarray, row: dict, cfg: dict) -> tuple[np.ndarray, str]:
    """Mirror Phase 12 target sampling for development targets after blind prediction."""
    if target not in cfg["development_species"]:
        return np.arange(len(labels), dtype=int), "external_all_cells_after_label_mapping"
    rows = [item for item in csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")) if item["role"] == "train"]
    order = {item["dataset_id"]: index for index, item in enumerate(rows)}
    eligible = pd.notna(labels) & (labels != "")
    positions = np.flatnonzero(eligible)
    local_labels = pd.Series(labels[positions], dtype="object")
    local = ogbase.stratified_indices(local_labels, int(row["max_cells"]), int(seed) + 1009 * order[row["dataset_id"]])
    return positions[local], "phase12_label_stratified_max_cells"


def fit_blocked_ols(frame: pd.DataFrame, response: str = "macro_f1") -> dict:
    """OLS with fixed target/seed effects, using a transparent NumPy implementation."""
    predictors = ["reference_pairwise_breadth", "mean_target_distance", "mean_family_species_support"]
    columns, names = [np.ones(len(frame))], ["intercept"]
    for name in predictors:
        columns.append(frame[name].astype(float).to_numpy())
        names.append(name)
    for name, values in (("target", frame["target_species"]), ("seed", frame["seed"].astype(str))):
        levels = sorted(values.unique())
        for level in levels[1:]:
            columns.append((values == level).to_numpy(dtype=float))
            names.append(f"{name}[{level}]")
    x = np.column_stack(columns)
    y = frame[response].astype(float).to_numpy()
    beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    return {
        "names": names, "coefficients": dict(zip(names, map(float, beta))),
        "rank": int(rank), "design_columns": int(x.shape[1]), "condition_number": float(np.linalg.cond(x)),
        "singular_values": [float(value) for value in singular],
    }


def blocked_bootstrap(frame: pd.DataFrame, iterations: int) -> dict[str, dict[str, float | None]]:
    """Resample complete target×seed blocks, retaining all reference pairs per block."""
    rng = np.random.default_rng(20260916)
    blocks = [part for _, part in frame.groupby(["target_species", "seed"], sort=True)]
    baseline = fit_blocked_ols(frame)
    primary = ["reference_pairwise_breadth", "mean_target_distance", "mean_family_species_support"]
    samples = {name: [] for name in primary}
    for _ in range(iterations):
        chosen = [blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))]
        boot = pd.concat(chosen, ignore_index=True)
        result = fit_blocked_ols(boot)
        if result["rank"] < result["design_columns"]:
            continue
        for name in primary:
            samples[name].append(result["coefficients"][name])
    output = {}
    for name in primary:
        values = np.asarray(samples[name], dtype=float)
        output[name] = {
            "estimate": baseline["coefficients"][name],
            "bootstrap_replicates": int(len(values)),
            "ci95_low": float(np.quantile(values, 0.025)) if len(values) else None,
            "ci95_high": float(np.quantile(values, 0.975)) if len(values) else None,
        }
    return {"ols": baseline, "coefficient_bootstrap": output}


def target_within_associations(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Target-specific pair means and associations, separate from global OLS."""
    predictor_names = ["reference_pairwise_breadth", "mean_target_distance", "mean_family_species_support"]
    grouping = ["target_species", "target_kind", "combination_id"]
    keep = grouping + predictor_names
    means = metrics.groupby(grouping, as_index=False).agg(
        mean_macro_f1=("macro_f1", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
        seed_replicates=("seed", "nunique"),
    ).merge(metrics[keep].drop_duplicates(grouping), on=grouping, how="left", validate="one_to_one")
    rows = []
    for target, frame in means.groupby("target_species", sort=True):
        for predictor in predictor_names:
            x, y = frame[predictor].astype(float).to_numpy(), frame["mean_macro_f1"].astype(float).to_numpy()
            if len(frame) < 3 or np.isclose(np.ptp(x), 0):
                slope, rho = np.nan, np.nan
            else:
                slope = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1])
                rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            rows.append({
                "target_species": target, "target_kind": frame["target_kind"].iloc[0], "predictor": predictor,
                "combination_count": len(frame), "ols_slope": slope, "spearman_rho": rho,
            })
    associations = pd.DataFrame(rows)
    breadth = associations.loc[associations["predictor"].eq("reference_pairwise_breadth")]
    external = breadth.loc[breadth["target_species"].isin(["Sorghum bicolor", "Catharanthus roseus"])]
    positive = dict(zip(external["target_species"], external["spearman_rho"] > 0))
    gate = {
        "rule": "Both external targets must have strictly positive within-target Spearman rho between pairwise breadth and five-seed mean Macro-F1.",
        "per_target_spearman_rho": dict(zip(external["target_species"], external["spearman_rho"])),
        "per_target_positive": positive,
        "passed": bool(len(external) == 2 and all(positive.values())),
    }
    return means, associations, gate


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing aggregate: {OUT}")
    targets = cfg["development_species"] + cfg["external_species"]
    loaded = {}
    for target in targets:
        for seed in cfg["seeds"]:
            run = ROOT / "runs" / f"{target.lower().replace(' ', '_')}_seed{seed}"
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "completed" or summary.get("target_labels_read") is not False:
                raise RuntimeError(f"Incomplete or non-blind run: {run}")
            prediction_path = Path(summary["blinded_predictions_file"])
            if sha256(prediction_path) != summary["blinded_predictions_sha256"]:
                raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
            loaded[(target, seed)] = {
                "prediction": pd.read_csv(prediction_path),
                "design": pd.read_csv(run / "combination_design.csv"),
                "summary": summary,
            }
    # This is the only point at which target labels are opened.
    target_truth = {}
    for target in targets:
        target_truth[target] = (
            target_labels(target, "train") if target in cfg["development_species"] else target_labels(target, "species_holdout")
        )
    OUT.mkdir(parents=True)
    metric_rows, family_rows = [], []
    for (target, seed), payload in loaded.items():
        expected_obs, labels, target_row = target_truth[target]
        evaluation_indices, evaluation_rule = target_evaluation_indices(target, seed, labels, target_row, cfg)
        prediction, design = payload["prediction"], payload["design"]
        for combination_id, frame in prediction.groupby("combination_id", sort=True):
            observed = frame["obs_name"].astype(str).to_numpy()
            if not np.array_equal(observed, expected_obs):
                raise RuntimeError(f"Observation order mismatch: {target} seed={seed} pair={combination_id}")
            common = json.loads(frame["common_classes"].iloc[0])
            eval_labels = labels[evaluation_indices]
            evaluable = pd.notna(eval_labels) & np.isin(eval_labels, common)
            truth = eval_labels[evaluable].astype(str)
            pred = frame["prediction"].astype(str).to_numpy()[evaluation_indices][evaluable]
            row = design.loc[design["combination_id"].eq(combination_id)].iloc[0].to_dict()
            metric_rows.append({
                **row, "target_kind": payload["summary"]["target_kind"],
                "target_cells_total": len(labels), "sampled_target_cells": int(len(evaluation_indices)),
                "target_eval_sampling_rule": evaluation_rule, "target_cells_evaluable": int(evaluable.sum()),
                **scores(truth, pred),
            })
            for family in common:
                mask = truth == family
                support = int(mask.sum())
                family_rows.append({
                    "target_species": target, "seed": seed, "combination_id": combination_id,
                    "canonical_family": family, "support": support,
                    "eligible_for_mechanism": bool(support > 0),
                    "f1": float(f1_score(mask, pred == family, zero_division=0)) if support else np.nan,
                })
    metrics = pd.DataFrame(metric_rows)
    family = pd.DataFrame(family_rows)
    metrics.to_csv(OUT / "combination_metrics.csv", index=False)
    family.to_csv(OUT / "family_metrics.csv", index=False)
    family_eligible = family.loc[family["eligible_for_mechanism"]].copy()
    family_eligible.to_csv(OUT / "family_metrics_nonzero_support.csv", index=False)
    analysis = blocked_bootstrap(metrics, iterations=1000)
    combination_means, within_associations, direction_gate = target_within_associations(metrics)
    combination_means.to_csv(OUT / "combination_means_five_seed.csv", index=False)
    within_associations.to_csv(OUT / "target_within_associations.csv", index=False)
    target_summary = metrics.groupby(["target_species", "target_kind"], as_index=False).agg(
        combinations=("combination_id", "count"), mean_macro_f1=("macro_f1", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_evaluable_cells=("target_cells_evaluable", "mean"),
    )
    target_summary.to_csv(OUT / "target_summary.csv", index=False)
    zero_support = family.groupby(["target_species", "canonical_family"], as_index=False).agg(
        runs=("support", "size"), zero_support_runs=("eligible_for_mechanism", lambda value: int((~value).sum())),
    )
    zero_support.to_csv(OUT / "family_support_audit.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed", "targets": targets,
        "seeds": cfg["seeds"], "metric_rows": int(len(metrics)),
        "family_rows": int(len(family)), "family_rows_eligible_for_mechanism": int(len(family_eligible)),
        "zero_support_policy": "Rows with zero target-family support are preserved in family_metrics.csv but excluded from family-level mechanism inference.",
        "mechanism_analysis": analysis,
        "external_direction_gate": direction_gate,
        "all_predictions_saved_and_hashed_before_label_open": True,
        "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
