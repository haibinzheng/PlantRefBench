#!/usr/bin/env python3
"""Open labels only after all frozen external scANVI predictions exist, then score."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

MANIFEST = PROJECT / "configs/pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs/root_label_mapping_draft.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    run_root = Path(cfg["report_root"])
    reused_root = Path(cfg["reused_seed_root"])
    out = run_root / "final_gate"
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite aggregate: {out}")
    loaded = {}
    for seed in cfg["seeds"]:
        seed_root = reused_root if seed == cfg["reused_seed"] else run_root
        for target in cfg["external_targets"]:
            target_slug = target.lower().replace(" ", "_")
            for condition in ("all_sources", "nearest_clade"):
                run_dir = seed_root / f"{target_slug}_{condition}_seed{seed}"
                summary = json.loads((run_dir / "summary.json").read_text())
                if summary.get("status") != "completed" or summary.get("target_labels_read") is not False:
                    raise RuntimeError(f"Incomplete or non-blind run: {run_dir}")
                prediction_path = Path(summary["blinded_predictions_file"])
                if sha256(prediction_path) != summary["blinded_predictions_sha256"]:
                    raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
                loaded[(seed, target, condition)] = pd.read_csv(prediction_path)

    out.mkdir(parents=True)
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    metric_rows, family_rows, scored_frames = [], [], []
    for target in cfg["external_targets"]:
        row = next(item for item in rows if item["species"] == target and item["role"] == "species_holdout")
        adata = ad.read_h5ad(row["source_h5ad"], backed="r")
        try:
            label_column = ogbase.source_label_column(adata)
            obs_names = np.asarray(adata.obs_names.astype(str))
            labels = adata.obs[label_column].astype(str).map(label_map).to_numpy(object)
        finally:
            adata.file.close()
        for seed in cfg["seeds"]:
            condition_frames = [loaded[(seed, target, condition)] for condition in ("all_sources", "nearest_clade")]
            if not all(np.array_equal(frame["obs_name"].astype(str).to_numpy(), obs_names) for frame in condition_frames):
                raise RuntimeError(f"Target observation order mismatch for {target}, seed {seed}")
            common_lists = [json.loads(frame["common_classes"].iloc[0]) for frame in condition_frames]
            if common_lists[0] != common_lists[1]:
                raise RuntimeError(f"Conditions use different class vocabularies for {target}, seed {seed}")
            common_classes = common_lists[0]
            evaluable = pd.notna(labels) & np.isin(labels, common_classes)
            y_true = labels[evaluable].astype(str)
            for condition, frame in zip(("all_sources", "nearest_clade"), condition_frames):
                y_pred = frame.loc[evaluable, "prediction"].astype(str).to_numpy()
                metric_rows.append({
                    "seed": int(seed), "target_species": target, "condition": condition,
                    "matched_train_cells": int(frame["matched_train_cells"].iloc[0]),
                    "common_train_classes": len(common_classes), "target_cells_total": len(frame),
                    "target_cells_evaluable": int(evaluable.sum()), **score(y_true, y_pred),
                })
                scored = frame.loc[evaluable, ["obs_name", "prediction"]].copy()
                scored.insert(0, "condition", condition)
                scored.insert(0, "target_species", target)
                scored.insert(0, "seed", int(seed))
                scored["true_label"] = y_true
                scored_frames.append(scored)
                for family in common_classes:
                    mask = y_true == family
                    family_rows.append({
                        "seed": int(seed), "target_species": target, "condition": condition,
                        "canonical_family": family, "support": int(mask.sum()),
                        "f1": float(f1_score(mask, y_pred == family, zero_division=0)),
                    })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out / "family_metrics.csv", index=False)
    pd.concat(scored_frames, ignore_index=True).to_csv(out / "predictions_scored.csv.gz", index=False)
    pivot = metrics.pivot(index=["seed", "target_species"], columns="condition", values="macro_f1").reset_index()
    pivot["nearest_minus_all"] = pivot["nearest_clade"] - pivot["all_sources"]
    pivot.to_csv(out / "paired_species_differences.csv", index=False)
    family = pd.DataFrame(family_rows).pivot(
        index=["seed", "target_species", "canonical_family", "support"], columns="condition", values="f1"
    ).reset_index()
    family["nearest_minus_all"] = family["nearest_clade"] - family["all_sources"]
    family.to_csv(out / "family_differences.csv", index=False)
    all_wins = int((pivot["nearest_minus_all"] < 0).sum())
    interpretation_locked = all_wins == len(cfg["external_targets"]) * len(cfg["seeds"])
    summary = {
        "run_id": cfg["run_id"],
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seeds": cfg["seeds"],
        "paired_target_seed_comparisons": int(len(pivot)),
        "mean_all_sources_macro_f1": float(pivot["all_sources"].mean()),
        "mean_nearest_clade_macro_f1": float(pivot["nearest_clade"].mean()),
        "mean_nearest_minus_all_macro_f1": float(pivot["nearest_minus_all"].mean()),
        "nearest_wins": int((pivot["nearest_minus_all"] > 0).sum()),
        "all_sources_wins": all_wins,
        "target_seed_results": pivot.to_dict(orient="records"),
        "all_predictions_saved_and_hashed_before_label_open": True,
        "interpretation_locked": interpretation_locked,
        "decision": (
            "Lock paper around model/target-dependent reference-composition effects and robust external benefits of broader reference diversity."
            if interpretation_locked else
            "Retain benchmark-only claim and stop algorithm expansion."
        ),
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
