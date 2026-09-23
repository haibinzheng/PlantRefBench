#!/usr/bin/env python3
"""Score frozen budget-dose predictions after both targets are saved and hashed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import phase18c_run_linear_budget_dose_v1 as base


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    root = Path(cfg["report_root"])
    final = root / "final_dose"
    if final.exists():
        raise RuntimeError(f"Refusing to overwrite aggregate: {final}")

    frozen = {}
    for target in cfg["external_targets"]:
        directory = root / target.lower().replace(" ", "_")
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("status") != "completed" or summary.get("target_labels_opened") is not False:
            raise RuntimeError(f"Incomplete or non-blind prediction set: {target}")
        prediction_path = Path(summary["blinded_predictions_file"])
        if sha256(prediction_path) != summary["blinded_predictions_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {target}")
        frozen[target] = pd.read_csv(prediction_path)

    rows = list(csv.DictReader(base.MANIFEST.open(encoding="utf-8", newline="")))
    mapping = pd.read_csv(base.LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    metrics, family_rows = [], []
    for target in cfg["external_targets"]:
        row = next(item for item in rows if item["species"] == target and item["role"] == "species_holdout")
        obs_names, labels = base.open_target_labels(row, label_map)
        label_lookup = dict(zip(obs_names, labels))
        frame_all = frozen[target]
        for (seed, fraction, condition), frame in frame_all.groupby(
            ["seed", "budget_fraction", "condition"], sort=True
        ):
            if not np.array_equal(frame["obs_name"].astype(str).to_numpy(), obs_names):
                raise RuntimeError(f"Target observation order changed: {target}/{seed}/{fraction}/{condition}")
            common_classes = json.loads(frame["common_classes"].iloc[0])
            truth = np.asarray([label_lookup[name] for name in frame["obs_name"]], dtype=object)
            evaluable = pd.notna(truth) & np.isin(truth, common_classes)
            y_true = truth[evaluable].astype(str)
            y_pred = frame.loc[evaluable, "prediction"].astype(str).to_numpy()
            metrics.append({
                "target_species": target, "seed": int(seed), "budget_fraction": float(fraction),
                "condition": condition, "matched_train_cells": int(frame["matched_train_cells"].iloc[0]),
                "full_matched_budget": int(frame["full_matched_budget"].iloc[0]),
                "common_train_classes": len(common_classes), "target_cells_evaluable": int(evaluable.sum()),
                **base.score(y_true, y_pred),
            })
            for family in common_classes:
                mask = y_true == family
                family_rows.append({
                    "target_species": target, "seed": int(seed), "budget_fraction": float(fraction),
                    "condition": condition, "canonical_family": family, "support": int(mask.sum()),
                    "f1": float(f1_score(mask, y_pred == family, zero_division=0)),
                })

    metric_table = pd.DataFrame(metrics)
    pivot = metric_table.pivot(index=["target_species", "seed", "budget_fraction"],
                               columns="condition", values=["macro_f1", "balanced_accuracy"])
    pivot.columns = [f"{metric}_{condition}" for metric, condition in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ("macro_f1", "balanced_accuracy"):
        pivot[f"all_minus_nearest_{metric}"] = pivot[f"{metric}_all_sources"] - pivot[f"{metric}_nearest_clade"]
    dose = pivot.groupby(["target_species", "budget_fraction"], as_index=False).agg(
        mean_all_minus_nearest_macro_f1=("all_minus_nearest_macro_f1", "mean"),
        mean_all_minus_nearest_balanced_accuracy=("all_minus_nearest_balanced_accuracy", "mean"),
        all_sources_wins=("all_minus_nearest_macro_f1", lambda values: int((values > 0).sum())),
    )
    final.mkdir(parents=True)
    metric_table.to_csv(final / "metrics.csv", index=False)
    pd.DataFrame(family_rows).to_csv(final / "family_metrics.csv", index=False)
    pivot.to_csv(final / "paired_differences.csv", index=False)
    dose.to_csv(final / "dose_curve.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "paired_comparisons": int(len(pivot)),
        "all_sources_wins": int((pivot["all_minus_nearest_macro_f1"] > 0).sum()),
        "dose_results": dose.to_dict(orient="records"),
        "both_targets_predictions_hashed_before_labels_opened": True,
        "config_sha256": sha256(config_path), "code_sha256": sha256(Path(__file__)),
    }
    (final / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
