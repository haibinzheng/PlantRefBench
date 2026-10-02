#!/usr/bin/env python3
"""Aggregate the frozen two-target by two-condition SAMap external matrix."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/data/reports/phylo_plant_fm_pilot/phase17_samap_external_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_deduplication_suffixes(names: list[str]) -> list[str]:
    """Normalize only AnnData-style suffixes when needed for paired identity checks."""
    normalized = [re.sub(r"-\d+$", "", name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Deduplication-suffix normalization is not one-to-one")
    return normalized


def main() -> None:
    final_dir = ROOT / "final"
    if final_dir.exists():
        raise RuntimeError("Refusing to overwrite final external SAMap summary")
    records, details = [], {}
    for target in ("sb", "cr"):
        details[target] = {}
        prediction_names = {}
        for condition in ("all_sources", "nearest_clade"):
            path = ROOT / target / condition / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["status"] != "completed":
                raise RuntimeError(f"Incomplete condition: {target}/{condition}")
            pred_path = Path(summary["blind_predictions"]["predictions"])
            if sha256(pred_path) != summary["blind_predictions"]["predictions_sha256"]:
                raise RuntimeError(f"Prediction hash mismatch: {target}/{condition}")
            prediction_names[condition] = pd.read_csv(pred_path)["target_obs_name"].astype(str).tolist()
            metric = summary["metrics"]
            records.append({"target": target, "condition": condition,
                            "macro_f1": metric["macro_f1"],
                            "balanced_accuracy": metric["balanced_accuracy"],
                            "prediction_coverage": metric["prediction_coverage"],
                            "evaluated_cells": metric["evaluated_cells"]})
            details[target][condition] = summary
        all_names = prediction_names["all_sources"]
        nearest_names = prediction_names["nearest_clade"]
        if all_names != nearest_names and (
            normalize_deduplication_suffixes(all_names)
            != normalize_deduplication_suffixes(nearest_names)
        ):
            raise RuntimeError(f"Target cells differ across conditions for {target}")
    table = pd.DataFrame(records)
    paired = {}
    for target in ("sb", "cr"):
        block = table.loc[table.target.eq(target)].set_index("condition")
        paired[target] = {
            "nearest_minus_all_macro_f1": float(block.loc["nearest_clade", "macro_f1"] - block.loc["all_sources", "macro_f1"]),
            "nearest_minus_all_balanced_accuracy": float(block.loc["nearest_clade", "balanced_accuracy"] - block.loc["all_sources", "balanced_accuracy"]),
            "winner_macro_f1": "nearest_clade" if block.loc["nearest_clade", "macro_f1"] > block.loc["all_sources", "macro_f1"] else "all_sources",
        }
    final_dir.mkdir(parents=True)
    table.to_csv(final_dir / "metrics.csv", index=False)
    final = {
        "run_id": "phase17_samap_external_v1", "status": "completed",
        "targets": details, "paired_comparison": paired,
        "mean_macro_f1": {
            condition: float(table.loc[table.condition.eq(condition), "macro_f1"].mean())
            for condition in ("all_sources", "nearest_clade")
        },
        "all_target_cells_identical_between_conditions": True,
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (final_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
