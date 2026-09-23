#!/usr/bin/env python3
"""Open target labels only after all SAMap stability predictions are frozen, then score."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import phase18b_run_samap_resampled_condition_v1 as runner  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_deduplication_suffixes(names: list[str]) -> list[str]:
    normalized = [re.sub(r"-\d+$", "", name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Deduplication-suffix normalization is not one-to-one")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base_cfg = json.loads(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    root = Path(cfg["report_root"])
    final_dir = root / "final_gate"
    if final_dir.exists():
        raise RuntimeError(f"Refusing to overwrite aggregate: {final_dir}")

    loaded: dict[tuple[int, str, str], tuple[pd.DataFrame, list[str]]] = {}
    for seed in cfg["seeds"]:
        input_manifest_path = (
            Path(cfg["reused_input_manifest"])
            if seed == cfg["reused_seed"]
            else Path(cfg["prepared_root"]) / f"seed{seed}" / "inputs" / "manifest.json"
        )
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        for target in ("sb", "cr"):
            vocabulary = input_manifest["targets"][target]["common_source_vocabulary"]
            names_by_condition: dict[str, list[str]] = {}
            for condition in ("all_sources", "nearest_clade"):
                report_dir = (
                    Path(cfg["reused_report_root"]) / target / condition
                    if seed == cfg["reused_seed"]
                    else root / f"seed{seed}" / target / condition
                )
                summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
                if summary.get("status") != "completed":
                    raise RuntimeError(f"Incomplete SAMap condition: {report_dir}")
                if seed != cfg["reused_seed"] and summary.get("target_labels_opened") is not False:
                    raise RuntimeError(f"New SAMap run is not target-label blind: {report_dir}")
                prediction_path = Path(summary["blind_predictions"]["predictions"])
                if sha256(prediction_path) != summary["blind_predictions"]["predictions_sha256"]:
                    raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
                frame = pd.read_csv(prediction_path)
                names_by_condition[condition] = frame["target_obs_name"].astype(str).tolist()
                loaded[(int(seed), target, condition)] = (frame, vocabulary)
            if names_by_condition["all_sources"] != names_by_condition["nearest_clade"] and (
                normalize_deduplication_suffixes(names_by_condition["all_sources"])
                != normalize_deduplication_suffixes(names_by_condition["nearest_clade"])
            ):
                raise RuntimeError(f"Target cells differ between conditions: seed={seed}, target={target}")

    metric_rows, family_rows = [], []
    for seed in cfg["seeds"]:
        for target in ("sb", "cr"):
            for condition in ("all_sources", "nearest_clade"):
                frame, vocabulary = loaded[(int(seed), target, condition)]
                metric = runner.score_predictions(frame, target, vocabulary, base_cfg)
                metric_rows.append({
                    "seed": int(seed), "target": target, "condition": condition,
                    "macro_f1": metric["macro_f1"],
                    "balanced_accuracy": metric["balanced_accuracy"],
                    "prediction_coverage": metric["prediction_coverage"],
                    "evaluated_cells": metric["evaluated_cells"],
                })
                for family, value in metric["per_family_f1"].items():
                    family_rows.append({
                        "seed": int(seed), "target": target, "condition": condition,
                        "canonical_family": family, "f1": value,
                    })

    metrics = pd.DataFrame(metric_rows)
    paired = metrics.pivot(index=["seed", "target"], columns="condition", values="macro_f1").reset_index()
    paired["nearest_minus_all"] = paired["nearest_clade"] - paired["all_sources"]
    family = pd.DataFrame(family_rows).pivot(
        index=["seed", "target", "canonical_family"], columns="condition", values="f1"
    ).reset_index()
    family["nearest_minus_all"] = family["nearest_clade"] - family["all_sources"]
    all_wins = int((paired["nearest_minus_all"] < 0).sum())
    gate_passed = all_wins == len(paired)

    final_dir.mkdir(parents=True)
    metrics.to_csv(final_dir / "metrics.csv", index=False)
    paired.to_csv(final_dir / "paired_target_seed_differences.csv", index=False)
    family.to_csv(final_dir / "family_differences.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seeds": cfg["seeds"], "paired_target_seed_comparisons": int(len(paired)),
        "mean_all_sources_macro_f1": float(paired["all_sources"].mean()),
        "mean_nearest_clade_macro_f1": float(paired["nearest_clade"].mean()),
        "mean_nearest_minus_all_macro_f1": float(paired["nearest_minus_all"].mean()),
        "all_sources_wins": all_wins,
        "nearest_wins": int((paired["nearest_minus_all"] > 0).sum()),
        "gate_passed": gate_passed,
        "target_seed_results": paired.to_dict(orient="records"),
        "all_new_predictions_saved_and_hashed_before_target_label_open": True,
        "targets_fixed_across_seeds": True,
        "reference_cells_resampled_across_seeds": True,
        "reciprocal_maps_reused": True,
        "decision": (
            "Proceed to Phase 18C and 18D; SAMap external reference-composition advantage is stable to reference-cell resampling."
            if gate_passed else
            "Stop algorithm expansion and retain a conditional benchmark claim."
        ),
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
