#!/usr/bin/env python3
"""Build frozen three-model per-family metrics, confusion flows and resource audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import phase18b_run_samap_resampled_condition_v1 as samap_runner  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_group(model: str, target: str, seed: int, condition: str,
              truth: np.ndarray, prediction: np.ndarray,
              family_rows: list[dict], confusion_rows: list[dict], coverage_rows: list[dict]) -> None:
    truth = np.asarray(truth, dtype=str)
    prediction = np.asarray(prediction, dtype=str)
    if len(truth) != len(prediction) or len(truth) == 0:
        raise RuntimeError(f"Empty or misaligned scored rows: {model}/{target}/{seed}/{condition}")
    families = sorted(set(truth))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=families, zero_division=0
    )
    for family, p, r, f, n in zip(families, precision, recall, f1, support):
        family_rows.append({"model": model, "target_species": target, "seed": seed,
                            "condition": condition, "canonical_family": family,
                            "support": int(n), "precision": float(p),
                            "recall": float(r), "f1": float(f)})
    flows = pd.DataFrame({"true_family": truth, "predicted_family": prediction}).value_counts().reset_index(name="cells")
    denominator = pd.Series(truth).value_counts()
    for flow in flows.itertuples(index=False):
        confusion_rows.append({
            "model": model, "target_species": target, "seed": seed, "condition": condition,
            "true_family": flow.true_family, "predicted_family": flow.predicted_family,
            "cells": int(flow.cells),
            "fraction_of_true_family": float(flow.cells / denominator[flow.true_family]),
        })
    coverage_rows.append({
        "model": model, "target_species": target, "seed": seed, "condition": condition,
        "scored_cells": len(truth), "assigned_fraction": float(np.mean(prediction != "__unassigned__")),
        "unassigned_cells": int(np.sum(prediction == "__unassigned__")),
    })


def samap_algorithm_minutes(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    matches = re.findall(r"Elapsed time: ([0-9.]+) minutes", log_path.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    out = Path(cfg["report_root"])
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")

    family_rows: list[dict] = []
    confusion_rows: list[dict] = []
    coverage_rows: list[dict] = []
    resource_rows: list[dict] = []
    input_hashes: dict[str, str] = {}

    for target, code in cfg["targets"].items():
        linear_path = Path(cfg["linear_scored_roots"][target]) / "predictions_scored.csv.gz"
        input_hashes[str(linear_path)] = sha256(linear_path)
        linear = pd.read_csv(linear_path)
        for (seed, condition), block in linear.loc[linear.seed.isin(cfg["seeds"])].groupby(["seed", "condition"]):
            add_group("orthogroup_linear", target, int(seed), condition,
                      block["true_label"].to_numpy(), block["prediction"].to_numpy(),
                      family_rows, confusion_rows, coverage_rows)

    scanvi_path = Path(cfg["scanvi_scored"])
    input_hashes[str(scanvi_path)] = sha256(scanvi_path)
    scanvi = pd.read_csv(scanvi_path)
    for (target, seed, condition), block in scanvi.loc[scanvi.seed.isin(cfg["seeds"])].groupby(
        ["target_species", "seed", "condition"]
    ):
        add_group("scANVI", target, int(seed), condition,
                  block["true_label"].to_numpy(), block["prediction"].to_numpy(),
                  family_rows, confusion_rows, coverage_rows)
        run_root = (Path("/data/reports/phylo_plant_fm_pilot/external_validation_v1/external_scanvi_v1")
                    if int(seed) == cfg["seeds"][0]
                    else scanvi_path.parent.parent)
        run_dir = run_root / f"{target.lower().replace(' ', '_')}_{condition}_seed{seed}"
        run_summary_path = run_dir / "summary.json"
        run_summary = json.loads(run_summary_path.read_text())
        resource_rows.append({
            "model": "scANVI", "target_species": target, "seed": int(seed), "condition": condition,
            "scvi_train_minutes": float(run_summary["scvi_seconds"]) / 60,
            "scanvi_train_minutes": float(run_summary["scanvi_seconds"]) / 60,
            "samap_algorithm_minutes": None,
            "blind_mapping_bytes": None,
            "peak_cpu_memory_bytes": None, "peak_gpu_memory_bytes": None,
        })

    base_cfg = json.loads(Path("/workspace/projects/phylo_plant_fm_pilot/configs/phase17_samap_external_v1.json").read_text())
    lookup = samap_runner.canonical_lookup()
    target_truth = {code: samap_runner.raw_label_series(code, base_cfg, lookup)
                    for code in cfg["targets"].values()}
    for seed in cfg["seeds"]:
        manifest_path = (Path(cfg["samap_reused_input_manifest"])
                         if seed == cfg["seeds"][0]
                         else Path(cfg["samap_resampled_input_root"]) / f"seed{seed}" / "inputs" / "manifest.json")
        manifest = json.loads(manifest_path.read_text())
        input_hashes[str(manifest_path)] = sha256(manifest_path)
        for target, code in cfg["targets"].items():
            vocabulary = manifest["targets"][code]["common_source_vocabulary"]
            truth_series = target_truth[code]
            for condition in ("all_sources", "nearest_clade"):
                report_dir = (Path(cfg["samap_reused_report_root"]) / code / condition
                              if seed == cfg["seeds"][0]
                              else Path(cfg["samap_resampled_report_root"]) / f"seed{seed}" / code / condition)
                summary_path = report_dir / "summary.json"
                summary = json.loads(summary_path.read_text())
                pred_path = Path(summary["blind_predictions"]["predictions"])
                if sha256(pred_path) != summary["blind_predictions"]["predictions_sha256"]:
                    raise RuntimeError(f"SAMap prediction hash mismatch: {pred_path}")
                input_hashes[str(pred_path)] = summary["blind_predictions"]["predictions_sha256"]
                frame = pd.read_csv(pred_path)
                names = pd.Index(frame["target_obs_name"].astype(str))
                resolved, _ = samap_runner.resolve_deduplicated_names(names, truth_series.index,
                                                                        f"SAMap {target}/{seed}/{condition}")
                aligned = truth_series.reindex(resolved)
                if aligned.isna().any():
                    raise RuntimeError(f"SAMap target labels missing: {target}/{seed}/{condition}")
                evaluable = aligned.isin(vocabulary).to_numpy()
                add_group("SAMap", target, int(seed), condition,
                          aligned.to_numpy()[evaluable], frame["prediction"].to_numpy()[evaluable],
                          family_rows, confusion_rows, coverage_rows)
                log_path = (Path("/data/logs/phylo_plant_fm_pilot/phase17_samap_external_v1/runs")
                            / f"{code}_{condition}.log" if seed == cfg["seeds"][0]
                            else Path("/data/logs/phylo_plant_fm_pilot/phase18b_samap_resampling_v1")
                            / f"seed{seed}_{code}_{condition}.log")
                blind_path = Path(summary["blind_mapping"]["stitched_h5ad"])
                resource_rows.append({
                    "model": "SAMap", "target_species": target, "seed": int(seed), "condition": condition,
                    "scvi_train_minutes": None, "scanvi_train_minutes": None,
                    "samap_algorithm_minutes": samap_algorithm_minutes(log_path),
                    "blind_mapping_bytes": blind_path.stat().st_size,
                    "peak_cpu_memory_bytes": None, "peak_gpu_memory_bytes": None,
                })

    family = pd.DataFrame(family_rows)
    confusion = pd.DataFrame(confusion_rows)
    coverage = pd.DataFrame(coverage_rows)
    resources = pd.DataFrame(resource_rows)
    expected_blocks = 3 * 2 * 3 * 2
    if len(coverage) != expected_blocks or coverage.duplicated(["model", "target_species", "seed", "condition"]).any():
        raise RuntimeError(f"Expected {expected_blocks} unique model-target-seed-condition blocks")

    out.mkdir(parents=True)
    family.to_csv(out / "family_precision_recall_f1.csv", index=False)
    confusion.to_csv(out / "confusion_flows.csv", index=False)
    coverage.to_csv(out / "prediction_coverage.csv", index=False)
    resources.to_csv(out / "resource_audit.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "evaluation_blocks": int(len(coverage)),
        "family_rows": int(len(family)), "confusion_flows": int(len(confusion)),
        "resource_rows": int(len(resources)),
        "unrecorded_resource_fields": ["peak_cpu_memory_bytes", "peak_gpu_memory_bytes"],
        "label_policy": "Frozen canonical labels and existing evaluable cells; no model refitting.",
        "input_sha256": input_hashes, "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
