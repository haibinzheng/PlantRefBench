#!/usr/bin/env python3
"""Describe frozen target orthogroup detection by canonical cell family."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import phase18b_run_samap_resampled_condition_v1 as runner  # noqa: E402


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
    out = Path(cfg["report_root"])
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    base_cfg = json.loads((PROJECT / "configs/phase17_samap_external_v1.json").read_text())
    lookup = runner.canonical_lookup()
    patterns_path = Path(cfg["family_pattern_table"])
    patterns = pd.read_csv(patterns_path)
    rows = []
    inputs = {"family_patterns": {"path": str(patterns_path), "sha256": sha256(patterns_path)}}
    for target, slug in cfg["targets"].items():
        code = "sb" if slug == "sorghum_bicolor" else "cr"
        summary_path = Path(cfg["target_cache_root"]) / slug / "summary.json"
        target_summary = json.loads(summary_path.read_text())
        matrix_path = Path(target_summary["matrix_file"])
        obs_path = Path(target_summary["obs_file"])
        if sha256(matrix_path) != target_summary["matrix_sha256"] or sha256(obs_path) != target_summary["obs_sha256"]:
            raise RuntimeError(f"Target cache hash mismatch: {target}")
        matrix = sparse.load_npz(matrix_path).tocsr()
        names = pd.read_csv(obs_path)["obs_name"].astype(str)
        if len(names) != matrix.shape[0] or names.duplicated().any():
            raise RuntimeError(f"Target cache rows are not unique/aligned: {target}")
        truth = runner.raw_label_series(code, base_cfg, lookup)
        labels = truth.reindex(names)
        if labels.isna().any():
            raise RuntimeError(f"Target cache names do not align to frozen canonical labels: {target}")
        for family in sorted(set(labels) - {"__unmapped__"}):
            positions = np.flatnonzero(labels.to_numpy(str) == family)
            block = matrix[positions]
            active = np.asarray(block.getnnz(axis=1)).ravel()
            detected = int((np.asarray(block.getnnz(axis=0)).ravel() > 0).sum())
            rows.append({
                "target_species": target, "canonical_family": family,
                "target_family_cells": len(positions), "orthogroup_vocabulary": matrix.shape[1],
                "mean_active_orthogroups_per_cell": float(active.mean()),
                "median_active_orthogroups_per_cell": float(np.median(active)),
                "orthogroups_detected_in_family": detected,
                "fraction_vocabulary_detected_in_family": float(detected / matrix.shape[1]),
            })
        inputs[target] = {"summary_sha256": sha256(summary_path), "matrix_sha256": sha256(matrix_path),
                          "obs_sha256": sha256(obs_path)}
    table = pd.DataFrame(rows)
    joined = patterns.merge(table, on=["target_species", "canonical_family"], how="left", validate="one_to_one")
    associations = []
    for target, block in joined.groupby("target_species"):
        valid = block.dropna(subset=["mean_model_gain", "mean_active_orthogroups_per_cell"])
        if len(valid) >= 3:
            rho = spearmanr(valid["mean_active_orthogroups_per_cell"], valid["mean_model_gain"]).statistic
            associations.append({"target_species": target, "families": len(valid),
                                 "spearman_rho_mean_active_vs_mean_gain": float(rho)})
    out.mkdir(parents=True)
    table.to_csv(out / "target_family_orthogroup_coverage.csv", index=False)
    joined.to_csv(out / "family_gain_with_coverage.csv", index=False)
    pd.DataFrame(associations).to_csv(out / "coverage_gain_descriptive_association.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "family_rows": len(table), "joined_gain_rows": len(joined),
        "descriptive_associations": associations,
        "interpretation": "Detection is descriptive. It does not establish marker specificity, paralog ambiguity or causation.",
        "input_sha256": inputs, "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
