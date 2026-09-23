#!/usr/bin/env python3
"""Integrate frozen family-level transfer effects across three model frameworks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = frame.loc[frame["support"].gt(0)].copy()
    keys = ["seed", "target_species", "canonical_family"]
    pivot = frame.pivot(index=keys, columns="condition", values="f1").reset_index()
    pivot = pivot.dropna(subset=["all_sources", "nearest_clade"])
    pivot["all_minus_nearest"] = pivot["all_sources"] - pivot["nearest_clade"]
    pivot["model"] = model
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    out = Path(cfg["report_root"])
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")

    linear_path = Path(cfg["linear_family_metrics"])
    scanvi_path = Path(cfg["scanvi_family_metrics"])
    samap_path = Path(cfg["samap_family_differences"])
    linear = pd.read_csv(linear_path)
    linear = linear.loc[linear["budget_fraction"].eq(1.0) & linear["seed"].isin(cfg["seeds"])]
    scanvi = pd.read_csv(scanvi_path)
    scanvi = scanvi.loc[scanvi["seed"].isin(cfg["seeds"])]
    samap = pd.read_csv(samap_path)
    samap = samap.loc[samap["seed"].isin(cfg["seeds"])].copy()
    samap["target_species"] = samap["target"].map({
        "sb": "Sorghum bicolor", "cr": "Catharanthus roseus"
    })
    samap["all_minus_nearest"] = samap["all_sources"] - samap["nearest_clade"]
    samap["model"] = "SAMap"

    paired = pd.concat([
        paired_frame(linear, "orthogroup_linear"),
        paired_frame(scanvi, "scANVI"),
        samap[["seed", "target_species", "canonical_family", "all_sources",
               "nearest_clade", "all_minus_nearest", "model"]],
    ], ignore_index=True)
    if paired["target_species"].isna().any() or paired.empty:
        raise RuntimeError("Missing target or empty family comparison table")

    model_family = paired.groupby(["target_species", "canonical_family", "model"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_gain=("all_minus_nearest", "mean"),
        min_gain=("all_minus_nearest", "min"), max_gain=("all_minus_nearest", "max"),
        positive_seeds=("all_minus_nearest", lambda values: int((values > 0).sum())),
        negative_seeds=("all_minus_nearest", lambda values: int((values < 0).sum())),
    )
    model_family["stable_positive"] = model_family["positive_seeds"].eq(3)
    model_family["stable_negative"] = model_family["negative_seeds"].eq(3)

    cache_path = Path(cfg["development_cache_manifest"])
    cache = json.loads(cache_path.read_text())
    counts = []
    for item in cache["datasets"]:
        obs_path = Path(item["obs_file"])
        if sha256(obs_path) != item["obs_sha256"]:
            raise RuntimeError(f"Development observation hash mismatch: {obs_path}")
        obs = pd.read_csv(obs_path)
        for family, number in obs["canonical_family"].value_counts().items():
            counts.append({"source_species": item["species"], "canonical_family": family,
                           "reference_cells": int(number)})
    source_counts = pd.DataFrame(counts)
    support_rows = []
    for family, block in source_counts.groupby("canonical_family"):
        values = block["reference_cells"].to_numpy(dtype=float)
        probabilities = values / values.sum()
        support_rows.append({
            "canonical_family": family, "source_species_support": int(len(block)),
            "source_cells_total": int(values.sum()),
            "source_support_entropy": float(-(probabilities * np.log(probabilities)).sum()),
        })
    support = pd.DataFrame(support_rows)
    model_family = model_family.merge(support, on="canonical_family", how="left", validate="many_to_one")
    family_target = model_family.groupby(["target_species", "canonical_family"], as_index=False).agg(
        models=("model", "nunique"), stable_positive_models=("stable_positive", "sum"),
        stable_negative_models=("stable_negative", "sum"),
        mean_model_gain=("mean_gain", "mean"),
    )
    family_target["positive_candidate"] = family_target["stable_positive_models"].ge(2)
    family_target["negative_boundary"] = family_target["stable_negative_models"].ge(2)
    family_target = family_target.merge(support, on="canonical_family", how="left", validate="many_to_one")

    out.mkdir(parents=True)
    paired.to_csv(out / "paired_family_model_seed.csv", index=False)
    model_family.to_csv(out / "model_family_stability.csv", index=False)
    family_target.to_csv(out / "family_target_patterns.csv", index=False)
    source_counts.to_csv(out / "source_family_counts.csv", index=False)
    support.to_csv(out / "source_family_support.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "paired_family_model_seed_rows": int(len(paired)),
        "positive_candidates": family_target.loc[family_target["positive_candidate"],
            ["target_species", "canonical_family", "stable_positive_models", "mean_model_gain"]].to_dict(orient="records"),
        "negative_boundaries": family_target.loc[family_target["negative_boundary"],
            ["target_species", "canonical_family", "stable_negative_models", "mean_model_gain"]].to_dict(orient="records"),
        "source_support_interpretation": "Descriptive only; associations do not identify a causal mechanism.",
        "config_sha256": sha256(config_path),
        "input_sha256": {str(path): sha256(path) for path in (linear_path, scanvi_path, samap_path, cache_path)},
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
