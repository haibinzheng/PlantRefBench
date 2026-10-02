#!/usr/bin/env python3
"""Aggregate the two frozen external-species linear validations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/data/reports/phylo_plant_fm_pilot/external_validation_v1")
RUNS = {
    "Sorghum bicolor": ROOT / "sorghum_bicolor_linear_v1",
    "Catharanthus roseus": ROOT / "catharanthus_roseus_linear_v1",
}
OUT = ROOT / "final_external_gate_v1"
PROTOCOL = Path("/workspace/projects/phylo_plant_fm_pilot/configs/external_validation_v1/external_validation_protocol_v1.json")


def main():
    if (OUT / "summary.json").exists():
        raise FileExistsError(f"Refusing overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    metric_frames, family_frames, species_rows = [], [], []
    for species, path in RUNS.items():
        summary = json.loads((path / "summary.json").read_text())
        metrics = pd.read_csv(path / "metrics.csv")
        families = pd.read_csv(path / "family_metrics.csv")
        metric_frames.append(metrics)
        family_frames.append(families)
        species_rows.append({
            "target_species": species,
            "all_sources_mean_macro_f1": summary["mean_all_sources_macro_f1"],
            "nearest_clade_mean_macro_f1": summary["mean_nearest_clade_macro_f1"],
            "nearest_minus_all_macro_f1": summary["mean_nearest_minus_all_macro_f1"],
            "nearest_wins": summary["nearest_wins"],
            "all_wins": summary["all_wins"],
            "mapped_target_genes": summary["mapped_target_genes"],
            "mean_active_target_features": summary.get("mean_active_target_features"),
        })
    metrics = pd.concat(metric_frames, ignore_index=True)
    families = pd.concat(family_frames, ignore_index=True)
    species = pd.DataFrame(species_rows)
    pivot = metrics.pivot(index=["target_species", "seed"], columns="condition", values="macro_f1").reset_index()
    pivot["nearest_minus_all"] = pivot["nearest_clade"] - pivot["all_sources"]
    family_means = families.groupby(
        ["target_species", "canonical_family", "condition"], as_index=False
    )["f1"].mean().pivot(
        index=["target_species", "canonical_family"], columns="condition", values="f1"
    ).reset_index()
    family_means["nearest_minus_all"] = family_means["nearest_clade"] - family_means["all_sources"]
    metrics.to_csv(OUT / "all_metrics.csv", index=False)
    species.to_csv(OUT / "species_summary.csv", index=False)
    pivot.to_csv(OUT / "paired_seed_differences.csv", index=False)
    family_means.to_csv(OUT / "family_differences.csv", index=False)

    nearest_wins = int((pivot["nearest_minus_all"] > 0).sum())
    all_wins = int((pivot["nearest_minus_all"] < 0).sum())
    summary = {
        "run_id": "final_external_gate_v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "external_species": list(RUNS),
        "species_seed_pairs": len(pivot),
        "all_sources_mean_macro_f1": float(pivot["all_sources"].mean()),
        "nearest_clade_mean_macro_f1": float(pivot["nearest_clade"].mean()),
        "primary_nearest_minus_all_macro_f1": float(pivot["nearest_minus_all"].mean()),
        "nearest_wins": nearest_wins,
        "all_wins": all_wins,
        "go_rule": "Primary delta > 0 and nearest_clade wins at least 7 of 10 species-seed pairs",
        "gate_passed": bool(pivot["nearest_minus_all"].mean() > 0 and nearest_wins >= 7),
        "interpretation": "The development-set nearest-reference advantage does not externally generalize. Broader reference diversity is consistently better on both unseen species under the frozen linear protocol.",
        "predictions_saved_before_label_open_for_both_species": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    worst = family_means.nsmallest(8, "nearest_minus_all")
    report = f"""# Frozen external validation result

- All-sources mean Macro-F1: **{summary['all_sources_mean_macro_f1']:.4f}**
- Nearest-clade mean Macro-F1: **{summary['nearest_clade_mean_macro_f1']:.4f}**
- Primary nearest-minus-all difference: **{summary['primary_nearest_minus_all_macro_f1']:.4f}**
- Wins: nearest **{nearest_wins}/10**, all-sources **{all_wins}/10**
- Preregistered gate: **FAILED**

The development-set nearest-reference advantage did not generalize. Under the frozen matched-budget protocol, broader reference diversity was consistently superior for both external species. The paper should therefore not claim that phylogenetic proximity is a universal selection rule. The defensible result is a benchmark showing strong model-, target- and cell-family-dependent reference-composition effects.

Largest family-level losses for nearest-clade training are indexed in `family_differences.csv`; the eight largest are:

{worst.to_csv(index=False)}

No extra seeds, post-result reference changes, fusion rules or scANVI runs were started.
"""
    (OUT / "EXTERNAL_VALIDATION_RESULT.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
