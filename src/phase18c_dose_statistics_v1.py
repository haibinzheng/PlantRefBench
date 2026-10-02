#!/usr/bin/env python3
"""Seed-blocked bootstrap intervals for frozen Phase 18C paired dose effects."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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


def exact_seed_bootstrap(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    boot = np.fromiter(
        (values[list(indices)].mean() for indices in itertools.product(range(n), repeat=n)),
        dtype=float, count=n ** n,
    )
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paired_path = args.paired.resolve()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    paired = pd.read_csv(paired_path)
    if len(paired) != 30 or paired.duplicated(["target_species", "seed", "budget_fraction"]).any():
        raise RuntimeError("Expected exactly 2 targets × 5 seeds × 3 budgets")
    rows = []
    for (target, fraction), block in paired.groupby(["target_species", "budget_fraction"]):
        ordered = block.sort_values("seed")
        for metric in ("macro_f1", "balanced_accuracy"):
            values = ordered[f"all_minus_nearest_{metric}"].to_numpy(dtype=float)
            low, high = exact_seed_bootstrap(values)
            rows.append({"target_species": target, "budget_fraction": fraction, "metric": metric,
                         "mean_all_minus_nearest": float(values.mean()),
                         "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                         "positive_seeds": int((values > 0).sum()), "n_seeds": len(values)})

    interactions = []
    for target, block in paired.groupby("target_species"):
        wide = block.pivot(index="seed", columns="budget_fraction",
                           values="all_minus_nearest_macro_f1")
        if list(wide.columns) != [0.25, 0.5, 1.0]:
            raise RuntimeError("Unexpected budget fractions")
        for high_fraction in (0.5, 1.0):
            values = (wide[high_fraction] - wide[0.25]).to_numpy(dtype=float)
            low, high = exact_seed_bootstrap(values)
            interactions.append({"target_species": target,
                                 "contrast": f"{high_fraction:g}-minus-0.25",
                                 "mean_change_in_all_minus_nearest_macro_f1": float(values.mean()),
                                 "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                                 "n_paired_seeds": len(values)})

    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "dose_bootstrap_ci.csv", index=False)
    pd.DataFrame(interactions).to_csv(out / "within_target_dose_interaction.csv", index=False)
    summary = {
        "run_id": "phase18c_dose_statistics_v1", "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "bootstrap_unit": "Paired sampling seed within target; these are technical resampling repeats, not biological replicates.",
        "bootstrap_method": "Exact percentile bootstrap enumerating all 5^5 within-target seed resamples.",
        "dose_interactions": interactions,
        "paired_input_sha256": sha256(paired_path), "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
